"""
Face Recognition Utilities
Uses InsightFace for face detection, alignment, and embedding generation
"""
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
import logging
from typing import Tuple, List, Dict, Optional
import os

# Try to import InsightFace
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace not installed. Some face recognition features will be limited.")

logger = logging.getLogger(__name__)

# Global face recognition model
_face_detector = None
_face_recognizer = None


def initialize_models():
    """Initialize face detection and recognition models"""
    global _face_detector, _face_recognizer
    
    if not INSIGHTFACE_AVAILABLE:
        logger.warning("InsightFace not available")
        return False
    
    try:
        # Initialize face detector
        _face_detector = insightface.app.FaceAnalysis(
            providers=['CPUProvider'],  # Use CUDAExecutionProvider for GPU
            allowed_modules=['detection', 'recognition']
        )
        _face_detector.prepare(ctx_id=-1, det_thresh=0.3, det_size=(640, 640))
        
        logger.info("✅ Face detection and recognition models initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize face models: {e}")
        return False


def get_face_detector():
    """Get or initialize face detector"""
    global _face_detector
    if _face_detector is None:
        initialize_models()
    return _face_detector


def base64_to_cv2(base64_string: str) -> Optional[np.ndarray]:
    """Convert base64 encoded image to OpenCV format"""
    try:
        # Remove data URI prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image then to OpenCV
        pil_image = Image.open(BytesIO(image_data))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return cv_image
    except Exception as e:
        logger.error(f"Error converting base64 to image: {e}")
        return None


def cv2_to_base64(cv_image: np.ndarray) -> str:
    """Convert OpenCV image to base64"""
    try:
        _, buffer = cv2.imencode('.jpg', cv_image)
        base64_string = base64.b64encode(buffer).decode()
        return base64_string
    except Exception as e:
        logger.error(f"Error converting image to base64: {e}")
        return ""


def apply_nms(boxes, overlap_threshold=0.3):
    """
    Apply Non-Maximum Suppression to remove overlapping bounding boxes
    
    Args:
        boxes: List of (x, y, w, h) bounding boxes
        overlap_threshold: IoU threshold for suppression
    
    Returns:
        List of non-overlapping boxes
    """
    if len(boxes) == 0:
        return []

    # Convert to (x1, y1, x2, y2) format
    boxes = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes])

    # Pick the box with highest area
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute area of each box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by area (largest first)
    idxs = np.argsort(area)[::-1]

    while len(idxs) > 0:
        # Pick the last box (smallest area)
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Find overlap with all other boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Compute width and height of overlap
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute IoU
        overlap = (w * h) / area[idxs[:last]]

        # Remove boxes with high overlap
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

    # Convert back to (x, y, w, h) format
    return [[boxes[i][0], boxes[i][1], boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]] for i in pick]


def detect_faces(image: np.ndarray) -> Tuple[List[dict], Optional[np.ndarray]]:
    """
    Detect faces in image using InsightFace or OpenCV Haar cascade fallback
    
    Returns:
        Tuple of (faces_list, annotated_image)
        faces_list: List of detected faces with bounding boxes and landmarks
    """
    try:
        detector = get_face_detector()
        if detector is not None:
            # Try InsightFace first
            faces = detector.get(image)

            if not faces:
                logger.debug("No faces detected with InsightFace, trying fallback...")
            else:
                logger.info(f"✅ Detected {len(faces)} face(s) in image using InsightFace")

                # Annotate image with bounding boxes
                annotated = image.copy()
                for i, face in enumerate(faces):
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(annotated, f"Face {i}", (bbox[0], bbox[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                return faces, annotated

        # Fallback to OpenCV Haar Cascade
        logger.info("Using OpenCV Haar Cascade fallback for face detection")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Load Haar cascade
        haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if haar_cascade.empty():
            logger.error("Failed to load Haar cascade classifier")
            return [], None

        # Detect faces with more lenient parameters
        faces_rects = haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More lenient (was 1.1)
            minNeighbors=3,     # More lenient (was 5)
            minSize=(20, 20),  # Smaller minimum face size (was 30, 30)
            maxSize=(500, 500) # Add max size to avoid false positives
        )

        logger.info(f"Haar cascade detected {len(faces_rects)} face(s)")

        # Apply Non-Maximum Suppression (NMS) to remove overlapping detections
        if len(faces_rects) > 1:
            faces_rects = apply_nms(faces_rects, overlap_threshold=0.3)
            logger.info(f"After NMS: {len(faces_rects)} face(s)")

        if len(faces_rects) == 0:
            logger.debug("No faces detected with Haar cascade, trying even more lenient parameters...")
            # Try with even more lenient parameters
            faces_rects = haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.01,
                minNeighbors=2,
                minSize=(15, 15)
            )
            logger.info(f"Second attempt detected {len(faces_rects)} face(s)")

            if len(faces_rects) == 0:
                logger.debug("No faces detected with Haar cascade")
                return [], image.copy()

        logger.info(f"✅ Detected {len(faces_rects)} face(s) in image using Haar cascade")

        # Convert to face info format similar to InsightFace
        faces = []
        annotated = image.copy()
        for i, (x, y, w, h) in enumerate(faces_rects):
            # Create a simple face info object
            class SimpleFaceInfo:
                def __init__(self, x, y, w, h):
                    self.bbox = np.array([x, y, x + w, y + h])
                    self.det_score = 0.8
                    self.embedding = None

            face_info = SimpleFaceInfo(x, y, w, h)
            faces.append(face_info)

            # Annotate
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, f"Face {i}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return faces, annotated

    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        return [], None


def generate_embedding(image: np.ndarray, face_info: dict) -> Optional[np.ndarray]:
    """
    Generate face embedding from image and face info
    
    Args:
        image: Input image
        face_info: Face detection info with landmarks
    
    Returns:
        Face embedding as 1D numpy array or None
    """
    try:
        detector = get_face_detector()
        if detector is not None and INSIGHTFACE_AVAILABLE:
            # InsightFace returns embedding directly in the face object
            if hasattr(face_info, 'embedding') and face_info.embedding is not None:
                embedding = face_info.embedding
                # Normalize embedding
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                return embedding.tolist()

        # Fallback: Use simple pixel-based embedding for OpenCV Haar Cascade
        logger.info("Using pixel-based embedding fallback")
        bbox = face_info.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        # Extract face region
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        # Resize to standard size
        face_resized = cv2.resize(face_crop, (64, 64))

        # Convert to grayscale and flatten
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        face_flattened = face_gray.flatten().astype(np.float32)

        # Normalize
        face_flattened = face_flattened / 255.0
        face_flattened = face_flattened / (np.linalg.norm(face_flattened) + 1e-8)

        return face_flattened.tolist()

    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def extract_face_crop(image: np.ndarray, face_info: dict, padding: int = 10) -> Optional[np.ndarray]:
    """
    Extract and align face region from image
    
    Args:
        image: Input image
        face_info: Face detection info with landmarks
        padding: Padding around face bbox
    
    Returns:
        Cropped and aligned face image or None
    """
    try:
        bbox = face_info.bbox.astype(int)
        
        # Add padding
        x1 = max(0, bbox[0] - padding)
        y1 = max(0, bbox[1] - padding)
        x2 = min(image.shape[1], bbox[2] + padding)
        y2 = min(image.shape[0], bbox[3] + padding)
        
        # Crop face
        face_crop = image[y1:y2, x1:x2].copy()
        
        if face_crop.size == 0:
            return None
        
        # Resize to standard size for consistency
        face_crop = cv2.resize(face_crop, (112, 112))
        
        return face_crop
    
    except Exception as e:
        logger.error(f"Error extracting face crop: {e}")
        return None


def calculate_face_quality(image: np.ndarray, face_info: dict) -> float:
    """
    Calculate face quality score (0-1)
    
    Checks for:
    - Image brightness
    - Contrast
    - Face size relative to image
    - Blur detection
    
    Returns:
        Quality score between 0 and 1
    """
    try:
        # Face size score
        bbox = face_info.bbox.astype(int)
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        image_area = image.shape[0] * image.shape[1]
        size_score = min(1.0, face_area / (image_area * 0.05))  # Face should be ~5% of image
        
        # Brightness score
        face_crop = extract_face_crop(image, face_info, padding=0)
        if face_crop is None:
            return 0.0
        
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 if 0.2 < brightness < 0.9 else 0.5
        
        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 100.0)
        
        # Confidence from face detector
        confidence_score = min(1.0, getattr(face_info, 'det_score', 0.5))
        
        # Weighted average
        quality = (size_score * 0.3 + brightness_score * 0.3 + 
                  blur_score * 0.2 + confidence_score * 0.2)
        
        return min(1.0, quality)
    
    except Exception as e:
        logger.error(f"Error calculating face quality: {e}")
        return 0.5


def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Returns:
        Similarity score between 0 and 1
    """
    try:
        emb1 = np.array(embedding1) if isinstance(embedding1, list) else embedding1
        emb2 = np.array(embedding2) if isinstance(embedding2, list) else embedding2
        
        # Normalize
        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2)
        
        # Convert to 0-1 range (from -1 to 1)
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    except Exception as e:
        logger.error(f"Error comparing embeddings: {e}")
        return 0.0


def find_best_match(detected_embedding: np.ndarray, employee_embeddings: Dict[str, list]) -> Tuple[Optional[str], float]:
    """
    Find best matching employee for detected face
    
    Args:
        detected_embedding: Embedding of detected face
        employee_embeddings: Dict of {employee_id: embedding_list}
    
    Returns:
        Tuple of (employee_id, confidence) or (None, 0.0)
    """
    if not employee_embeddings:
        return None, 0.0
    
    best_match = None
    best_score = 0.0
    
    for employee_id, embedding in employee_embeddings.items():
        score = compare_embeddings(detected_embedding, embedding)
        if score > best_score:
            best_score = score
            best_match = employee_id
    
    return best_match, best_score


# Initialize models on import
if INSIGHTFACE_AVAILABLE:
    try:
        initialize_models()
    except Exception as e:
        logger.warning(f"Could not initialize face models on import: {e}")
