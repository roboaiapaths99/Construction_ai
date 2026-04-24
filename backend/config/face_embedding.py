"""
Advanced face recognition with embedding generation and matching
"""
import numpy as np
import cv2
import logging
from typing import Optional, Tuple, List, Dict
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Try to import InsightFace
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("⚠️ InsightFace not installed")

# Global models
_face_analyzer = None
_cascade_classifier = None

def initialize_face_analyzer():
    """Initialize face detection and recognition models"""
    global _face_analyzer, _cascade_classifier
    
    if not INSIGHTFACE_AVAILABLE:
        logger.warning("InsightFace not available - using OpenCV fallback")
        # Initialize OpenCV cascade classifier as fallback
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            logger.info(f"Loading cascade from: {cascade_path}")
            _cascade_classifier = cv2.CascadeClassifier(cascade_path)
            if _cascade_classifier.empty():
                logger.error("❌ Cascade classifier failed to load (empty)")
                return False
            logger.info("✅ OpenCV cascade classifier initialized as fallback")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize cascade classifier: {e}")
            return False
    
    try:
        _face_analyzer = insightface.app.FaceAnalysis(
            providers=['CPUProvider'],
            allowed_modules=['detection', 'recognition']
        )
        _face_analyzer.prepare(ctx_id=-1, det_thresh=0.5, det_size=(640, 640))
        logger.info("✅ Face analyzer initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize face analyzer: {e}")
        return False

def get_face_embedding(image_data) -> Optional[np.ndarray]:
    """
    Extract face embedding from image data
    Args:
        image_data: Can be file path, PIL Image, or numpy array
    Returns:
        Face embedding as numpy array or None
    """
    if not INSIGHTFACE_AVAILABLE or _face_analyzer is None:
        # Fallback: use OpenCV to detect face and return simple embedding
        if _cascade_classifier is not None:
            try:
                # Convert different input types to numpy array
                if isinstance(image_data, str):
                    # File path
                    frame = cv2.imread(image_data)
                elif isinstance(image_data, Image.Image):
                    # PIL Image
                    frame = np.array(image_data)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif isinstance(image_data, bytes):
                    # Bytes data
                    nparr = np.frombuffer(image_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    # Assume numpy array
                    frame = image_data
                
                if frame is None:
                    return None
                
                # Convert to grayscale for detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces with more lenient parameters
                faces = _cascade_classifier.detectMultiScale(
                    gray, 
                    scaleFactor=1.05, 
                    minNeighbors=3, 
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # If no faces detected, try even more lenient parameters
                if len(faces) == 0:
                    faces = _cascade_classifier.detectMultiScale(
                        gray, 
                        scaleFactor=1.01, 
                        minNeighbors=2, 
                        minSize=(20, 20),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                
                if len(faces) == 0:
                    logger.warning("No faces detected in image (OpenCV fallback)")
                    return None
                
                # Get the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # Extract face region
                face_region = frame[y:y+h, x:x+w]
                
                # Resize to fixed size
                face_region = cv2.resize(face_region, (128, 128))
                
                # Flatten and normalize as simple embedding
                embedding = face_region.flatten().astype(np.float32)
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
            except Exception as e:
                logger.error(f"❌ Error in fallback face detection: {e}")
                return None
        return None
    
    try:
        # Convert different input types to numpy array
        if isinstance(image_data, str):
            # File path
            frame = cv2.imread(image_data)
        elif isinstance(image_data, Image.Image):
            # PIL Image
            frame = np.array(image_data)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif isinstance(image_data, bytes):
            # Bytes data
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Assume numpy array
            frame = image_data
        
        if frame is None:
            return None
        
        # Detect faces and get embeddings
        faces = _face_analyzer.get(frame)
        
        if len(faces) == 0:
            logger.warning("No faces detected in image")
            return None
        
        # Get the largest/first face embedding
        face = faces[0]
        embedding = face.embedding
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    except Exception as e:
        logger.error(f"❌ Error extracting face embedding: {e}")
        return None

def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.6) -> Tuple[float, bool]:
    """
    Compare two face embeddings
    Returns: (similarity_score, is_match)
    """
    try:
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        
        # Similarity ranges from -1 to 1, convert to 0-1
        similarity = (similarity + 1) / 2
        
        is_match = similarity >= threshold
        return float(similarity), is_match
    except Exception as e:
        logger.error(f"❌ Error comparing embeddings: {e}")
        return 0.0, False

def detect_faces_in_frame(frame: np.ndarray) -> List[Dict]:
    """
    Detect all faces in a frame and get their embeddings
    Returns list of faces with bounding boxes and embeddings
    """
    if not INSIGHTFACE_AVAILABLE or _face_analyzer is None:
        # Fallback: use OpenCV cascade classifier
        if _cascade_classifier is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = _cascade_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                results = []
                for (x, y, w, h) in faces:
                    face_region = frame[y:y+h, x:x+w]
                    face_region = cv2.resize(face_region, (128, 128))
                    embedding = face_region.flatten().astype(np.float32)
                    embedding = embedding / np.linalg.norm(embedding)
                    
                    results.append({
                        "bbox": [x, y, x+w, y+h],
                        "embedding": embedding,
                        "confidence": 0.8  # Fixed confidence for OpenCV
                    })
                
                return results
            except Exception as e:
                logger.error(f"❌ Error in fallback face detection: {e}")
                return []
        return []
    
    try:
        faces = _face_analyzer.get(frame)
        results = []
        
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.embedding / np.linalg.norm(face.embedding)
            
            results.append({
                "bbox": bbox,
                "embedding": embedding,
                "confidence": face.det_score
            })
        
        return results
    except Exception as e:
        logger.error(f"❌ Error detecting faces: {e}")
        return []

def match_face_to_workers(embedding: np.ndarray, worker_embeddings: Dict[str, np.ndarray], threshold: float = 0.6) -> Optional[Tuple[str, float]]:
    """
    Match a detected face to known workers
    Returns: (worker_id, confidence) or None
    """
    best_match = None
    best_score = 0.0
    
    for worker_id, worker_embedding in worker_embeddings.items():
        similarity, is_match = compare_embeddings(embedding, worker_embedding, threshold)
        
        if is_match and similarity > best_score:
            best_match = worker_id
            best_score = similarity
    
    if best_match:
        return best_match, best_score
    return None

# Initialize on import
initialize_face_analyzer()
