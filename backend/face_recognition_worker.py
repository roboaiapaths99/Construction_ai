#!/usr/bin/env python3
"""
Face Recognition Worker - SQLite Version
Monitors RTSP camera, detects faces, matches with enrolled workers, auto-marks attendance
"""

import cv2
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.worker_db import worker_db
from config.face_embedding import get_face_embedding

# Try to import InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("⚠️ InsightFace not installed. Using OpenCV fallback.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RTSP_URL = "rtsp://localhost:8554/sitecam"
CONFIDENCE_THRESHOLD = 0.70  # 70% confidence for matching
MIN_FACE_SIZE = 80  # Minimum face size in pixels
DETECTION_INTERVAL = 3  # Process every Nth frame
COOLDOWN_SECONDS = 60  # Cooldown between attendance marks for same worker

class FaceRecognitionWorker:
    """Face recognition worker for attendance marking"""
    
    def __init__(self):
        self.running = True
        self.frame_count = 0
        self.recognition_count = 0
        self.last_recognition_time = {}  # Track last recognition per worker
        
        # Initialize face recognition
        self.face_app = None
        self.face_cascade = None
        
        if INSIGHTFACE_AVAILABLE:
            try:
                logger.info("🔍 Initializing InsightFace...")
                self.face_app = FaceAnalysis(
                    name='buffalo_l',
                    providers=['CPUExecutionProvider']  # Fixed provider name
                )
                self.face_app.prepare(ctx_id=0, det_thresh=0.5)
                logger.info("✅ InsightFace initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize InsightFace: {e}")
                self.face_app = None
        
        # Fallback to OpenCV
        if self.face_app is None:
            try:
                logger.info("🔍 Initializing OpenCV Haar Cascade...")
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("✅ OpenCV Haar Cascade initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenCV: {e}")
    
    def get_enrolled_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all enrolled worker embeddings from SQLite"""
        try:
            return worker_db.get_all_embeddings()
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return {}
    
    def detect_faces(self, frame: np.ndarray):
        """Detect faces in frame"""
        if self.face_app:
            faces = self.face_app.get(frame)
            return faces
        elif self.face_cascade:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rects = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
            )
            
            # Convert to face-like objects
            class SimpleFace:
                def __init__(self, x, y, w, h):
                    self.bbox = np.array([x, y, x + w, y + h])
                    self.embedding = None
            
            return [SimpleFace(x, y, w, h) for x, y, w, h in faces_rects]
        return []
    
    def extract_embedding(self, frame: np.ndarray, face) -> Optional[np.ndarray]:
        """Extract face embedding"""
        try:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Check face size
            face_width = x2 - x1
            face_height = y2 - y1
            if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                return None
            
            # Extract face crop
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                return None
            
            # If using InsightFace and embedding already exists
            if self.face_app and hasattr(face, 'embedding') and face.embedding is not None:
                return np.array(face.embedding, dtype=np.float32)
            
            # Fallback: use get_face_embedding function
            class FaceInfo:
                def __init__(self, bbox):
                    self.bbox = bbox
            
            embedding = get_face_embedding(frame, FaceInfo(bbox))
            if embedding:
                return np.array(embedding, dtype=np.float32)
            
            return None
        except Exception as e:
            logger.debug(f"Error extracting embedding: {e}")
            return None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except:
            return 0.0
    
    def find_best_match(self, detected_embedding: np.ndarray, enrolled_embeddings: Dict) -> Tuple[Optional[str], float]:
        """Find best matching worker"""
        best_match_id = None
        best_score = 0.0
        
        for worker_id, enrolled_embedding in enrolled_embeddings.items():
            score = self.cosine_similarity(detected_embedding, enrolled_embedding)
            
            if self.frame_count % 100 == 0:
                logger.info(f"   - Worker {worker_id}: similarity {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_match_id = worker_id
        
        if self.frame_count % 100 == 0:
            logger.info(f"   Best match: {best_match_id} with score {best_score:.4f} (threshold: {CONFIDENCE_THRESHOLD})")
        
        if best_score >= CONFIDENCE_THRESHOLD:
            return best_match_id, best_score
        
        return None, best_score
    
    def mark_attendance(self, worker_id: str) -> bool:
        """Mark attendance for worker"""
        try:
            # Check cooldown
            last_time = self.last_recognition_time.get(worker_id, datetime.now() - timedelta(minutes=5))
            time_diff = (datetime.now() - last_time).total_seconds()
            
            if time_diff < COOLDOWN_SECONDS:
                logger.debug(f"Skipping attendance for {worker_id} - cooldown active ({time_diff}s)")
                return False
            
            # Mark attendance
            success = worker_db.mark_attendance(worker_id, "check_in", detected_by="face_recognition")
            
            if success:
                self.recognition_count += 1
                self.last_recognition_time[worker_id] = datetime.now()
                logger.info(f"✅ Attendance marked for {worker_id}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray):
        """Process frame for face recognition"""
        try:
            # Get enrolled embeddings
            enrolled_embeddings = self.get_enrolled_embeddings()
            if not enrolled_embeddings:
                logger.debug("No enrolled workers found")
                return
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            if self.frame_count % 100 == 0:
                logger.info(f"🔍 Frame {self.frame_count}: Detected {len(faces)} face(s)")
            
            if not faces:
                return
            
            for face in faces:
                # Extract embedding
                embedding = self.extract_embedding(frame, face)
                if embedding is None:
                    continue
                
                # Find best match
                matched_worker_id, confidence = self.find_best_match(embedding, enrolled_embeddings)
                
                if matched_worker_id:
                    logger.info(f"👤 Recognized {matched_worker_id} with confidence {confidence:.2f}")
                    
                    # Mark attendance if confidence is high enough
                    if confidence >= 0.75:
                        self.mark_attendance(matched_worker_id)
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def run(self):
        """Main recognition loop"""
        logger.info("🚀 Face Recognition Worker starting...")
        
        # Load embeddings
        enrolled_embeddings = self.get_enrolled_embeddings()
        logger.info(f"📊 Loaded {len(enrolled_embeddings)} enrolled workers")
        
        # Connect to camera
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            logger.error(f"❌ Failed to connect to RTSP: {RTSP_URL}")
            return
        
        logger.info(f"✅ Connected to RTSP: {RTSP_URL}")
        
        frame_skip = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ Failed to read frame, reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            
            self.frame_count += 1
            frame_skip += 1
            
            # Process every Nth frame
            if frame_skip >= DETECTION_INTERVAL:
                frame_skip = 0
                self.process_frame(frame)
            
            # Log stats periodically
            if self.frame_count % 300 == 0:
                logger.info(f"📊 Frames: {self.frame_count}, Recognitions: {self.recognition_count}")
            
            time.sleep(0.01)
        
        cap.release()
        logger.info("🛑 Face Recognition Worker stopped")

if __name__ == "__main__":
    worker = FaceRecognitionWorker()
    worker.run()
