"""
Simple Recognition Worker - Face Detection + Attendance Marking
Uses OpenCV for face detection when InsightFace is not available
"""

import cv2
import numpy as np
import threading
import logging
import time
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List
from bson import ObjectId
import traceback

logger = logging.getLogger(__name__)


class SimpleRecognitionWorker:
    """
    Simplified face detection and recognition worker
    - Works without InsightFace (uses OpenCV cascade classifier)
    - Detects faces in each frame
    - Matches against enrolled employee faces
    - Auto-marks attendance on face match
    """

    def __init__(self, db, rtsp_url: str, worker_id: str = "main"):
        """Initialize recognition worker"""
        print(f"🔧 Initializing SimpleRecognitionWorker {worker_id}...")
        print(f"🔧 RTSP URL: {rtsp_url}")
        logger.info(f"🔧 Initializing SimpleRecognitionWorker {worker_id}...")
        logger.info(f"🔧 RTSP URL: {rtsp_url}")

        self.db = db
        self.rtsp_url = rtsp_url
        self.worker_id = worker_id
        self.running = True
        self.thread = None
        self.frame_count = 0
        self.detection_count = 0
        self.match_count = 0
        self.last_detection_time = {}  # Track last detection per employee
        self.min_face_size = 80  # Minimum face size in pixels
        self.detection_interval = 2  # Process every Nth frame
        self.employee_embeddings = {}  # Cache employee embeddings
        self.similarity_threshold = 0.5  # Lower threshold for pixel-based embeddings

        # Initialize OpenCV face detector
        try:
            print("🔍 Loading OpenCV CascadeClassifier face detector...")
            logger.info("🔍 Loading OpenCV CascadeClassifier face detector...")
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✅ OpenCV face detector loaded successfully")
            logger.info("✅ OpenCV face detector loaded successfully")
            self.detector_ready = True
        except Exception as e:
            print(f"❌ Failed to load face detector: {str(e)}")
            logger.error(f"❌ Failed to load face detector: {str(e)}")
            self.detector_ready = False

        # Load employee embeddings
        try:
            print("🔄 Attempting to load employee embeddings...")
            logger.info("🔄 Attempting to load employee embeddings...")
            self.load_employee_embeddings()
            print(f"📊 Employee embeddings loaded: {len(self.employee_embeddings)} employees")
            logger.info(f"📊 Employee embeddings loaded: {len(self.employee_embeddings)} employees")
        except Exception as e:
            print(f"❌ Error loading employee embeddings: {str(e)}")
            logger.error(f"❌ Error loading employee embeddings: {str(e)}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")

    def load_employee_embeddings(self):
        """Load all enrolled employee face embeddings from database"""
        try:
            logger.info("📂 Loading employee embeddings from database...")
            self.employee_embeddings = {}

            # Get all face profiles
            face_profiles = list(self.db.employee_face_profiles.find({"embedding": {"$exists": True}}))

            for profile in face_profiles:
                employee_id = str(profile.get("employee_id"))
                embedding = profile.get("embedding")

                if embedding and employee_id:
                    if employee_id not in self.employee_embeddings:
                        self.employee_embeddings[employee_id] = []
                    self.employee_embeddings[employee_id].append(embedding)

            logger.info(f"✅ Loaded embeddings for {len(self.employee_embeddings)} employees")
            for emp_id, embeddings in self.employee_embeddings.items():
                logger.info(f"   Employee {emp_id}: {len(embeddings)} face profile(s)")

        except Exception as e:
            logger.error(f"❌ Error loading employee embeddings: {str(e)}")

    def generate_embedding(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Generate simple pixel-based embedding for detected face"""
        try:
            x, y, w, h = face_bbox

            # Extract face region
            face_crop = frame[y:y+h, x:x+w]
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

            return face_flattened

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None

    def compare_embeddings(self, detected_embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """Compare detected embedding against enrolled employee embeddings"""
        try:
            best_match = None
            best_score = 0.0

            for employee_id, embeddings in self.employee_embeddings.items():
                for stored_embedding in embeddings:
                    # Calculate cosine similarity
                    stored_emb = np.array(stored_embedding)
                    similarity = np.dot(detected_embedding, stored_emb)

                    if similarity > best_score:
                        best_score = similarity
                        best_match = employee_id

            if best_score > self.similarity_threshold:
                return best_match, best_score

            return None

        except Exception as e:
            logger.error(f"Error comparing embeddings: {str(e)}")
            return None

    def mark_attendance(self, employee_id: str, confidence: float) -> bool:
        """Mark attendance for matched employee"""
        try:
            # Prevent duplicate events within 2 minutes
            last_time = self.last_detection_time.get(employee_id, datetime.utcnow() - timedelta(minutes=5))
            time_diff = (datetime.utcnow() - last_time).total_seconds()

            if time_diff < 120:  # 2 minute cooldown
                logger.debug(f"Employee {employee_id} cooldown active ({time_diff}s)")
                return False

            today = datetime.utcnow().strftime("%Y-%m-%d")
            now = datetime.utcnow()

            # Check if already checked in today
            existing = self.db.attendance_logs.find_one({
                "employee_id": ObjectId(employee_id),
                "date": today
            })

            if existing:
                # Update check-out time
                self.db.attendance_logs.update_one(
                    {"_id": existing["_id"]},
                    {"$set": {"check_out_time": now}}
                )
                logger.info(f"✅ Check-out marked for employee {employee_id}")
            else:
                # Create new attendance record
                self.db.attendance_logs.insert_one({
                    "employee_id": ObjectId(employee_id),
                    "date": today,
                    "check_in_time": now,
                    "check_out_time": None,
                    "source": "rtsp_face",
                    "confidence": confidence,
                    "status": "present"
                })
                logger.info(f"✅ Check-in marked for employee {employee_id} (confidence: {confidence:.2f})")

            self.match_count += 1
            self.last_detection_time[employee_id] = datetime.utcnow()
            return True

        except Exception as e:
            logger.error(f"Error marking attendance: {str(e)}")
            return False

    def detect_faces(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect faces in frame and match against enrolled employees

        Args:
            frame: Input image

        Returns:
            Tuple of (annotated_frame, faces_detected)
        """
        try:
            if not self.detector_ready:
                return frame, 0

            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces with more lenient parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(self.min_face_size, self.min_face_size),
                maxSize=(500, 500)
            )

            for (x, y, w, h) in faces:
                self.detection_count += 1

                # Generate embedding for detected face
                detected_embedding = self.generate_embedding(frame, (x, y, w, h))

                if detected_embedding is not None and self.employee_embeddings:
                    # Match against enrolled employees
                    match_result = self.compare_embeddings(detected_embedding)

                    if match_result:
                        employee_id, confidence = match_result
                        # Mark attendance for matched employee
                        self.mark_attendance(employee_id, confidence)

                        # Draw green rectangle and label
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"Matched: {employee_id[:8]} ({confidence:.2f})", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        logger.info(f"👤 Face matched to employee {employee_id} (confidence: {confidence:.2f})")
                    else:
                        # Draw yellow rectangle for unmatched face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                else:
                    # Draw red rectangle for no embedding
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

            return frame, len(faces)

        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return frame, 0
    
    def connect_camera(self) -> Optional[cv2.VideoCapture]:
        """Connect to RTSP camera"""
        try:
            logger.info(f"🎥 Connecting to camera: {self.rtsp_url}")

            # Convert to integer if it's a webcam index
            camera_source = self.rtsp_url
            logger.info(f"🔍 Camera source before conversion: {camera_source} (type: {type(camera_source)})")
            if camera_source.isdigit():
                camera_source = int(camera_source)
                logger.info(f"🔍 Camera source after conversion: {camera_source} (type: {type(camera_source)})")

            cap = cv2.VideoCapture(camera_source)
            logger.info(f"🔍 VideoCapture created: {cap.isOpened()}")
            
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 10)  # 10 FPS for lower resource usage
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

            # Test connection
            logger.info("🔍 Attempting to read test frame...")
            ret, _ = cap.read()
            logger.info(f"🔍 Test frame read result: {ret}")
            
            if ret:
                logger.info("✅ Camera connected")
                return cap
            else:
                logger.error("❌ Camera connection failed - could not read frame")
                cap.release()
                return None

        except Exception as e:
            logger.error(f"❌ Error connecting to camera: {str(e)}")
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return None
    
    def run(self):
        """Main worker loop"""
        logger.info(f"🚀 Recognition worker {self.worker_id} starting...")

        retry_count = 0
        max_retries = 5

        while self.running:
            try:
                cap = self.connect_camera()
                if not cap:
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded")
                        break

                    wait_time = min(30, 5 * retry_count)
                    logger.warning(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                retry_count = 0
                frame_skip = 0

                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("Failed to read frame")
                        break

                    self.frame_count += 1
                    frame_skip += 1

                    if frame_skip >= self.detection_interval:
                        frame_skip = 0
                        annotated_frame, faces_detected = self.detect_faces(frame)

                        if self.frame_count % 300 == 0:
                            logger.info(f"📊 Worker {self.worker_id}: {self.frame_count} frames processed, "
                                       f"{faces_detected} faces in last frame, {self.match_count} attendance events")

                    time.sleep(0.01)

                cap.release()

            except Exception as e:
                logger.error(f"❌ Error in recognition loop: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(5)

        logger.info(f"🛑 Recognition worker {self.worker_id} stopped")
                    
    def start(self):
        """Start worker in background thread"""
        if not self.detector_ready:
            logger.error("Detector not ready")
            return False
        
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run, daemon=True, name=f"RecognitionWorker-{self.worker_id}")
            self.thread.start()
            logger.info(f"✅ Recognition worker {self.worker_id} started")
            return True
        
        return False
    
    def stop(self):
        """Stop worker"""
        logger.info(f"Stopping recognition worker {self.worker_id}...")
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            logger.info(f"✅ Recognition worker stopped")
    
    def get_stats(self) -> Dict:
        """Get worker statistics"""
        return {
            "worker_id": self.worker_id,
            "running": self.running,
            "frames_processed": self.frame_count,
            "detections": self.detection_count,
            "detector_ready": self.detector_ready
        }
