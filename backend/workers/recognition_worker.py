"""
Recognition Worker - Real-time Face Detection and Attendance Marking
Monitors RTSP camera, detects faces, matches with enrolled employees, auto-marks attendance
"""

import cv2
import numpy as np
import threading
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from bson import ObjectId
import traceback

logger = logging.getLogger(__name__)


class FaceRecognitionWorker:
    """
    Real-time face detection and recognition worker
    - Connects to RTSP camera
    - Detects faces in each frame
    - Matches faces with enrolled employee embeddings
    - Auto-marks attendance with high confidence matches
    """
    
    def __init__(self, db, rtsp_url: str, worker_id: str = "main"):
        """
        Initialize recognition worker
        
        Args:
            db: MongoDB database connection
            rtsp_url: RTSP camera URL
            worker_id: Unique worker identifier
        """
        self.db = db
        self.rtsp_url = rtsp_url
        self.worker_id = worker_id
        self.running = True
        self.thread = None
        self.frame_count = 0
        self.recognition_count = 0
        self.last_recognition_time = {}  # Track last recognition per employee
        self.confidence_threshold = 0.70  # 70% confidence for matching
        self.min_face_size = 80  # Minimum face size in pixels
        self.detection_interval = 2  # Process every Nth frame
        
        # Initialize face recognition engine
        try:
            from insightface.app import FaceAnalysis
            logger.info("🔍 Initializing InsightFace face recognition engine...")

            # Use CPU provider for compatibility
            self.face_app = FaceAnalysis(
                name='buffalo_l',
                root='./face_models',
                providers=['CPUProvider']
            )
            self.face_app.prepare(ctx_id=0, det_thresh=0.5)
            logger.info("✅ InsightFace initialized successfully")
            self.face_engine_ready = True
        except Exception as e:
            logger.error(f"❌ Failed to initialize InsightFace: {str(e)}")
            logger.error(traceback.format_exc())
            self.face_engine_ready = False

        # Fallback to OpenCV Haar Cascade
        if not self.face_engine_ready:
            try:
                logger.info("🔍 Initializing OpenCV Haar Cascade fallback...")
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                self.opencv_fallback = True
                logger.info("✅ OpenCV Haar Cascade fallback initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize OpenCV fallback: {str(e)}")
                self.opencv_fallback = False
    
    def get_enrolled_embeddings(self) -> Dict[str, Dict]:
        """
        Retrieve all enrolled employee face embeddings from database
        
        Returns:
            Dict mapping employee_id to employee info with embeddings
        """
        try:
            profiles = list(self.db.employee_face_profiles.find(
                {"embedding": {"$exists": True}},
                {
                    "_id": 1,
                    "employee_id": 1,
                    "embedding": 1,
                    "quality_score": 1,
                    "is_primary": 1
                }
            ))
            
            employees = {}
            for profile in profiles:
                emp_id = str(profile["employee_id"])
                if emp_id not in employees:
                    # Get employee info
                    employee = self.db.employees.find_one({"_id": profile["employee_id"]})
                    if employee:
                        employees[emp_id] = {
                            "name": employee.get("name", "Unknown"),
                            "employee_code": employee.get("employee_code", "N/A"),
                            "embeddings": []
                        }
                
                if emp_id in employees:
                    employees[emp_id]["embeddings"].append({
                        "embedding": np.array(profile["embedding"], dtype=np.float32),
                        "quality": profile.get("quality_score", 0),
                        "is_primary": profile.get("is_primary", False)
                    })
            
            return employees
        
        except Exception as e:
            logger.error(f"Error retrieving enrolled embeddings: {str(e)}")
            return {}
    
    def extract_face_embedding(self, image: np.ndarray, bbox: Tuple) -> Optional[np.ndarray]:
        """
        Extract face embedding from image using detected bounding box
        
        Args:
            image: Input image (BGR format from OpenCV)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            512-dimensional embedding vector or None if face too small
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            face_width = x2 - x1
            face_height = y2 - y1
            
            # Filter small faces
            if face_width < self.min_face_size or face_height < self.min_face_size:
                return None
            
            face_crop = image[y1:y2, x1:x2]
            
            # Get embedding using InsightFace
            faces = self.face_app.get(face_crop)
            if faces:
                embedding = faces[0].embedding
                return np.array(embedding, dtype=np.float32)
            
            return None
        
        except Exception as e:
            logger.debug(f"Error extracting embedding: {str(e)}")
            return None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1, vec2: 1D numpy arrays
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except:
            return 0.0
    
    def find_best_match(self, detected_embedding: np.ndarray, enrolled_employees: Dict) -> Tuple[Optional[str], float]:
        """
        Find best matching enrolled employee for detected face
        
        Args:
            detected_embedding: Embedding of detected face
            enrolled_employees: Dict of enrolled employees with embeddings
            
        Returns:
            Tuple of (employee_id, confidence_score) or (None, 0) if no good match
        """
        best_match_id = None
        best_score = 0.0
        
        for emp_id, emp_data in enrolled_employees.items():
            for enrollment in emp_data["embeddings"]:
                enrolled_embedding = enrollment["embedding"]
                score = self.cosine_similarity(detected_embedding, enrolled_embedding)
                
                if score > best_score:
                    best_score = score
                    best_match_id = emp_id
        
        # Only return match if confidence exceeds threshold
        if best_score >= self.confidence_threshold:
            return best_match_id, best_score
        
        return None, best_score
    
    def mark_attendance(self, employee_id: str, event_type: str = "check_in") -> bool:
        """
        Mark attendance for employee (auto check-in/check-out)
        
        Args:
            employee_id: MongoDB ObjectId of employee
            event_type: "check_in" or "check_out"
            
        Returns:
            True if marked successfully, False otherwise
        """
        try:
            # Prevent duplicate check-ins within 1 minute
            last_time = self.last_recognition_time.get(employee_id, datetime.utcnow() - timedelta(minutes=5))
            time_diff = (datetime.utcnow() - last_time).total_seconds()
            
            if time_diff < 60:  # 60 second cooldown
                logger.debug(f"Skipping {event_type} for {employee_id} - cooldown active ({time_diff}s)")
                return False
            
            today = datetime.utcnow().strftime("%Y-%m-%d")
            emp_obj_id = ObjectId(employee_id)
            
            # Find or create attendance log for today
            log = self.db.attendance_logs.find_one({
                "employee_id": emp_obj_id,
                "date": today
            })
            
            if event_type == "check_in":
                if not log:
                    # Create new attendance log with check-in
                    result = self.db.attendance_logs.insert_one({
                        "employee_id": emp_obj_id,
                        "date": today,
                        "check_in_time": datetime.utcnow(),
                        "check_out_time": None,
                        "status": "incomplete",
                        "marked_by": "face_recognition",
                        "confidence": 0.0,
                        "worker_id": self.worker_id
                    })
                    logger.info(f"✅ Auto check-in: {employee_id}")
                    self.recognition_count += 1
                    self.last_recognition_time[employee_id] = datetime.utcnow()
                    return True
                elif log.get("check_in_time") and not log.get("check_out_time"):
                    logger.debug(f"Employee {employee_id} already checked in today")
                    return False
            
            elif event_type == "check_out":
                if log and log.get("check_in_time") and not log.get("check_out_time"):
                    # Update with check-out
                    self.db.attendance_logs.update_one(
                        {"_id": log["_id"]},
                        {"$set": {
                            "check_out_time": datetime.utcnow(),
                            "status": "present",
                            "marked_by": "face_recognition",
                            "worker_id": self.worker_id
                        }}
                    )
                    logger.info(f"✅ Auto check-out: {employee_id}")
                    self.recognition_count += 1
                    self.last_recognition_time[employee_id] = datetime.utcnow()
                    return True
                else:
                    logger.debug(f"Employee {employee_id} has no active check-in")
                    return False
            
            return False
        
        except Exception as e:
            logger.error(f"Error marking attendance: {str(e)}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Process single frame for face detection and recognition

        Args:
            frame: Input image in BGR format

        Returns:
            Tuple of (annotated_frame, faces_detected)
        """
        try:
            # Get enrolled employees
            enrolled_employees = self.get_enrolled_embeddings()
            if not enrolled_employees:
                logger.debug("No enrolled employees found")
                return frame, 0

            # Use InsightFace if available, otherwise OpenCV fallback
            if self.face_engine_ready:
                # Detect faces in frame
                faces = self.face_app.get(frame)
            elif getattr(self, 'opencv_fallback', False):
                # Use OpenCV Haar Cascade for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_rects = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=3,
                    minSize=(self.min_face_size, self.min_face_size),
                    maxSize=(500, 500)
                )

                # Convert to face objects similar to InsightFace format
                class SimpleFace:
                    def __init__(self, x, y, w, h):
                        self.bbox = np.array([x, y, x + w, y + h])

                faces = [SimpleFace(x, y, w, h) for x, y, w, h in faces_rects]
            else:
                return frame, 0
            
            for face in faces:
                bbox = face.bbox
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                # Check face size
                face_width = x2 - x1
                if face_width < self.min_face_size:
                    continue
                
                # Extract embedding
                embedding = self.extract_face_embedding(frame, bbox)
                if embedding is None:
                    continue
                
                # Find best match
                matched_employee_id, confidence = self.find_best_match(embedding, enrolled_employees)
                
                # Draw on frame
                if matched_employee_id:
                    employee_id_str = str(matched_employee_id)
                    emp_name = enrolled_employees[employee_id_str]["name"]
                    
                    # Green box for recognized face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{emp_name} ({confidence:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Auto-mark attendance
                    if confidence >= 0.75:  # Higher threshold for auto-marking
                        self.mark_attendance(employee_id_str, "check_in")
                else:
                    # Yellow box for unrecognized face
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "Unknown", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            return frame, len(faces)
        
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            logger.error(traceback.format_exc())
            return frame, 0
    
    def connect_camera(self, timeout: int = 5) -> Optional[cv2.VideoCapture]:
        """
        Establish connection to RTSP camera
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            VideoCapture object or None if connection failed
        """
        try:
            logger.info(f"🎥 Connecting to camera: {self.rtsp_url}")
            
            cap = cv2.VideoCapture(self.rtsp_url)
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
            cap.set(cv2.CAP_PROP_FPS, 15)  # Max 15 FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Test connection
            ret, _ = cap.read()
            if ret:
                logger.info("✅ Camera connected successfully")
                return cap
            else:
                logger.error("❌ Camera connection failed - cannot read frame")
                cap.release()
                return None
        
        except Exception as e:
            logger.error(f"❌ Error connecting to camera: {str(e)}")
            return None
    
    def run(self):
        """
        Main recognition worker loop - runs continuously in background thread
        """
        try:
            logger.info(f"🚀 Recognition worker {self.worker_id} starting...")
            logger.info(f"🔍 Face engine ready: {self.face_engine_ready}")
            logger.info(f"🔍 OpenCV fallback: {getattr(self, 'opencv_fallback', False)}")

            # Load employee embeddings
            enrolled_employees = self.get_enrolled_embeddings()
            logger.info(f"📊 Loaded {len(enrolled_employees)} enrolled employees for recognition")

            if not enrolled_employees:
                logger.warning("⚠️ No enrolled employees found. Recognition will not mark attendance.")
            else:
                for emp_id, emp_info in enrolled_employees.items():
                    logger.info(f"   - {emp_info['name']} ({emp_id}): {len(emp_info['embeddings'])} face profile(s)")

            retry_count = 0
            max_retries = 5

            while self.running:
                try:
                    # Try to connect to camera
                    cap = self.connect_camera()
                    if not cap:
                        retry_count += 1
                        if retry_count > max_retries:
                            logger.error(f"❌ Max retries ({max_retries}) exceeded. Worker stopping.")
                            break

                        wait_time = min(30, 5 * retry_count)  # Exponential backoff
                        logger.warning(f"⏳ Retrying in {wait_time}s... (attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)
                        continue

                    retry_count = 0
                    frame_skip = 0

                    # Read frames continuously
                    while self.running and cap.isOpened():
                        ret, frame = cap.read()

                        if not ret:
                            logger.warning("⚠️ Failed to read frame from camera")
                            break

                        self.frame_count += 1
                        frame_skip += 1

                        # Process every Nth frame to reduce CPU usage
                        if frame_skip >= self.detection_interval:
                            frame_skip = 0
                            annotated_frame, faces_detected = self.process_frame(frame)

                            if self.frame_count % 300 == 0:  # Log every 300 frames (~20s at 15 FPS)
                                logger.info(f"📊 Worker {self.worker_id}: {self.frame_count} frames processed, "
                                           f"{faces_detected} faces in last frame, {self.recognition_count} attendance events")

                        # Small delay to prevent CPU spinning
                        time.sleep(0.01)

                    cap.release()

                except Exception as e:
                    logger.error(f"❌ Error in recognition loop: {str(e)}")
                    logger.error(traceback.format_exc())
                    time.sleep(5)

            logger.info(f"🛑 Recognition worker {self.worker_id} stopped")

        except Exception as e:
            logger.error(f"❌ Error in recognition worker: {str(e)}")
            logger.error(traceback.format_exc())

    def start(self):
        """Start recognition worker in background thread"""
        logger.info(f"🔧 Starting recognition worker {self.worker_id}...")
        logger.info(f"🔍 Face engine ready: {self.face_engine_ready}")
        logger.info(f"🔍 OpenCV fallback: {getattr(self, 'opencv_fallback', False)}")

        if not self.face_engine_ready and not getattr(self, 'opencv_fallback', False):
            logger.error("❌ Face recognition engine not ready. Cannot start worker.")
            return False

        if self.thread is None or not self.thread.is_alive():
            logger.info(f"🧵 Creating thread for recognition worker {self.worker_id}...")
            self.thread = threading.Thread(target=self.run, daemon=True, name=f"RecognitionWorker-{self.worker_id}")
            self.thread.start()
            logger.info(f"✅ Recognition worker {self.worker_id} started (thread: {self.thread.name}, alive: {self.thread.is_alive()})")
            return True

        logger.warning(f"⚠️ Recognition worker {self.worker_id} thread already running")
        return False
    
    def stop(self):
        """Stop recognition worker"""
        logger.info(f"🛑 Stopping recognition worker {self.worker_id}...")
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
            logger.info(f"✅ Recognition worker {self.worker_id} stopped")
    
    def get_stats(self) -> Dict:
        """Get worker statistics"""
        return {
            "worker_id": self.worker_id,
            "running": self.running,
            "frames_processed": self.frame_count,
            "recognition_events": self.recognition_count,
            "face_engine_ready": self.face_engine_ready,
            "uptime_frames": self.frame_count,
            "approximate_uptime_seconds": self.frame_count / 15  # Assuming 15 FPS
        }
