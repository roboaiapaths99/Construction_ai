#!/usr/bin/env python3
"""
Standalone AI Worker for Face Recognition and Attendance
- Reads from MediaMTX RTSP stream
- Processes every Nth frame for performance
- Auto-reconnects on failure
- Provides health monitoring endpoints
"""

import os
import cv2
import time
import threading
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import face recognition modules
from config.worker_db import worker_db
from config.face_embedding import detect_faces_in_frame, match_face_to_workers

# =========================================================
# CONFIGURATION
# =========================================================
# MediaMTX RTSP URL for AI worker
MEDIAMTX_RTSP_URL = os.getenv("MEDIAMTX_RTSP_URL", "rtsp://localhost:8554/sitecam")
# Frame skipping: process every Nth frame (default: 3)
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))
# Reconnect delay on failure (seconds)
RECONNECT_DELAY = int(os.getenv("RECONNECT_DELAY", "5"))
# Maximum consecutive bad reads before reconnect
MAX_BAD_READS = int(os.getenv("MAX_BAD_READS", "10"))

# =========================================================
# FASTAPI APP SETUP
# =========================================================
app = FastAPI(title="AI Worker", version="2.0.0")

# =========================================================
# GLOBAL STATE
# =========================================================
worker_state = {
    "running": False,
    "camera_connected": False,
    "last_frame_time": "",
    "frames_processed": 0,
    "faces_detected": 0,
    "attendance_marked": 0,
    "bad_reads": 0,
    "reconnect_count": 0,
    "last_error": "",
    "worker_id": "ai_worker_v2"
}

worker_embeddings = {}
last_detected_workers = {}  # Track recent detections to avoid duplicate marks
state_lock = threading.Lock()

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def update_state(**kwargs):
    with state_lock:
        worker_state.update(kwargs)

def get_state():
    with state_lock:
        return dict(worker_state)

def check_and_mark_attendance(worker_id: str, confidence: float):
    """Check if worker should be marked present and mark attendance"""
    global last_detected_workers
    current_time = time.time()
    
    # Check if worker was already marked recently (1 hour)
    if worker_id in last_detected_workers:
        if current_time - last_detected_workers[worker_id] < 3600:
            return False
    
    # Mark attendance
    worker_db.mark_attendance(worker_id, event_type="check_in", confidence=confidence, detected_by="ai_worker")
    last_detected_workers[worker_id] = current_time
    logger.info(f"✅ Auto-marked attendance for worker {worker_id}")
    
    with state_lock:
        worker_state["attendance_marked"] += 1
    
    return True

def load_worker_embeddings():
    """Load all worker embeddings from database"""
    global worker_embeddings
    worker_embeddings = worker_db.get_all_embeddings()
    logger.info(f"✅ Loaded {len(worker_embeddings)} worker embeddings")

def connect_to_stream():
    """Connect to MediaMTX RTSP stream"""
    try:
        logger.info(f"Connecting to MediaMTX RTSP: {MEDIAMTX_RTSP_URL}")
        cap = cv2.VideoCapture(MEDIAMTX_RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            logger.info("✅ Connected to MediaMTX stream")
            update_state(camera_connected=True, last_error="")
            return cap
        else:
            logger.error("❌ Failed to open MediaMTX stream")
            update_state(camera_connected=False, last_error="Failed to open stream")
            return None
    except Exception as e:
        logger.error(f"❌ Error connecting to stream: {e}")
        update_state(camera_connected=False, last_error=str(e))
        return None

def process_frames():
    """Main processing loop - reads from MediaMTX and performs face detection"""
    cap = None
    frame_count = 0
    bad_read_count = 0
    
    update_state(running=True)
    
    while get_state()["running"]:
        # Connect if not connected
        if cap is None or not cap.isOpened():
            cap = connect_to_stream()
            if cap is None:
                time.sleep(RECONNECT_DELAY)
                with state_lock:
                    worker_state["reconnect_count"] += 1
                continue
        
        try:
            # Read frame
            success, frame = cap.read()
            
            if not success:
                bad_read_count += 1
                logger.warning(f"Bad read #{bad_read_count}/{MAX_BAD_READS}")
                
                if bad_read_count >= MAX_BAD_READS:
                    logger.error("Too many bad reads, reconnecting...")
                    cap.release()
                    cap = None
                    bad_read_count = 0
                    with state_lock:
                        worker_state["reconnect_count"] += 1
                    time.sleep(RECONNECT_DELAY)
                    continue
                
                time.sleep(0.1)
                continue
            
            # Reset bad read counter on success
            bad_read_count = 0
            
            # Update last frame time
            update_state(last_frame_time=datetime.utcnow().isoformat())
            
            # Process face detection only every Nth frame
            frame_count += 1
            if frame_count % FRAME_SKIP == 0 and worker_embeddings:
                try:
                    faces = detect_faces_in_frame(frame)
                    
                    for face_info in faces:
                        embedding = face_info["embedding"]
                        confidence = face_info["confidence"]
                        
                        # Match detected face to known workers
                        match = match_face_to_workers(embedding, worker_embeddings, threshold=0.95)
                        if match:
                            worker_id, similarity = match
                            logger.info(f"🎯 Matched worker {worker_id} with confidence {similarity:.2f}")
                            
                            # Auto-mark attendance
                            check_and_mark_attendance(worker_id, similarity)
                    
                    with state_lock:
                        worker_state["faces_detected"] += len(faces)
                        worker_state["frames_processed"] += 1
                        
                except Exception as e:
                    logger.error(f"Face detection error: {e}")
                    update_state(last_error=str(e))
            
            # Small sleep to prevent CPU overload
            time.sleep(0.01)
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            update_state(last_error=str(e))
            cap.release()
            cap = None
            time.sleep(RECONNECT_DELAY)
    
    # Cleanup
    if cap is not None:
        cap.release()
    update_state(running=False, camera_connected=False)
    logger.info("AI worker stopped")

# =========================================================
# API ENDPOINTS
# =========================================================

class WorkerStatus(BaseModel):
    running: bool
    camera_connected: bool
    last_frame_time: str = ""
    frames_processed: int
    faces_detected: int
    attendance_marked: int
    bad_reads: int
    reconnect_count: int
    last_error: str = ""
    worker_id: str

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    state = get_state()
    return {
        "status": "healthy" if state["camera_connected"] else "degraded",
        "running": state["running"],
        "camera_connected": state["camera_connected"]
    }

@app.get("/stats", response_model=WorkerStatus)
async def get_stats():
    """Get detailed worker statistics"""
    return get_state()

@app.post("/restart")
async def restart_worker():
    """Restart the worker"""
    state = get_state()
    if state["running"]:
        update_state(running=False)
        time.sleep(2)
    
    # Start worker in background thread
    thread = threading.Thread(target=process_frames, daemon=True)
    thread.start()
    
    return {"message": "Worker restarted"}

@app.on_event("startup")
async def startup():
    """Initialize worker on startup"""
    logger.info("Starting AI Worker...")
    load_worker_embeddings()
    
    # Start worker in background thread
    thread = threading.Thread(target=process_frames, daemon=True)
    thread.start()
    logger.info("AI Worker started")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    logger.info("Stopping AI Worker...")
    update_state(running=False)
    logger.info("AI Worker stopped")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
