#!/usr/bin/env python3
"""
Simple AI Construction Safety System Backend
- Webcam support (index 0)
- YOLO model support
- Simple API endpoints
- No complex worker conflicts
"""

import os
import cv2
import time
from datetime import datetime
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# =========================================================
APP_HOST = "0.0.0.0"
APP_PORT = 8002

# Webcam configuration
WEBCAM_INDEX = 0  # Use laptop webcam

# YOLO model path
MODEL_PATH = "../ai/models/yolov8n.pt"

# =========================================================
# FASTAPI APP SETUP
# =========================================================
app = FastAPI(title="AI Construction Safety System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# GLOBAL VARIABLES
# =========================================================
model = None
camera = None
last_detection = None
camera_status = "disconnected"

# =========================================================
# INITIALIZATION
# =========================================================
def initialize_model():
    """Initialize YOLO model"""
    global model
    try:
        print(f"Loading YOLO model from: {MODEL_PATH}")
        # Use weights_only=False for compatibility with PyTorch 2.6
        import torch
        model = YOLO(MODEL_PATH)
        print("✅ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load YOLO model: {e}")
        # Try without weights_only restriction
        try:
            import torch
            torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
            model = YOLO(MODEL_PATH)
            print("✅ YOLO model loaded successfully (with safe globals)")
            return True
        except Exception as e2:
            print(f"❌ Failed to load YOLO model (retry): {e2}")
            return False

def initialize_camera():
    """Initialize webcam"""
    global camera, camera_status
    try:
        print(f"Opening webcam at index {WEBCAM_INDEX}")
        # Try default backend first
        camera = cv2.VideoCapture(WEBCAM_INDEX)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if camera.isOpened():
            print("✅ Webcam opened successfully (default backend)")
            camera_status = "connected"
            return True
        else:
            print("❌ Failed to open webcam with default backend")
            # Try MSMF backend for Windows
            camera = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_MSMF)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if camera.isOpened():
                print("✅ Webcam opened successfully (MSMF backend)")
                camera_status = "connected"
                return True
            else:
                print("❌ Failed to open webcam with MSMF backend")
                camera_status = "disconnected"
                return False
    except Exception as e:
        print(f"❌ Webcam initialization error: {e}")
        camera_status = "disconnected"
        return False

# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Construction Safety System Backend",
        "status": "running",
        "camera_status": camera_status,
        "model_loaded": model is not None
    }

@app.get("/cameras")
async def get_cameras():
    """Get camera information"""
    return {
        "cameras": [
            {
                "id": 1,
                "name": "Laptop Webcam",
                "status": camera_status,
                "location": "Development Environment",
                "type": "webcam",
                "index": WEBCAM_INDEX,
                "model_loaded": model is not None,
                "last_detection": last_detection
            }
        ]
    }

@app.get("/stream")
async def stream():
    """MJPEG stream endpoint"""
    def generate_frames():
        global camera, last_detection
        
        if camera is None or not camera.isOpened():
            print("Camera not initialized, attempting to initialize...")
            if not initialize_camera():
                print("Failed to initialize camera for streaming")
                return
        
        try:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Failed to read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Streaming error: {e}")
    
    return Response(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/detect")
async def detect():
    """Run YOLO detection on current frame"""
    global camera, model, last_detection
    
    if model is None:
        return {"error": "Model not loaded"}
    
    if camera is None or not camera.isOpened():
        return {"error": "Camera not connected"}
    
    try:
        success, frame = camera.read()
        if not success:
            return {"error": "Failed to read frame"}
        
        # Run YOLO detection
        results = model(frame)
        
        # Parse results
        detections = []
        for result in results:
            for box in result.boxes:
                detection = {
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                }
                detections.append(detection)
        
        last_detection = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "count": len(detections),
            "detections": detections
        }
        
        return last_detection
    except Exception as e:
        return {"error": str(e)}

@app.on_event("startup")
async def startup_event():
    """Startup initialization"""
    print("=" * 60)
    print("Starting AI Construction Safety System Backend")
    print("=" * 60)
    
    # Initialize YOLO model
    initialize_model()
    
    # Initialize camera
    initialize_camera()
    
    print(f"✅ Backend started on http://{APP_HOST}:{APP_PORT}")
    print("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown cleanup"""
    global camera
    if camera is not None:
        camera.release()
        print("✅ Camera released")
    print("✅ Backend shutdown complete")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
