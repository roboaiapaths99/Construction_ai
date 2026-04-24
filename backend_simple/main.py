#!/usr/bin/env python3
"""
Simple AI Construction Safety Backend
Version: 2.0 - Simplified and Error-Free
"""

import os
import cv2
import base64
import json
from datetime import datetime
from typing import Optional, Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
import numpy as np
from pathlib import Path
import warnings

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# =========================================================
# CONFIGURATION
# =========================================================
APP_HOST = "0.0.0.0"
APP_PORT = 8000

# Model path - relative to backend_simple directory
MODEL_PATH = "../ai/models/yolov8n.pt"

# Violation tracking
VIOLATIONS_DIR = Path("./violations")
VIOLATIONS_DIR.mkdir(exist_ok=True)

# =========================================================
# PYTORCH FIX
# =========================================================
# Fix for PyTorch compatibility with newer versions
import pickle
torch.serialization.add_safe_globals([pickle.loads])

# Monkey patch torch.load to handle mmap parameter
_original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=pickle, **kwargs):
    # Remove weights_only if present since older models don't support it
    kwargs.pop('weights_only', None)
    # Handle mmap parameter for older PyTorch compatibility
    kwargs.pop('mmap', None)
    return _original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)

torch.load = patched_torch_load

# =========================================================
# FASTAPI APP SETUP
# =========================================================
app = FastAPI(
    title="AI Safety Monitoring System",
    version="2.0",
    description="Simple backend for construction safety detection"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# GLOBAL STATE
# =========================================================
class AppState:
    model: Optional[YOLO] = None
    camera: Optional[cv2.VideoCapture] = None
    is_running: bool = False
    last_frame: Optional[np.ndarray] = None
    detection_results: Dict = {}

app_state = AppState()

# =========================================================
# MODEL INITIALIZATION
# =========================================================
def load_model():
    """Load YOLO model"""
    try:
        model_abs_path = os.path.abspath(MODEL_PATH)
        print(f"Loading model from: {model_abs_path}")
        
        if not os.path.exists(model_abs_path):
            print(f"ERROR: Model not found at {model_abs_path}")
            return False
        
        # Load model with error handling
        try:
            app_state.model = YOLO(model_abs_path)
            print("✅ Model loaded successfully")
            return True
        except Exception as e1:
            print(f"First attempt failed: {str(e1)}")
            # Try alternative approach
            try:
                from ultralytics import YOLO as YOLO_ALT
                app_state.model = YOLO_ALT(model_abs_path)
                print("✅ Model loaded successfully (alternative method)")
                return True
            except Exception as e2:
                print(f"❌ Error loading model: {str(e2)}")
                return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting application...")
    if not load_model():
        print("⚠️ Warning: Model failed to load, but continuing...")

# =========================================================
# ROUTES - HEALTH CHECK
# =========================================================
@app.get("/health")
async def health_check():
    """Check if backend is running"""
    return {
        "status": "ok",
        "model_loaded": app_state.model is not None,
        "timestamp": datetime.now().isoformat()
    }

# =========================================================
# ROUTES - INFERENCE
# =========================================================
@app.post("/detect")
async def detect_frame(file: UploadFile = File(...)):
    """Run detection on uploaded frame"""
    try:
        if app_state.model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Read uploaded file
        contents = await file.read()
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Run inference
        results = app_state.model(frame, conf=0.4)
        
        # Process results
        detections = []
        if results and len(results) > 0:
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        detection = {
                            "confidence": float(box.conf[0]),
                            "class_id": int(box.cls[0]),
                            "class_name": result.names[int(box.cls[0])],
                            "bbox": box.xyxy[0].tolist()
                        }
                        detections.append(detection)
        
        # Store results
        app_state.detection_results = {
            "timestamp": datetime.now().isoformat(),
            "detections": detections,
            "frame_shape": frame.shape
        }
        
        return {
            "status": "success",
            "detections": detections,
            "timestamp": app_state.detection_results["timestamp"]
        }
        
    except Exception as e:
        print(f"Error in /detect: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# ROUTES - CAMERA OPERATIONS
# =========================================================
@app.get("/camera/start")
async def start_camera():
    """Start webcam"""
    try:
        if app_state.camera is not None:
            return {"status": "already_running"}
        
        app_state.camera = cv2.VideoCapture(0)
        app_state.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not app_state.camera.isOpened():
            app_state.camera = None
            raise HTTPException(status_code=500, detail="Failed to open camera")
        
        app_state.is_running = True
        return {
            "status": "success",
            "message": "Camera started"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/camera/stop")
async def stop_camera():
    """Stop webcam"""
    try:
        if app_state.camera is not None:
            app_state.camera.release()
            app_state.camera = None
        app_state.is_running = False
        return {"status": "success", "message": "Camera stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/camera/frame")
async def get_frame():
    """Get current frame from camera"""
    try:
        if app_state.camera is None:
            raise HTTPException(status_code=400, detail="Camera not started")
        
        ret, frame = app_state.camera.read()
        
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to read frame")
        
        # Encode frame to jpg
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode()
        
        return {
            "status": "success",
            "frame": img_str,
            "shape": frame.shape
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================
# ROUTES - STATUS
# =========================================================
@app.get("/status")
async def get_status():
    """Get current system status"""
    return {
        "model_loaded": app_state.model is not None,
        "camera_running": app_state.is_running,
        "last_detection": app_state.detection_results,
        "timestamp": datetime.now().isoformat()
    }

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on {APP_HOST}:{APP_PORT}")
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
