#!/usr/bin/env python3
"""
AI Construction Safety System Backend - API Only
- Handles REST APIs for attendance, employees, incidents
- No direct camera access (handled by MediaMTX + AI Worker)
- Database operations and business logic
"""

import os
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import uuid

# Import worker database and face embedding modules
from config.worker_db import worker_db
from config.face_embedding import get_face_embedding

# =========================================================
# CONFIGURATION
# =========================================================
APP_HOST = "0.0.0.0"
APP_PORT = 8080

# MediaMTX configuration
MEDIAMTX_RTSP_URL = os.getenv("MEDIAMTX_RTSP_URL", "rtsp://localhost:8554/sitecam")
MEDIAMTX_WEBRTC_URL = os.getenv("MEDIAMTX_WEBRTC_URL", "http://localhost:8889/sitecam")
MEDIAMTX_HLS_URL = os.getenv("MEDIAMTX_HLS_URL", "http://localhost:8888/sitecam/index.m3u8")

# AI Worker configuration
AI_WORKER_URL = os.getenv("AI_WORKER_URL", "http://localhost:8001")

ENROLLMENT_DIR = os.path.join(os.path.dirname(__file__), "data", "images", "enrollments")
MAX_ENROLLMENT_IMAGE_SIZE = 5 * 1024 * 1024

# =========================================================
# FASTAPI APP SETUP
# =========================================================
app = FastAPI(title="AI Construction Safety System", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3002", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# GLOBAL VARIABLES
# =========================================================
# No camera or model variables - handled by MediaMTX + AI Worker

# Face recognition tracking
worker_embeddings = {}  # Cached embeddings
last_detected_workers = {}  # Track recent detections to avoid duplicate marks (worker_id: last_timestamp)
last_detection = {}  # Track last detection data

# Camera configuration
camera_config = {
    "camera_source": "custom",  # 0=default webcam, 1=secondary, custom=IP camera
    "camera_type": "ip_camera",
    "camera_status": "disconnected",
    "custom_url": "rtsp://192.168.1.16:554/11"  # For IP camera RTSP/HTTP URLs
}

# =========================================================
# INITIALIZATION
# =========================================================
def load_worker_embeddings():
    """Load all worker embeddings from database"""
    worker_embeddings = worker_db.get_all_embeddings()
    print(f"✅ Loaded {len(worker_embeddings)} worker embeddings")
    return worker_embeddings

# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Construction Safety System Backend - API Only",
        "status": "running",
        "architecture": "MediaMTX + AI Worker + Backend API",
        "mediamtx_webrtc": MEDIAMTX_WEBRTC_URL,
        "mediamtx_hls": MEDIAMTX_HLS_URL,
        "ai_worker": AI_WORKER_URL
    }

@app.get("/cameras")
async def get_cameras():
    """Get camera information from MediaMTX"""
    return {
        "cameras": [
            {
                "id": 1,
                "name": "Site Camera",
                "status": "active",
                "location": "Construction Site",
                "type": "ip_camera",
                "stream_urls": {
                    "webrtc": f"{MEDIAMTX_WEBRTC_URL}/whep",
                    "hls": MEDIAMTX_HLS_URL,
                    "rtsp": MEDIAMTX_RTSP_URL
                }
            }
        ]
    }

@app.get("/api/recognition/stats")
async def get_recognition_stats():
    """Get recognition worker statistics from AI Worker"""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{AI_WORKER_URL}/stats", timeout=5.0)
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "message": "Failed to fetch AI worker stats"
                }
    except Exception as e:
        return {
            "status": "error",
            "message": f"AI worker not reachable: {str(e)}"
        }

@app.post("/api/recognition/restart")
async def restart_recognition_worker():
    """Restart the AI worker"""
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{AI_WORKER_URL}/restart", timeout=10.0)
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "message": "Failed to restart AI worker"
                }
    except Exception as e:
        return {
            "success": False,
            "message": f"AI worker not reachable: {str(e)}"
        }

# =========================================================
# ROUTES - DASHBOARD DATA
# =========================================================
@app.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        return {
            "total_violations": 0,
            "total_alerts": 0,
            "active_cameras": 1,
            "detection_status": "active",
            "uptime_seconds": int(time.time()),
            "last_detection": last_detection or {}
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/violations")
async def get_violations():
    """Get all violations"""
    try:
        return {
            "violations": [],
            "total": 0,
            "page": 1,
            "total_pages": 0
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/alerts")
async def get_alerts():
    """Get all alerts"""
    try:
        return {
            "alerts": [],
            "total": 0,
            "page": 1,
            "total_pages": 0
        }
    except Exception as e:
        return {"error": str(e)}

# =========================================================
# ROUTES - CAMERAS
# =========================================================
@app.get("/cameras")
async def get_cameras():
    """Get all available cameras"""
    try:
        return {
            "cameras": [
                {
                    "id": 1,
                    "name": "Laptop Webcam",
                    "status": camera_status if camera is not None else "disconnected",
                    "stream_url": "http://0.0.0.0:8002/stream"
                }
            ]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/cameras/{camera_id}")
async def get_camera_detail(camera_id: int):
    """Get specific camera details"""
    try:
        return {
            "id": camera_id,
            "name": "Laptop Webcam",
            "status": camera_status if camera is not None else "disconnected",
            "stream_url": "http://0.0.0.0:8002/stream"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/cameras/{camera_id}/detection")
async def get_camera_detection(camera_id: int):
    """Get latest detection for camera"""
    try:
        return last_detection or {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "count": 0,
            "detections": []
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/cameras/{camera_id}/detections/latest")
async def get_camera_detections_latest(camera_id: int):
    """Get latest detections for camera"""
    try:
        return last_detection or {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "count": 0,
            "detections": []
        }
    except Exception as e:
        return {"error": str(e)}

# =========================================================
# ROUTES - CAMERA CONFIGURATION
# =========================================================
@app.get("/api/camera/config")
async def get_camera_config():
    """Get current camera configuration"""
    return {
        "camera_source": camera_config["camera_source"],
        "camera_type": camera_config["camera_type"],
        "camera_status": camera_config["camera_status"],
        "custom_url": camera_config.get("custom_url", "")
    }

@app.post("/api/camera/configure")
async def configure_camera(source: str = Form(...), custom_url: str = Form(None)):
    """Configure camera source"""
    global camera_config
    
    try:
        # Validate source
        valid_sources = ["0", "1", "custom"]
        if source not in valid_sources:
            raise HTTPException(status_code=400, detail="Invalid camera source")
        
        # Update camera configuration
        camera_config["camera_source"] = source
        
        if source == "custom" and custom_url:
            camera_config["custom_url"] = custom_url
            camera_config["camera_type"] = "ip_camera"
            
            # Try to determine protocol and test connection
            if custom_url.startswith("rtsp://"):
                camera_config["camera_type"] = "ip_camera_rtsp"
            elif custom_url.startswith("http"):
                camera_config["camera_type"] = "ip_camera_http"
            
            # Test if we can connect to the URL
            camera_config["camera_status"] = await test_camera_connection(custom_url)
            
        elif source in ["0", "1"]:
            camera_config["camera_type"] = "webcam"
            camera_config["custom_url"] = ""
            # Webcam is considered connected if we accept the configuration
            camera_config["camera_status"] = "connected"
        
        return {
            "success": True,
            "message": "Camera configuration updated",
            "camera_source": camera_config["camera_source"],
            "camera_type": camera_config["camera_type"],
            "camera_status": camera_config["camera_status"],
            "custom_url": camera_config.get("custom_url", "")
        }
    except Exception as e:
        camera_config["camera_status"] = "disconnected"
        return {
            "success": False,
            "message": f"Failed to configure camera: {str(e)}",
            "camera_status": "disconnected"
        }

async def test_camera_connection(url: str) -> str:
    """Test if camera URL is reachable"""
    import httpx
    try:
        async with httpx.AsyncClient(verify=False) as client:
            # Set a short timeout for testing
            response = await client.get(url, timeout=3.0)
            # Just check if we get any response (even 401 means the camera is reachable)
            if response.status_code < 500:
                return "connected"
            else:
                return "disconnected"
    except Exception as e:
        print(f"Camera connection test failed: {str(e)}")
        return "disconnected"

# =========================================================
# ROUTES - ATTENDANCE
# =========================================================
@app.get("/api/attendance/today")
async def get_attendance_today():
    """Get today's attendance records"""
    try:
        records = worker_db.get_today_attendance()
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "records": records,
            "total": len(records)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/attendance/employees")
async def get_attendance_employees():
    """Get all enrolled workers"""
    try:
        employees = worker_db.get_all_workers()
        return {
            "employees": employees,
            "total": len(employees)
        }
    except Exception as e:
        return {"error": str(e)}

@app.delete("/api/attendance/employees/{worker_id}")
async def delete_employee(worker_id: str):
    """Delete a worker and all associated data"""
    try:
        if not worker_db.worker_exists(worker_id):
            return JSONResponse(status_code=404, content={"success": False, "error": "Worker not found"})
        
        success = worker_db.delete_worker(worker_id)
        
        if success:
            # Reload embeddings cache
            global worker_embeddings
            worker_embeddings = load_worker_embeddings()
            
            return {"success": True, "message": f"Worker {worker_id} deleted successfully"}
        else:
            return JSONResponse(status_code=500, content={"success": False, "error": "Failed to delete worker"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.post("/api/attendance/enroll")
async def enroll_employee(
    worker_id: str = Form(None),
    name: str = Form(...),
    email: str = Form(None),
    phone: str = Form(None),
    image: UploadFile = File(...)
):
    """
    Enroll a new worker with face image
    Extracts face embedding from uploaded image
    """
    try:
        if not name or not name.strip():
            return JSONResponse(status_code=400, content={"success": False, "error": "Worker name is required"})

        if image.content_type and not image.content_type.startswith("image/"):
            return JSONResponse(status_code=400, content={"success": False, "error": "Please upload a valid image file"})

        worker_id = (worker_id or "").strip() or f"worker_{uuid.uuid4().hex[:10]}"

        if worker_db.worker_exists(worker_id):
            return JSONResponse(status_code=409, content={"success": False, "error": f"Worker ID already exists: {worker_id}"})

        image_data = await image.read()
        if not image_data:
            return JSONResponse(status_code=400, content={"success": False, "error": "Uploaded image is empty"})

        if len(image_data) > MAX_ENROLLMENT_IMAGE_SIZE:
            return JSONResponse(status_code=400, content={"success": False, "error": "Image must be smaller than 5 MB"})

        os.makedirs(ENROLLMENT_DIR, exist_ok=True)
        extension = os.path.splitext(image.filename or "")[1].lower() or ".jpg"
        saved_filename = f"{worker_id}_{int(time.time())}{extension}"
        saved_path = os.path.join(ENROLLMENT_DIR, saved_filename)

        with open(saved_path, "wb") as image_file:
            image_file.write(image_data)

        # Extract face embedding
        embedding = get_face_embedding(image_data)
        if embedding is None:
            if os.path.exists(saved_path):
                os.remove(saved_path)
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Could not detect a clear face in the uploaded photo."}
            )
        
        # Add worker to database
        if not worker_db.add_worker(worker_id, name.strip(), email, phone):
            if os.path.exists(saved_path):
                os.remove(saved_path)
            return JSONResponse(status_code=500, content={"success": False, "error": "Failed to save worker profile"})
        
        # Store face embedding
        if not worker_db.store_embedding(worker_id, embedding):
            return JSONResponse(status_code=500, content={"success": False, "error": "Failed to store face embedding"})
        
        # Reload embeddings in memory
        global worker_embeddings
        worker_embeddings[worker_id] = embedding
        
        return {
            "success": True,
            "message": "Worker enrolled successfully",
            "worker_id": worker_id,
            "name": name.strip(),
            "image_path": saved_path
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/api/attendance/{employee_id}")
async def get_employee_attendance(employee_id: str):
    """Get attendance history for a specific worker (last 30 days)"""
    try:
        records = worker_db.get_worker_attendance(employee_id, days=30)
        return {
            "employee_id": employee_id,
            "records": records,
            "total": len(records)
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/attendance/mark")
async def mark_attendance(data: dict):
    """Manually mark attendance (check-in/check-out)"""
    try:
        worker_id = (data.get("employee_id") or data.get("worker_id") or "").strip()
        event_type = (data.get("event_type") or "check_in").strip()

        if not worker_id:
            return JSONResponse(status_code=400, content={"success": False, "error": "Worker ID is required"})

        if event_type not in {"check_in", "check_out"}:
            return JSONResponse(status_code=400, content={"success": False, "error": "event_type must be check_in or check_out"})

        if not worker_db.worker_exists(worker_id):
            return JSONResponse(status_code=404, content={"success": False, "error": f"Worker not found: {worker_id}"})

        success = worker_db.mark_attendance(worker_id, event_type=event_type, detected_by="manual")
        if not success:
            return JSONResponse(
                status_code=409,
                content={"success": False, "error": f"Could not mark {event_type.replace('_', ' ')} for {worker_id}"}
            )
        
        return {
            "success": True,
            "message": f"Attendance marked: {event_type}",
            "worker_id": worker_id
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

# =========================================================
# ROUTES - WORKERS
# =========================================================
@app.get("/workers")
async def get_workers():
    """Get all active workers"""
    try:
        workers = worker_db.get_all_workers()
        return {
            "workers": workers,
            "total": len(workers)
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/workers")
async def create_worker(data: dict):
    """Create a new worker profile"""
    try:
        worker_id = data.get("worker_id", str(uuid.uuid4())[:8])
        name = data.get("name")
        email = data.get("email")
        phone = data.get("phone")
        
        worker_db.add_worker(worker_id, name, email, phone)
        
        return {
            "success": True,
            "worker_id": worker_id,
            "message": "Worker created successfully"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.on_event("startup")
async def startup_event():
    """Startup initialization"""
    print("=" * 60)
    print("Starting AI Construction Safety System Backend - API Only")
    print("=" * 60)

    os.makedirs(ENROLLMENT_DIR, exist_ok=True)
    
    # Load worker embeddings
    load_worker_embeddings()
    
    print(f"✅ Backend started on http://{APP_HOST}:{APP_PORT}")
    print(f"✅ MediaMTX WebRTC: {MEDIAMTX_WEBRTC_URL}")
    print(f"✅ MediaMTX HLS: {MEDIAMTX_HLS_URL}")
    print(f"✅ AI Worker: {AI_WORKER_URL}")
    print("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown cleanup"""
    print("✅ Backend shutdown complete")

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
