from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymongo.errors import PyMongoError
import os
import base64
import io
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from bson import ObjectId

from PIL import Image
import numpy as np
import cv2
import torch

# Lazy import for YOLO
YOLO = None
model = None

# Import configuration modules
from config.mongodb import MongoDBConfig, initialize_mongodb
from config.settings import Settings, settings
from config.ai_config import AIConfig, ai_config
from routers.attendance import router as attendance_router
from workers.simple_recognition_worker import SimpleRecognitionWorker

# =========================================================
# ENV / APP SETTINGS
# =========================================================
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8002"))

# Camera source selection: "sitecam" (IP camera) or "webcam" (laptop webcam)
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "sitecam")
CAMERA_RTSP_URL = "0"  # Use webcam (index 0)
MEDIAMTX_RTSP_URL = os.getenv("MEDIAMTX_RTSP_URL", CAMERA_RTSP_URL)

# Set these to the URLs your frontend should use.
# Adjust based on your MediaMTX deployment/version and reverse proxy setup.
MEDIAMTX_WEBRTC_URL = os.getenv("MEDIAMTX_WEBRTC_URL", f"http://127.0.0.1:8889/{CAMERA_SOURCE}/")
MEDIAMTX_HLS_URL = os.getenv("MEDIAMTX_HLS_URL", f"http://127.0.0.1:8888/{CAMERA_SOURCE}/index.m3u8")

AI_ENABLED = os.getenv("AI_ENABLED", "false").lower() == "true"
AI_FRAME_SKIP = int(os.getenv("AI_FRAME_SKIP", "10"))      # detect every Nth frame
AI_RESIZE_WIDTH = int(os.getenv("AI_RESIZE_WIDTH", "640"))
AI_RESIZE_HEIGHT = int(os.getenv("AI_RESIZE_HEIGHT", "360"))
AI_RECONNECT_DELAY_SEC = float(os.getenv("AI_RECONNECT_DELAY_SEC", "1.0"))

RECOGNITION_ENABLED = os.getenv("RECOGNITION_ENABLED", "false").lower() == "true"

VIOLATIONS_DIR = "data/images/violations"
os.makedirs(VIOLATIONS_DIR, exist_ok=True)

ENROLLMENT_DIR = "data/images/enrollments"
os.makedirs(ENROLLMENT_DIR, exist_ok=True)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI Construction Safety Monitoring System API"
)

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/violations_files", StaticFiles(directory=VIOLATIONS_DIR), name="violations_files")
app.mount("/enrollment_files", StaticFiles(directory=ENROLLMENT_DIR), name="enrollment_files")

# Include attendance routes
app.include_router(attendance_router)

# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================
class Incident(BaseModel):
    camera_name: str
    violation_type: str
    confidence: float
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    timestamp: str
    image_path: str

class ImageData(BaseModel):
    image: str

# =========================================================
# IN-MEMORY CAMERA / AI STATE
# =========================================================
camera_state_lock = threading.Lock()
camera_state: Dict[str, Any] = {
    "camera_id": 1,
    "camera_name": "IP Camera - Construction Site",
    "source_rtsp_url": CAMERA_RTSP_URL,
    "mediamtx_rtsp_url": MEDIAMTX_RTSP_URL,
    "webrtc_url": MEDIAMTX_WEBRTC_URL,
    "hls_url": MEDIAMTX_HLS_URL,
    "stream_connected": False,
    "ai_running": False,
    "last_frame_at": None,
    "last_inference_at": None,
    "last_error": None,
    "reconnect_count": 0,
    "latest_detections": [],
    "latest_violations": [],
    "latest_snapshot_path": None,
    "latest_processing_time_ms": None,
}

ai_worker_thread: Optional[threading.Thread] = None
ai_worker_stop_event = threading.Event()

# Recognition worker for face detection
recognition_worker: Optional[SimpleRecognitionWorker] = None

# =========================================================
# DATABASE
# =========================================================
def get_db():
    """Get MongoDB database instance"""
    return MongoDBConfig.get_database()

# =========================================================
# MODEL LOADING
# =========================================================
def load_yolo_model():
    global model, YOLO
    if model is not None:
        return model

    if YOLO is None:
        try:
            from ultralytics import YOLO as YOLOClass
            YOLO = YOLOClass
        except Exception as e:
            print(f"❌ Failed to import YOLO: {e}")
            return None

    try:
        model_path = AIConfig.MODEL_PATH

        original_load = torch.load

        def patched_load(f, *args, **kwargs):
            kwargs["weights_only"] = False
            return original_load(f, *args, **kwargs)

        torch.load = patched_load
        try:
            model_instance = YOLO(model_path)
            model = model_instance
            print("✅ Model loaded successfully")
            print(f"   Path: {model_path}")
            print(f"   Confidence: {AIConfig.MODEL_CONFIDENCE}")
            print(f"   Classes: {len(AIConfig.DETECTION_CLASSES)}")
            return model
        finally:
            torch.load = original_load

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

# =========================================================
# HELPERS
# =========================================================
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def update_camera_state(**kwargs):
    with camera_state_lock:
        camera_state.update(kwargs)

def get_camera_state_snapshot() -> Dict[str, Any]:
    with camera_state_lock:
        return dict(camera_state)

def save_snapshot(frame: np.ndarray, prefix: str = "violation") -> str:
    filename = f"{prefix}_{int(time.time() * 1000)}.jpg"
    path = os.path.join(VIOLATIONS_DIR, filename)
    cv2.imwrite(path, frame)
    return f"/violations_files/{filename}"

def process_yolo_frame(frame: np.ndarray) -> Dict[str, Any]:
    yolo_model = load_yolo_model()
    if not yolo_model:
        return {
            "detections": [],
            "violations": [],
            "processing_time_ms": None,
            "error": "YOLO model not available"
        }

    start = time.time()
    results = yolo_model.predict(
        frame,
        conf=ai_config.MODEL_CONFIDENCE,
        iou=ai_config.MODEL_IOU_THRESHOLD,
        verbose=False
    )
    processing_time = round((time.time() - start) * 1000, 2)

    detections: List[Dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = ai_config.DETECTION_CLASSES.get(class_id, f"unknown_{class_id}")

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                }
            })

    try:
        violations = ai_config.detect_violations(detections)
    except Exception:
        violations = []

    return {
        "detections": detections,
        "violations": violations,
        "processing_time_ms": processing_time,
        "error": None
    }

def insert_incident_row(
    camera_name: str,
    violation_type: str,
    confidence: float,
    bbox_x: int,
    bbox_y: int,
    bbox_width: int,
    bbox_height: int,
    timestamp: str,
    image_path: str
):
    """Insert incident record into MongoDB"""
    try:
        db = get_db()
        if db is None:
            print("⚠️ Database connection failed while saving incident")
            return
        
        incident = {
            "camera_name": camera_name,
            "violation_type": violation_type,
            "confidence": confidence,
            "bbox_x": bbox_x,
            "bbox_y": bbox_y,
            "bbox_width": bbox_width,
            "bbox_height": bbox_height,
            "timestamp": datetime.fromisoformat(timestamp.replace('Z', '+00:00')) if isinstance(timestamp, str) else timestamp,
            "image_path": image_path
        }
        
        result = db.incidents.insert_one(incident)
        print(f"✅ Incident saved with ID: {result.inserted_id}")
        
    except Exception as e:
        print(f"⚠️ Failed to save incident: {e}")

# =========================================================
# AI WORKER
# =========================================================
def ai_worker_loop():
    print("🤖 AI worker thread starting...")
    print(f"[DEBUG] CAMERA_RTSP_URL value: {CAMERA_RTSP_URL}")
    update_camera_state(ai_running=True, last_error=None)

    cap = None
    frame_count = 0

    while not ai_worker_stop_event.is_set():
        try:
            if cap is None or not cap.isOpened():
                print(f"🎥 Connecting AI worker to Camera RTSP: {CAMERA_RTSP_URL}")
                # Check if using webcam (numeric index) or RTSP URL
                if CAMERA_RTSP_URL.isdigit():
                    # Direct webcam access - use MSMF backend for Windows (more stable than DSHOW)
                    cap = cv2.VideoCapture(int(CAMERA_RTSP_URL), cv2.CAP_MSMF)
                else:
                    # RTSP stream - use FFMPEG with TCP transport
                    rtsp_url_with_transport = f"{CAMERA_RTSP_URL}?transport=tcp"
                    cap = cv2.VideoCapture(rtsp_url_with_transport, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not cap.isOpened():
                    update_camera_state(
                        stream_connected=False,
                        last_error="Failed to open Camera RTSP stream"
                    )
                    time.sleep(AI_RECONNECT_DELAY_SEC)
                    continue

                update_camera_state(stream_connected=True, last_error=None)

            ok, frame = cap.read()
            if not ok or frame is None:
                update_camera_state(
                    stream_connected=False,
                    last_error="Failed to read frame from MediaMTX RTSP"
                )
                update_camera_state(reconnect_count=get_camera_state_snapshot()["reconnect_count"] + 1)

                if cap:
                    cap.release()
                cap = None
                time.sleep(AI_RECONNECT_DELAY_SEC)
                continue

            update_camera_state(stream_connected=True, last_frame_at=now_iso())
            frame_count += 1

            # Keep worker cheap; only infer every N frames
            if frame_count % AI_FRAME_SKIP != 0:
                continue

            frame = cv2.resize(frame, (AI_RESIZE_WIDTH, AI_RESIZE_HEIGHT))
            result = process_yolo_frame(frame)

            snapshot_path = None
            if result["violations"]:
                snapshot_path = save_snapshot(frame, prefix="violation")

            update_camera_state(
                last_inference_at=now_iso(),
                latest_detections=result["detections"],
                latest_violations=result["violations"],
                latest_snapshot_path=snapshot_path,
                latest_processing_time_ms=result["processing_time_ms"],
                last_error=result["error"]
            )

            if snapshot_path and result["detections"]:
                ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                for detection in result["detections"]:
                    bbox = detection["bbox"]
                    insert_incident_row(
                        camera_name="IP Camera - Construction Site",
                        violation_type=result["violations"][0].get("type", detection["class_name"]),
                        confidence=detection["confidence"],
                        bbox_x=int(bbox["x"]),
                        bbox_y=int(bbox["y"]),
                        bbox_width=int(bbox["width"]),
                        bbox_height=int(bbox["height"]),
                        timestamp=ts,
                        image_path=snapshot_path
                    )

        except Exception as e:
            update_camera_state(
                stream_connected=False,
                last_error=f"AI worker error: {str(e)}"
            )
            print(f"⚠️ AI worker error: {e}")
            time.sleep(AI_RECONNECT_DELAY_SEC)

    if cap:
        cap.release()

    update_camera_state(ai_running=False)
    print("🛑 AI worker thread stopped")

def start_ai_worker():
    global ai_worker_thread
    if not AI_ENABLED:
        print("ℹ️ AI worker disabled by config")
        return

    if ai_worker_thread and ai_worker_thread.is_alive():
        return

    ai_worker_stop_event.clear()
    ai_worker_thread = threading.Thread(target=ai_worker_loop, daemon=True)
    ai_worker_thread.start()

def stop_ai_worker():
    ai_worker_stop_event.set()

# =========================================================
# STARTUP / SHUTDOWN
# =========================================================
@app.on_event("startup")
async def startup_event():
    global recognition_worker
    
    print("🚀 Starting AI Construction Safety System...")

    if initialize_mongodb():
        print("✅ MongoDB initialized successfully")
    else:
        print("❌ MongoDB initialization failed")

    errors = Settings.validate_settings()
    if errors:
        print("⚠️ Settings validation warnings:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Settings validation passed")

    # Disabled AI worker to prevent webcam conflicts
    # if AI_ENABLED:
    #     start_ai_worker()

    # Start face recognition worker for attendance (only if enabled and database available)
    if RECOGNITION_ENABLED:
        try:
            db = get_db()
            if db is not None:
                print("🎯 Initializing face recognition worker...")
                recognition_worker = SimpleRecognitionWorker(
                    db=db,
                    rtsp_url=CAMERA_RTSP_URL,
                    worker_id="main"
                )
                if recognition_worker.start():
                    print("✅ Face recognition worker started successfully")
                else:
                    print("⚠️ Face recognition worker failed to start")
            else:
                print("⚠️ Cannot start recognition worker - database not available")
        except Exception as e:
            print(f"⚠️ Error initializing recognition worker: {str(e)}")
    else:
        print("ℹ️ Face recognition worker disabled (RECOGNITION_ENABLED=false)")

    print(f"✅ {settings.APP_NAME} v{settings.APP_VERSION} started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    global recognition_worker
    
    print("🛑 Shutting down AI Construction Safety System...")
    stop_ai_worker()
    
    if recognition_worker:
        recognition_worker.stop()

# =========================================================
# ROOT / HEALTH
# =========================================================
@app.get("/")
async def root():
    return {"message": "Construction AI Safety Monitoring System API"}

@app.get("/health")
async def health_check():
    state = get_camera_state_snapshot()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "ai_enabled": AI_ENABLED,
        "camera_stream_connected": state["stream_connected"],
        "ai_running": state["ai_running"],
        "mediamtx_rtsp_url": MEDIAMTX_RTSP_URL
    }

@app.get("/api/recognition/stats")
async def get_recognition_stats():
    """Get face recognition worker statistics"""
    if not recognition_worker:
        return {"status": "not_initialized"}
    
    stats = recognition_worker.get_stats()
    return {
        "status": "running" if stats["running"] else "stopped",
        **stats
    }

@app.post("/api/recognition/restart")
async def restart_recognition_worker():
    """Restart the face recognition worker"""
    global recognition_worker
    
    if recognition_worker:
        recognition_worker.stop()
        time.sleep(1)
    
    db = get_db()
    if db:
        recognition_worker = FaceRecognitionWorker(
            db=db,
            rtsp_url=CAMERA_RTSP_URL,
            worker_id="main"
        )
        if recognition_worker.start():
            return {"status": "restarted", "message": "Recognition worker restarted successfully"}
        else:
            return {"status": "error", "message": "Failed to start recognition worker"}, 500
    else:
        return {"status": "error", "message": "Database not available"}, 500

# =========================================================
# DIRECT IMAGE AI DETECTION (for webcam / tests)
# =========================================================
@app.post("/detect_base64")
async def detect_base64(data: ImageData):
    try:
        if not data.image:
            return {"success": False, "error": "No image data provided"}

        yolo_model = load_yolo_model()
        if not yolo_model:
            mock_detections = ai_config.generate_mock_detections()
            violations = ai_config.detect_violations(mock_detections)
            return {
                "success": True,
                "detections": mock_detections,
                "violations": violations,
                "model": "Mock Detection (Demo Mode)",
                "message": "YOLO model not found - using mock detection for demo",
                "processing_time_ms": 25
            }

        if "," in data.image:
            image_data = base64.b64decode(data.image.split(",")[1])
        else:
            image_data = base64.b64decode(data.image)

        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")

        frame = np.array(image)
        result = process_yolo_frame(frame)

        return {
            "success": True,
            "detections": result["detections"],
            "violations": result["violations"],
            "model": "YOLOv8 Safety Detection",
            "processing_time_ms": result["processing_time_ms"],
            "frame_info": {
                "width": frame.shape[1],
                "height": frame.shape[0],
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1
            }
        }

    except Exception as e:
        return {"success": False, "error": f"Detection failed: {str(e)}"}

# =========================================================
# CAMERAS / AI METADATA
# =========================================================
@app.get("/cameras")
async def get_cameras():
    state = get_camera_state_snapshot()
    return {
        "cameras": [
            {
                "id": 1,
                "name": state["camera_name"],
                "status": "active" if state["stream_connected"] else "degraded",
                "location": "Main Construction Area",
                "type": "rtsp",
                "ip": "192.168.1.71",
                "source_rtsp_url": state["source_rtsp_url"],
                "mediamtx_rtsp_url": state["mediamtx_rtsp_url"],
                "webrtc_url": state["webrtc_url"],
                "hls_url": state["hls_url"],
                "ai_enabled": AI_ENABLED,
                "ai_running": state["ai_running"],
                "last_frame_at": state["last_frame_at"],
                "last_inference_at": state["last_inference_at"],
                "last_error": state["last_error"],
            }
        ]
    }

@app.get("/cameras/{camera_id}/status")
async def get_camera_status(camera_id: int):
    if camera_id != 1:
        raise HTTPException(status_code=404, detail="Camera not found")
    return get_camera_state_snapshot()

@app.get("/cameras/{camera_id}/detections/latest")
async def get_latest_camera_detections(camera_id: int):
    if camera_id != 1:
        raise HTTPException(status_code=404, detail="Camera not found")
    state = get_camera_state_snapshot()
    return {
        "camera_id": camera_id,
        "camera_name": state["camera_name"],
        "last_inference_at": state["last_inference_at"],
        "processing_time_ms": state["latest_processing_time_ms"],
        "detections": state["latest_detections"],
        "violations": state["latest_violations"],
        "snapshot_path": state["latest_snapshot_path"],
    }

@app.post("/cameras/{camera_id}/ai/start")
async def start_camera_ai(camera_id: int):
    if camera_id != 1:
        raise HTTPException(status_code=404, detail="Camera not found")
    start_ai_worker()
    return {"message": "AI worker started"}

@app.post("/cameras/{camera_id}/ai/stop")
async def stop_camera_ai(camera_id: int):
    if camera_id != 1:
        raise HTTPException(status_code=404, detail="Camera not found")
    stop_ai_worker()
    return {"message": "AI worker stopping"}

# =========================================================
# INCIDENTS / DASHBOARD / ALERTS
# =========================================================
@app.post("/incidents")
async def create_incident(incident: Incident):
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        incident_data = {
            "camera_name": incident.camera_name,
            "violation_type": incident.violation_type,
            "confidence": incident.confidence,
            "bbox_x": incident.bbox_x,
            "bbox_y": incident.bbox_y,
            "bbox_width": incident.bbox_width,
            "bbox_height": incident.bbox_height,
            "timestamp": datetime.fromisoformat(incident.timestamp.replace('Z', '+00:00')) if isinstance(incident.timestamp, str) else incident.timestamp,
            "image_path": incident.image_path
        }
        
        result = db.incidents.insert_one(incident_data)
        return {"message": "Incident recorded successfully", "id": str(result.inserted_id)}
    
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/incidents")
async def get_incidents(limit: int = 50):
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        incidents = list(db.incidents.find().sort("timestamp", -1).limit(limit))
        
        # Convert ObjectId to string for JSON serialization
        for incident in incidents:
            incident["_id"] = str(incident["_id"])
            if isinstance(incident.get("timestamp"), datetime):
                incident["timestamp"] = incident["timestamp"].isoformat()
        
        return {"incidents": incidents}
    
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    finally:
        if cursor:
            cursor.close()
        connection.close()

@app.get("/stats")
async def get_stats():
    from config.mongodb import MongoDBConfig
    from datetime import datetime, timedelta
    
    db = MongoDBConfig.get_database()
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        # Get counts from MongoDB collections
        total_violations = db["violations"].count_documents({})
        total_alerts = db["alerts"].count_documents({})
        
        # Get active alerts from last 24 hours
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        active_alerts = db["alerts"].count_documents({"created_at": {"$gte": twenty_four_hours_ago}})
        
        # Get unique cameras
        cameras = db["violations"].distinct("camera_name")
        total_cameras = len(cameras) if cameras else 1
        
        # Get worker count
        total_workers = db["workers"].count_documents({})

        return {
            "total_workers": total_workers,
            "total_violations": total_violations,
            "active_alerts": active_alerts,
            "connected_cameras": total_cameras
        }

    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        return {
            "total_workers": 0,
            "total_violations": 0,
            "active_alerts": 0,
            "connected_cameras": 0
        }

@app.get("/dashboard/stats")
async def get_dashboard_stats():
    return await get_stats()

@app.get("/violations")
async def get_violations():
    from config.mongodb import MongoDBConfig
    
    db = MongoDBConfig.get_database()
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        # Get violations from MongoDB, sorted by timestamp descending, limit 50
        violations = list(
            db["violations"].find(
                {},
                {"_id": 1, "camera_name": 1, "violation_type": 1, "confidence": 1, "timestamp": 1, "status": 1, "image_path": 1}
            ).sort("timestamp", -1).limit(50)
        )
        
        # Convert ObjectId to string for JSON serialization
        for v in violations:
            v["_id"] = str(v.get("_id", ""))
            if "timestamp" in v:
                v["timestamp"] = v["timestamp"].isoformat() if hasattr(v["timestamp"], "isoformat") else str(v["timestamp"])
        
        return {"violations": violations}

    except Exception as e:
        print(f"Error getting violations: {str(e)}")
        return {"violations": []}

@app.get("/workers")
async def get_workers():
    return {
        "workers": [
            {"id": 1, "name": "John Smith", "department": "Construction", "status": "active"},
            {"id": 2, "name": "Mike Johnson", "department": "Safety", "status": "active"},
            {"id": 3, "name": "Sarah Wilson", "department": "Engineering", "status": "active"},
            {"id": 4, "name": "Tom Davis", "department": "Construction", "status": "inactive"}
        ]
    }

@app.get("/alerts")
async def get_alerts():
    from config.mongodb import MongoDBConfig
    from datetime import datetime, timedelta

    db = MongoDBConfig.get_database()
    if db is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        # Get alerts from last 24 hours, sorted by creation time descending
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        alerts = list(
            db["alerts"].find(
                {"created_at": {"$gte": twenty_four_hours_ago}},
                {"_id": 1, "message": 1, "level": 1, "camera_name": 1, "status": 1, "created_at": 1}
            ).sort("created_at", -1)
        )

        # Convert ObjectId and datetime to strings for JSON serialization
        for a in alerts:
            a["_id"] = str(a.get("_id", ""))
            if "created_at" in a:
                a["created_at"] = a["created_at"].isoformat() if hasattr(a["created_at"], "isoformat") else str(a["created_at"])

        return {"alerts": alerts}

    except Exception as e:
        print(f"Error getting alerts: {str(e)}")
        return {"alerts": []}

# =========================================================
# DIRECT CAMERA STREAM (MJPEG) - Bypasses MediaMTX
# =========================================================
def generate_mjpeg_stream():
    """Generate MJPEG stream directly from camera"""
    # Use webcam index 0 with MSMF backend for Windows
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera, retrying...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(CAMERA_RTSP_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue

            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    finally:
        cap.release()

@app.get("/stream")
async def stream_camera():
    """MJPEG streaming endpoint - bypasses MediaMTX audio codec issues"""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
