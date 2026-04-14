from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import mysql.connector
from mysql.connector import Error
import os
import base64
import io
import time
from PIL import Image
import numpy as np
import torch

# Lazy import for YOLO to handle NumPy compatibility issues
YOLO = None

# Import configuration modules
from config.database import DatabaseConfig, initialize_database
from config.settings import Settings, settings
from config.ai_config import AIConfig, ai_config

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI Construction Safety Monitoring System API"
)

# =========================================================
# CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    **Settings.get_cors_config()
)

# =========================================================
# STATIC FILES
# =========================================================
VIOLATIONS_DIR = "data/images/violations"
os.makedirs(VIOLATIONS_DIR, exist_ok=True)
app.mount("/violations_files", StaticFiles(directory=VIOLATIONS_DIR), name="violations_files")

# =========================================================
# DATABASE CONNECTION
# =========================================================
def get_db_connection():
    return DatabaseConfig.get_connection()

# =========================================================
# INITIALIZATION
# =========================================================
@app.on_event("startup")
async def startup_event():
    """Initialize database and AI model on startup"""
    print("🚀 Starting AI Construction Safety System...")
    
    # Initialize database
    if initialize_database():
        print("✅ Database initialized successfully")
    else:
        print("❌ Database initialization failed")
    
    # Validate settings
    errors = Settings.validate_settings()
    if errors:
        print("⚠️ Settings validation warnings:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Settings validation passed")
    
    print(f"✅ {settings.APP_NAME} v{settings.APP_VERSION} started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("🛑 Shutting down AI Construction Safety System...")

# =========================================================
# REQUEST MODELS
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
# AI MODEL LOADING (LAZY)
# =========================================================
model = None

def load_yolo_model():
    """Lazy load YOLO model on first use"""
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
        # Use PPE model from AIConfig
        model_path = AIConfig.get_model_path()
        model = YOLO(model_path)
        print(f"✅ PPE Detection model loaded successfully!")
        print(f"   Model: Hansung-Cho/yolov8-ppe-detection")
        print(f"   Path: {model_path}")
        print(f"   Confidence: {AIConfig.MODEL_CONFIDENCE}")
        print(f"   Classes: {len(AIConfig.DETECTION_CLASSES)} (Hardhat, Vest, Mask, Person, etc.)")
        return model
    except Exception as e:
        print(f"❌ Failed to load PPE model: {e}")
        return None

# =========================================================
# AI DETECTION ENDPOINT
# =========================================================
@app.post("/detect_base64")
async def detect_base64(data: ImageData):
    try:
        print(f"🔍 Starting AI detection process...")
        
        # Validate input
        if not data.image:
            return {"success": False, "error": "No image data provided"}
        
        # Lazy load YOLO model
        yolo_model = load_yolo_model()
        
        # Check if model is available
        if not yolo_model:
            print("🤖 YOLO model not loaded, using mock detection")
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
        
        print(f"✅ YOLO model is loaded: {type(yolo_model)}")
        print(f"📊 Model config: confidence={ai_config.MODEL_CONFIDENCE}, iou={ai_config.MODEL_IOU_THRESHOLD}")
        
        # Decode base64 image
        try:
            if ',' in data.image:
                # Remove data:image/jpeg;base64, prefix
                image_data = base64.b64decode(data.image.split(',')[1])
            else:
                # Direct base64 string
                image_data = base64.b64decode(data.image)
            
            print(f"📷 Image decoded successfully, size: {len(image_data)} bytes")
        except Exception as e:
            print(f"❌ Image decoding error: {e}")
            return {"success": False, "error": f"Image decoding failed: {str(e)}"}
        
        # Open and process image
        try:
            image = Image.open(io.BytesIO(image_data))
            print(f"🖼️ Image opened: mode={image.mode}, size={image.size}")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                print(f"🔄 Converted to RGB")
            
            # Convert to numpy array
            frame = np.array(image)
            print(f"📊 Frame shape: {frame.shape}")
            
        except Exception as e:
            print(f"❌ Image processing error: {e}")
            return {"success": False, "error": f"Image processing failed: {str(e)}"}
        
        # Run YOLO detection
        try:
            start_time = time.time()
            results = model.predict(
                frame, 
                conf=ai_config.MODEL_CONFIDENCE, 
                iou=ai_config.MODEL_IOU_THRESHOLD,
                verbose=False  # Reduce log spam
            )
            processing_time = (time.time() - start_time) * 1000
            
            print(f"🤖 YOLO inference completed in {processing_time:.2f}ms")
            print(f"📊 Results: {len(results)} result(s)")
            
        except Exception as e:
            print(f"❌ YOLO inference error: {e}")
            return {"success": False, "error": f"YOLO inference failed: {str(e)}"}
        
        # Process detections
        detections = []
        try:
            for i, result in enumerate(results):
                print(f"🔍 Processing result {i+1}")
                boxes = result.boxes
                if boxes is not None:
                    print(f"📦 Found {len(boxes)} boxes")
                    for j, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        print(f"  Box {j+1}: class_id={class_id}, confidence={confidence:.3f}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                        
                        # Map class ID to class name using AI config
                        class_name = ai_config.DETECTION_CLASSES.get(class_id, f'unknown_{class_id}')
                        
                        detection = {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": round(confidence, 3),
                            "bbox": {
                                "x": float(x1),
                                "y": float(y1), 
                                "width": float(x2 - x1),
                                "height": float(y2 - y1)
                            }
                        }
                        detections.append(detection)
                        print(f"  ✅ Added detection: {class_name} ({confidence:.3f})")
                else:
                    print(f"  ❌ No boxes found in result {i+1}")
                    
        except Exception as e:
            print(f"❌ Detection processing error: {e}")
            return {"success": False, "error": f"Detection processing failed: {str(e)}"}
        
        print(f"📈 Total detections: {len(detections)}")
        
        # Detect violations
        try:
            violations = ai_config.detect_violations(detections)
            print(f"⚠️ Violations detected: {len(violations)}")
            for violation in violations:
                print(f"  🚨 {violation.get('type', 'unknown')}: {violation.get('severity', 'medium')}")
        except Exception as e:
            print(f"❌ Violation detection error: {e}")
            violations = []
        
        response = {
            "success": True,
            "detections": detections,
            "violations": violations,
            "model": "YOLOv8 Safety Detection",
            "processing_time_ms": round(processing_time, 2),
            "frame_info": {
                "width": frame.shape[1],
                "height": frame.shape[0],
                "channels": frame.shape[2] if len(frame.shape) > 2 else 1
            }
        }
        
        print(f"✅ Detection completed successfully: {len(detections)} objects, {len(violations)} violations")
        return response
        
    except Exception as e:
        print(f"❌ General detection error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": f"Detection failed: {str(e)}"}

# =========================================================
# API ENDPOINTS
# =========================================================

@app.get("/")
async def root():
    return {"message": "Construction AI Safety Monitoring System API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/incidents")
async def create_incident(incident: Incident):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO incidents (camera_name, violation_type, confidence, bbox_x, bbox_y, bbox_width, bbox_height, timestamp, image_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            incident.camera_name,
            incident.violation_type,
            incident.confidence,
            incident.bbox_x,
            incident.bbox_y,
            incident.bbox_width,
            incident.bbox_height,
            incident.timestamp,
            incident.image_path
        ))
        connection.commit()
        return {"message": "Incident recorded successfully", "id": cursor.lastrowid}
    
    except Error as e:
        connection.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        connection.close()

@app.get("/incidents")
async def get_incidents(limit: int = 50):
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM incidents ORDER BY timestamp DESC LIMIT %s"
        cursor.execute(query, (limit,))
        incidents = cursor.fetchall()
        return {"incidents": incidents}
    
    except Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        connection.close()

@app.get("/stats")
async def get_stats():
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = connection.cursor()
        
        # Get total incidents
        cursor.execute("SELECT COUNT(*) FROM incidents")
        total_incidents = cursor.fetchone()[0]
        
        # Get total persons (unique worker count)
        cursor.execute("SELECT COUNT(DISTINCT camera_name) FROM incidents")
        total_cameras = cursor.fetchone()[0]
        
        # Get active alerts (last 24 hours)
        cursor.execute("SELECT COUNT(*) FROM incidents WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)")
        active_alerts = cursor.fetchone()[0]
        
        # Get total workers (mock data)
        cursor.execute("SELECT COUNT(*) FROM incidents")
        total_persons = cursor.fetchone()[0]

        return {
            "total_workers": total_persons,
            "total_violations": total_incidents,
            "active_alerts": active_alerts,
            "connected_cameras": total_cameras if total_cameras > 0 else 1
        }

    except Error as e:
        raise HTTPException(status_code=500, detail=f"MySQL stats error: {str(e)}")

    finally:
        if cursor:
            cursor.close()
        connection.close()

@app.get("/dashboard/stats")
async def get_dashboard_stats():
    # Alias for the same stats endpoint to match frontend API
    return await get_stats()

@app.get("/violations")
async def get_violations():
    # Return incidents as violations for dashboard
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM incidents ORDER BY timestamp DESC LIMIT 50"
        cursor.execute(query)
        violations = cursor.fetchall()
        return {"violations": violations}
    
    except Error as e:
        # If table doesn't exist, return empty data
        return {"violations": []}
    
    finally:
        if cursor:
            cursor.close()
        connection.close()

@app.get("/workers")
async def get_workers():
    # Mock worker data for dashboard
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
    # Return recent incidents as alerts
    connection = get_db_connection()
    if not connection:
        raise HTTPException(status_code=500, detail="Database connection failed")
    
    try:
        cursor = connection.cursor(dictionary=True)
        query = "SELECT * FROM incidents WHERE timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR) ORDER BY timestamp DESC"
        cursor.execute(query)
        alerts = cursor.fetchall()
        return {"alerts": alerts}
    
    except Error as e:
        # If table doesn't exist, return empty data
        return {"alerts": []}
    
    finally:
        if cursor:
            cursor.close()
        connection.close()

@app.get("/cameras")
async def get_cameras():
    # Camera data including user's IP camera with streaming endpoint
    return {
        "cameras": [
            {"id": 1, "name": "IP Camera - Construction Site", "status": "active", "location": "Main Construction Area", "ip": "192.168.1.71", "rtsp_url": "rtsp://192.168.1.71:554/11?tcp", "type": "rtsp", "stream_url": "/stream/camera/1"},
            {"id": 2, "name": "Main Entrance", "status": "active", "location": "Front Gate", "type": "webcam"},
            {"id": 3, "name": "Construction Area", "status": "active", "location": "Building Site", "type": "webcam"},
            {"id": 4, "name": "Parking Lot", "status": "inactive", "location": "Parking Area", "type": "webcam"}
        ]
    }

@app.get("/video_feed")
def video_feed():
    """Stream RTSP camera to browser via MJPEG - simple endpoint"""
    from fastapi.responses import StreamingResponse
    import cv2
    
    RTSP_URL = "rtsp://192.168.1.71:554/11?tcp"
    
    def generate_frames():
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

        while True:
            success, frame = cap.read()
            if not success:
                continue

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingResponse(generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/camera/{camera_id}")
async def stream_camera(camera_id: int):
    """Stream RTSP camera to browser via MJPEG"""
    from fastapi.responses import StreamingResponse
    import cv2
    
    # Camera configuration
    camera_config = {
        1: {"rtsp_url": "rtsp://192.168.1.71:554/11?tcp", "name": "IP Camera - Construction Site"}
    }
    
    if camera_id not in camera_config:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    config = camera_config[camera_id]
    
    def generate_frames():
        """Generate frames from RTSP camera with stable reconnect logic"""
        while True:
            cap = cv2.VideoCapture(config["rtsp_url"], cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                print(f"❌ Failed to connect to RTSP camera: {config['rtsp_url']}")
                import time
                time.sleep(3)
                continue
            
            print(f"✅ Connected to RTSP camera: {config['name']}")
            
            try:
                while True:
                    success, frame = cap.read()
                    if not success:
                        print("⚠️ Frame not received, reconnecting...")
                        break
                    
                    # Reduce load - resize frame
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Encode to JPEG
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            finally:
                cap.release()
    
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera/frame")
async def get_camera_frame():
    """Get a single JPEG frame from IP camera for polling-based display"""
    from fastapi.responses import StreamingResponse
    import cv2
    import io
    from PIL import Image
    
    RTSP_URL = "rtsp://192.168.1.71:554/11?tcp"
    
    try:
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            # Return a blank frame if camera is not available
            blank_frame = cv2.cvtColor(cv2.zeros((480, 640, 3), dtype="uint8"), cv2.COLOR_BGR2RGB)
            blank_frame = cv2.putText(blank_frame, "Camera Offline", (150, 240), 
                                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
        
        # Read frames until we get a good one
        for _ in range(5):
            success, frame = cap.read()
            if success:
                # Resize for performance
                frame = cv2.resize(frame, (640, 480))
                _, buffer = cv2.imencode('.jpg', frame)
                cap.release()
                return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
        
        cap.release()
        return {"error": "Could not read frame from camera"}
        
    except Exception as e:
        print(f"❌ Error getting camera frame: {e}")
        return {"error": str(e)}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
