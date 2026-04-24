"""
Attendance and Face Enrollment API Routes
"""
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pymongo.errors import PyMongoError
from datetime import datetime, timedelta
import os
import base64
import cv2
import numpy as np
from typing import Optional, List
import logging

from config.attendance_models import (
    EmployeeCreate, EmployeeResponse, FaceEnrollmentRequest,
    FaceProfileResponse, CameraCreate, CameraResponse, 
    RecognitionMatch, AttendanceLog
)
from config.face_recognition import (
    base64_to_cv2, cv2_to_base64, detect_faces, generate_embedding,
    extract_face_crop, calculate_face_quality, compare_embeddings,
    find_best_match
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/attendance", tags=["attendance"])

# Imported from server.py
def get_db():
    from config.mongodb import MongoDBConfig
    return MongoDBConfig.get_database()


# ============ EMPLOYEE MANAGEMENT ============

@router.post("/employees", response_model=dict)
async def create_employee(employee: EmployeeCreate):
    """Create new employee"""
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")

        employee_data = employee.dict()

        # Check if email already exists
        if employee_data.get("email"):
            existing_email = db.employees.find_one({"email": employee_data["email"]})
            if existing_email:
                raise HTTPException(status_code=400, detail="Employee with this email already exists")

        # Generate employee_code if not provided
        if not employee_data.get("employee_code"):
            import uuid
            employee_data["employee_code"] = f"EMP-{str(uuid.uuid4())[:8].upper()}"

        # Check if employee code already exists
        existing = db.employees.find_one({"employee_code": employee_data["employee_code"]})
        if existing:
            raise HTTPException(status_code=400, detail="Employee code already exists")

        employee_data["created_at"] = datetime.utcnow()
        employee_data["status"] = "active"

        result = db.employees.insert_one(employee_data)

        return {
            "message": "Employee created successfully",
            "employee_id": str(result.inserted_id)
        }

    except PyMongoError as e:
        # Handle duplicate key errors specifically
        if "duplicate key error" in str(e):
            raise HTTPException(status_code=400, detail="Employee with this email or employee code already exists")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/employees")
async def list_employees(status: Optional[str] = None):
    """List all employees"""
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        query = {}
        if status:
            query["status"] = status
        
        employees = list(db.employees.find(query).sort("created_at", -1))
        
        # Convert ObjectId to string
        for emp in employees:
            emp["_id"] = str(emp["_id"])
        
        return {"employees": employees}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/employees/{employee_id}")
async def get_employee(employee_id: str):
    """Get employee by ID"""
    try:
        from bson import ObjectId
        db = get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        employee = db.employees.find_one({"_id": ObjectId(employee_id)})
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        employee["_id"] = str(employee["_id"])
        return employee
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ============ FACE ENROLLMENT ============

@router.post("/employees/{employee_id}/enroll-face")
async def enroll_face(employee_id: str, file: UploadFile = File(...), is_primary: bool = Form(False)):
    """
    Enroll employee face photo
    
    Process:
    1. Read uploaded image
    2. Detect face
    3. Validate face quality
    4. Generate embedding
    5. Save face profile
    """
    try:
        from bson import ObjectId
        db = get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        # Verify employee exists
        employee = db.employees.find_one({"_id": ObjectId(employee_id)})
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        # Read uploaded image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        logger.info(f"📸 Processing face enrollment for employee {employee_id}")
        logger.info(f"Image shape: {image.shape}")

        # Detect faces (will use InsightFace if available, otherwise OpenCV fallback)
        faces, annotated_image = detect_faces(image)

        if not faces:
            logger.error(f"No faces detected in image. Image shape: {image.shape}")
            raise HTTPException(status_code=400, detail="No face detected in image. Please upload a clear photo with a visible face.")

        # If multiple faces detected, use the largest one
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected ({len(faces)}), using the largest one")
            # Sort by face area and pick the largest
            faces_sorted = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            face_info = faces_sorted[0]
        else:
            face_info = faces[0]
        
        # Calculate quality score
        quality_score = calculate_face_quality(image, face_info)
        logger.info(f"Face quality score: {quality_score:.2f}")
        
        if quality_score < 0.4:
            raise HTTPException(status_code=400, 
                              detail=f"Face quality too low ({quality_score:.2f}). Please ensure good lighting and clear face.")
        
        # Extract face crop
        face_crop = extract_face_crop(image, face_info, padding=20)
        if face_crop is None:
            raise HTTPException(status_code=400, detail="Could not extract face region")
        
        # Generate embedding
        embedding = generate_embedding(image, face_info)
        if embedding is None:
            raise HTTPException(status_code=400, detail="Could not generate face embedding")
        
        # Save original image
        ENROLL_DIR = "data/images/enrollments"
        os.makedirs(ENROLL_DIR, exist_ok=True)
        
        filename = f"employee_{employee_id}_{datetime.utcnow().timestamp()}.jpg"
        filepath = os.path.join(ENROLL_DIR, filename)
        cv2.imwrite(filepath, image)
        
        # Save cropped face
        crop_filename = f"crop_{employee_id}_{datetime.utcnow().timestamp()}.jpg"
        crop_filepath = os.path.join(ENROLL_DIR, crop_filename)
        cv2.imwrite(crop_filepath, face_crop)
        
        # Save face profile to database
        face_profile = {
            "employee_id": ObjectId(employee_id),
            "photo_url": f"/enrollment_files/{filename}",
            "cropped_face_url": f"/enrollment_files/{crop_filename}",
            "embedding": embedding,
            "quality_score": quality_score,
            "is_primary": is_primary,
            "created_at": datetime.utcnow()
        }
        
        result = db.employee_face_profiles.insert_one(face_profile)
        
        logger.info(f"✅ Face enrolled successfully for employee {employee_id}")
        
        return {
            "message": "Face enrolled successfully",
            "face_id": str(result.inserted_id),
            "quality_score": quality_score,
            "embedding_size": len(embedding)
        }
    
    except HTTPException:
        raise
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Error enrolling face: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/employees/{employee_id}/face-profiles")
async def get_employee_faces(employee_id: str):
    """Get all face profiles for an employee"""
    try:
        from bson import ObjectId
        db = get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        # Verify employee exists
        employee = db.employees.find_one({"_id": ObjectId(employee_id)})
        if not employee:
            raise HTTPException(status_code=404, detail="Employee not found")
        
        faces = list(db.employee_face_profiles.find({"employee_id": ObjectId(employee_id)}))
        
        # Convert ObjectIds to strings
        for face in faces:
            face["_id"] = str(face["_id"])
            face["employee_id"] = str(face["employee_id"])
        
        return {"faces": faces, "count": len(faces)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ============ CAMERA MANAGEMENT ============

@router.post("/cameras", response_model=dict)
async def register_camera(camera: CameraCreate):
    """Register RTSP camera"""
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        camera_data = camera.dict()
        camera_data["created_at"] = datetime.utcnow()
        camera_data["last_frame_time"] = None
        camera_data["health_status"] = "initializing"
        
        result = db.cameras.insert_one(camera_data)
        
        logger.info(f"✅ Camera registered: {camera.camera_name}")
        
        return {
            "message": "Camera registered successfully",
            "camera_id": str(result.inserted_id)
        }
    
    except PyMongoError as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/cameras")
async def list_cameras():
    """List all registered cameras"""
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        cameras = list(db.cameras.find().sort("created_at", -1))
        
        for camera in cameras:
            camera["_id"] = str(camera["_id"])
        
        return {"cameras": cameras}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# ============ ATTENDANCE RECORDS ============

@router.get("/today")
async def get_today_attendance():
    """Get attendance records for today"""
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        records = list(db.attendance_logs.find({"date": today}).sort("check_in_time", -1))
        
        for record in records:
            record["_id"] = str(record["_id"])
            record["employee_id"] = str(record["employee_id"])
            record["camera_id"] = str(record["camera_id"])
        
        return {"date": today, "records": records, "count": len(records)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/mark")
async def mark_attendance(
    employee_id: str,
    camera_id: str,
    confidence: float,
    attendance_type: str = "check_in"  # check_in or check_out
):
    """Mark attendance when face is recognized"""
    try:
        from bson import ObjectId
        db = get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database connection failed")
        
        today = datetime.utcnow().strftime("%Y-%m-%d")
        now = datetime.utcnow().isoformat()
        
        # Check if attendance already marked
        existing = db.attendance_logs.find_one({
            "employee_id": ObjectId(employee_id),
            "camera_id": ObjectId(camera_id),
            "date": today
        })
        
        if existing:
            if attendance_type == "check_in" and existing.get("check_in_time"):
                return {"message": "Already checked in today", "status": "duplicate"}
        
        # Create/update attendance record
        attendance = {
            "employee_id": ObjectId(employee_id),
            "camera_id": ObjectId(camera_id),
            "date": today,
            "check_in_time": now if attendance_type == "check_in" else existing.get("check_in_time"),
            "check_out_time": now if attendance_type == "check_out" else None,
            "source": "rtsp_face",
            "confidence": confidence,
            "status": "active"
        }
        
        if existing:
            db.attendance_logs.update_one(
                {"_id": existing["_id"]},
                {"$set": attendance}
            )
            record_id = str(existing["_id"])
        else:
            result = db.attendance_logs.insert_one(attendance)
            record_id = str(result.inserted_id)
        
        logger.info(f"✅ Attendance marked: {employee_id} - {attendance_type}")
        
        return {
            "message": f"Attendance {attendance_type} marked successfully",
            "record_id": record_id,
            "attendance": attendance
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Mount static files for enrollment images
# This will be added to server.py: app.mount("/enrollment_files", StaticFiles(directory="data/images/enrollments"), name="enrollment_files")
