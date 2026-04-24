"""
Attendance and Face Recognition Models
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from bson import ObjectId


# ============ EMPLOYEE MODELS ============

class EmployeeBase(BaseModel):
    employee_code: Optional[str] = None  # Auto-generated if not provided
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    department: Optional[str] = None
    site_id: Optional[str] = None
    status: str = "active"  # active, inactive


class EmployeeCreate(EmployeeBase):
    pass


class EmployeeResponse(EmployeeBase):
    id: str = Field(alias="_id")
    created_at: datetime

    class Config:
        populate_by_name = True


# ============ FACE PROFILE MODELS ============

class FaceProfileBase(BaseModel):
    photo_url: str
    cropped_face_url: str
    embedding: List[float]
    quality_score: float
    is_primary: bool = False


class FaceProfileCreate(FaceProfileBase):
    employee_id: str


class FaceProfileResponse(FaceProfileBase):
    id: str = Field(alias="_id")
    employee_id: str
    created_at: datetime

    class Config:
        populate_by_name = True


# ============ CAMERA MODELS ============

class CameraBase(BaseModel):
    camera_name: str
    rtsp_url: str
    location_name: str
    site_id: Optional[str] = None
    active: bool = True
    frame_interval_sec: int = 2
    match_enabled: bool = True


class CameraCreate(CameraBase):
    pass


class CameraResponse(CameraBase):
    id: str = Field(alias="_id")
    created_at: datetime

    class Config:
        populate_by_name = True


# ============ ATTENDANCE MODELS ============

class AttendanceLog(BaseModel):
    employee_id: str
    site_id: Optional[str] = None
    camera_id: str
    date: str  # YYYY-MM-DD
    check_in_time: Optional[str] = None
    check_out_time: Optional[str] = None
    source: str = "rtsp_face"
    confidence: float
    snapshot_url: Optional[str] = None
    status: str = "active"  # active, reviewed, rejected


# ============ FACE DETECTION EVENT MODELS ============

class FaceDetectionEvent(BaseModel):
    camera_id: str
    site_id: Optional[str] = None
    timestamp: datetime
    snapshot_url: str
    detected_faces_count: int
    matches: List[dict] = []  # [{employee_id, name, confidence, face_idx}]
    processed: bool = False


class RecognitionMatch(BaseModel):
    employee_id: str
    name: str
    confidence: float
    face_index: int


class FaceEnrollmentRequest(BaseModel):
    employee_id: str
    photo_base64: str  # base64 encoded image
    is_primary: bool = False


# ============ UNKNOWN FACE MODELS ============

class UnknownFace(BaseModel):
    camera_id: str
    timestamp: datetime
    snapshot_url: str
    detected_face_embedding: List[float]
    assigned_employee_id: Optional[str] = None
    reviewed: bool = False
    confidence_scores: dict = {}  # {employee_id: confidence}


# ============ ATTENDANCE RULES ============

class AttendanceRules(BaseModel):
    site_id: str
    late_threshold_minutes: int = 10
    duplicate_cooldown_seconds: int = 600  # 10 minutes
    check_in_window_start: str = "06:00"  # HH:MM
    check_in_window_end: str = "10:00"
    check_out_window_start: str = "16:00"
    check_out_window_end: str = "20:00"
    min_confidence: float = 0.75
    max_faces_per_frame: int = 5
    ignore_if_mask: bool = False
    ignore_if_helmet: bool = False
