"""Production-ready request/response models and validation"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum

# =========================================================
# ENUMS
# =========================================================
class ViolationType(str, Enum):
    NO_HARD_HAT = "No Hard Hat"
    NO_SAFETY_VEST = "No Safety Vest"
    NO_SAFETY_SHOES = "No Safety Shoes"
    UNSAFE_POSTURE = "Unsafe Posture"
    BLOCKED_EXIT = "Blocked Exit"
    FIRE_HAZARD = "Fire Hazard"
    OTHER = "Other"

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ViolationStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"

# =========================================================
# REQUEST MODELS
# =========================================================
class IncidentCreate(BaseModel):
    """Model for creating a new incident"""
    camera_name: str = Field(..., min_length=1, max_length=100)
    violation_type: str = Field(..., min_length=1, max_length=100)
    confidence: float = Field(..., ge=0, le=1)
    bbox_x: int = Field(..., ge=0)
    bbox_y: int = Field(..., ge=0)
    bbox_width: int = Field(..., gt=0)
    bbox_height: int = Field(..., gt=0)
    image_path: str = Field(..., min_length=1, max_length=255)
    persons: Optional[int] = Field(default=None, ge=0)
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v

class CameraCreate(BaseModel):
    """Model for creating a camera"""
    name: str = Field(..., min_length=1, max_length=100)
    location: str = Field(..., min_length=1, max_length=255)
    rtsp_url: str = Field(..., min_length=1, max_length=500)
    is_active: bool = True
    
class WorkerCreate(BaseModel):
    """Model for creating a worker"""
    name: str = Field(..., min_length=1, max_length=100)
    employee_id: str = Field(..., min_length=1, max_length=50)
    department: Optional[str] = Field(default=None, max_length=100)
    contact: Optional[str] = Field(default=None, max_length=20)

class AlertCreate(BaseModel):
    """Model for creating an alert"""
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    severity: AlertSeverity
    related_violation_id: Optional[int] = None

class StatusUpdate(BaseModel):
    """Model for updating incident status"""
    status: ViolationStatus

# =========================================================
# RESPONSE MODELS
# =========================================================
class IncidentResponse(BaseModel):
    """Model for incident response"""
    id: int
    camera_name: str
    violation_type: str
    confidence: float
    timestamp: datetime
    status: str
    image_path: Optional[str]
    persons: Optional[int]
    
    class Config:
        orm_mode = True

class CameraResponse(BaseModel):
    """Model for camera response"""
    id: int
    name: str
    location: str
    rtsp_url: str
    is_active: bool
    last_seen: Optional[datetime]
    
    class Config:
        orm_mode = True

class WorkerResponse(BaseModel):
    """Model for worker response"""
    id: int
    name: str
    employee_id: str
    department: Optional[str]
    contact: Optional[str]
    created_at: datetime
    
    class Config:
        orm_mode = True

class AlertResponse(BaseModel):
    """Model for alert response"""
    id: int
    title: str
    description: Optional[str]
    severity: str
    created_at: datetime
    is_read: bool
    
    class Config:
        orm_mode = True

class HealthResponse(BaseModel):
    """Model for health check response"""
    status: str
    database: str
    version: str
    timestamp: datetime

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None

class PaginatedResponse(BaseModel):
    """Generic paginated response"""
    data: List
    total: int
    page: int
    per_page: int
    total_pages: int
