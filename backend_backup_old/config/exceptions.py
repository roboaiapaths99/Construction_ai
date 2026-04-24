"""Comprehensive error handling and standardized responses"""
from fastapi import HTTPException
from datetime import datetime
import uuid
from typing import Any, Dict, Optional

class AppError(Exception):
    """Base application error"""
    def __init__(self, message: str, error_code: str = "UNKNOWN", status_code: int = 500):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)

class ValidationError(AppError):
    """Input validation error"""
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR", 400)

class AuthenticationError(AppError):
    """Authentication error"""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, "AUTHENTICATION_ERROR", 401)

class AuthorizationError(AppError):
    """Authorization error"""
    def __init__(self, message: str = "Not authorized"):
        super().__init__(message, "AUTHORIZATION_ERROR", 403)

class NotFoundError(AppError):
    """Resource not found error"""
    def __init__(self, resource: str):
        super().__init__(f"{resource} not found", "NOT_FOUND", 404)

class ConflictError(AppError):
    """Resource conflict error"""
    def __init__(self, message: str):
        super().__init__(message, "CONFLICT", 409)

class DatabaseError(AppError):
    """Database operation error"""
    def __init__(self, message: str = "Database operation failed"):
        super().__init__(message, "DATABASE_ERROR", 500)

class ExternalServiceError(AppError):
    """External service error"""
    def __init__(self, service: str, message: str = "Service unavailable"):
        super().__init__(f"{service}: {message}", "EXTERNAL_SERVICE_ERROR", 503)

# =========================================================
# RESPONSE BUILDERS
# =========================================================
class ApiResponse:
    """Standardized API response builder"""
    
    @staticmethod
    def success(
        data: Any = None,
        message: str = "Success",
        status_code: int = 200
    ) -> Dict:
        """Build a success response"""
        return {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4())
        }
    
    @staticmethod
    def error(
        message: str,
        error_code: str = "ERROR",
        status_code: int = 500,
        details: Optional[Dict] = None
    ) -> Dict:
        """Build an error response"""
        return {
            "success": False,
            "error": error_code,
            "message": message,
            "details": details,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4())
        }
    
    @staticmethod
    def paginated(
        data: list,
        total: int,
        page: int,
        per_page: int,
        message: str = "Success"
    ) -> Dict:
        """Build a paginated response"""
        total_pages = (total + per_page - 1) // per_page
        return {
            "success": True,
            "message": message,
            "data": data,
            "pagination": {
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": total_pages
            },
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4())
        }

# =========================================================
# ERROR TO HTTP EXCEPTION CONVERTER
# =========================================================
def error_to_http_exception(error: AppError) -> HTTPException:
    """Convert AppError to HTTPException"""
    return HTTPException(
        status_code=error.status_code,
        detail=ApiResponse.error(
            message=error.message,
            error_code=error.error_code,
            status_code=error.status_code
        )
    )
