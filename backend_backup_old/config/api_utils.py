"""Production API utilities and middleware"""
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import time
import uuid
from config.logging_config import app_logger
from datetime import datetime

# =========================================================
# REQUEST/RESPONSE LOGGING MIDDLEWARE
# =========================================================
async def logging_middleware(request: Request, call_next):
    """Middleware to log requests and responses"""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Log request
    start_time = time.time()
    app_logger.info(
        f"REQUEST [{request_id}]: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown"
        }
    )
    
    try:
        response = await call_next(request)
    except Exception as e:
        app_logger.error(
            f"ERROR [{request_id}]: {str(e)}",
            extra={"error": str(e)}
        )
        raise
    
    # Log response
    process_time = time.time() - start_time
    app_logger.info(
        f"RESPONSE [{request_id}]: {response.status_code} ({process_time:.3f}s)",
        extra={
            "status_code": response.status_code,
            "process_time": process_time
        }
    )
    
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# =========================================================
# SECURITY HEADERS MIDDLEWARE
# =========================================================
async def security_headers_middleware(request: Request, call_next):
    """Middleware to add security headers"""
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response

# =========================================================
# RATE LIMITING
# =========================================================
from collections import defaultdict
from datetime import datetime, timedelta

class RateLimiter:
    """Simple rate limiter"""
    def __init__(self, max_requests: int = 100, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = timedelta(seconds=time_window)
        self.requests = defaultdict(list)
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        now = datetime.utcnow()
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.time_window
        ]
        
        # Check limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        
        return False

rate_limiter = RateLimiter(max_requests=100, time_window=60)

async def rate_limit_middleware(request: Request, call_next):
    """Middleware to enforce rate limiting"""
    client_ip = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_ip):
        app_logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={
                "error": "TOO_MANY_REQUESTS",
                "message": "Rate limit exceeded. Maximum 100 requests per minute."
            }
        )
    
    return await call_next(request)

# =========================================================
# PAGINATION HELPER
# =========================================================
class PaginationHelper:
    """Helper for pagination"""
    
    @staticmethod
    def get_pagination_params(page: int = 1, per_page: int = 10):
        """Validate pagination parameters"""
        page = max(1, page)
        per_page = max(1, min(100, per_page))  # Max 100 per page
        
        offset = (page - 1) * per_page
        
        return page, per_page, offset

# =========================================================
# DATA VALIDATION
# =========================================================
class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Validate phone format"""
        import re
        pattern = r'^[+]?[0-9\-\s()]{7,}$'
        return re.match(pattern, phone) is not None
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format"""
        import re
        pattern = r'^https?://'
        return re.match(pattern, url) is not None
