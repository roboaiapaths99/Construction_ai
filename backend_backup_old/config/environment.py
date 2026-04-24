"""Production environment configuration"""
import os
from typing import Optional

class EnvironmentConfig:
    """Environment configuration management"""
    
    # =========================================================
    # ENVIRONMENT
    # =========================================================
    ENV = os.getenv("ENV", "development").lower()
    DEBUG = ENV == "development"
    
    # =========================================================
    # API CONFIGURATION
    # =========================================================
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_WORKERS = int(os.getenv("API_WORKERS", "4"))  # For production
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
    
    # =========================================================
    # DATABASE CONFIGURATION
    # =========================================================
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = int(os.getenv("DB_PORT", "3306"))
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "safety_ai")
    DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
    DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    
    # =========================================================
    # SECURITY
    # =========================================================
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-in-production")
    JWT_ALGORITHM = "HS256"
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "30"))
    
    # =========================================================
    # CORS CONFIGURATION
    # =========================================================
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    CORS_CREDENTIALS = True
    CORS_METHODS = ["*"]
    CORS_HEADERS = ["*"]
    
    # =========================================================
    # LOGGING
    # =========================================================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR = os.getenv("LOG_DIR", "logs")
    LOG_FORMAT = "json" if ENV == "production" else "text"
    
    # =========================================================
    # AI MODEL
    # =========================================================
    MODEL_PATH = os.getenv("MODEL_PATH", "ai/models/yolov8n.pt")
    MODEL_CONFIDENCE = float(os.getenv("MODEL_CONFIDENCE", "0.5"))
    MODEL_IOU = float(os.getenv("MODEL_IOU", "0.45"))
    
    # =========================================================
    # FILE UPLOAD
    # =========================================================
    MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "10485760"))  # 10MB
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif"}
    
    # =========================================================
    # RATE LIMITING
    # =========================================================
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    
    # =========================================================
    # EMAIL CONFIGURATION (for alerts)
    # =========================================================
    SMTP_SERVER = os.getenv("SMTP_SERVER", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    SMTP_FROM = os.getenv("SMTP_FROM", "noreply@constructionsafety.ai")
    
    # =========================================================
    # ALERT CONFIGURATION
    # =========================================================
    ALERT_ENABLED = os.getenv("ALERT_ENABLED", "true").lower() == "true"
    ALERT_EMAIL_ENABLED = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"
    ALERT_WEBHOOK_ENABLED = os.getenv("ALERT_WEBHOOK_ENABLED", "false").lower() == "true"
    ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
    ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.7"))
    
    # =========================================================
    # PERSISTENCE
    # =========================================================
    DATA_DIR = os.getenv("DATA_DIR", "data")
    VIOLATIONS_DIR = os.path.join(DATA_DIR, "images/violations")
    ALERTS_DIR = os.path.join(DATA_DIR, "alerts")
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        if cls.ENV not in ["development", "testing", "production"]:
            errors.append(f"Invalid ENV: {cls.ENV}")
        
        if cls.API_PORT < 1 or cls.API_PORT > 65535:
            errors.append(f"Invalid API_PORT: {cls.API_PORT}")
        
        if cls.SECRET_KEY == "change-this-in-production" and cls.ENV == "production":
            errors.append("SECRET_KEY must be changed for production")
        
        if cls.ALERT_WEBHOOK_ENABLED and not cls.ALERT_WEBHOOK_URL:
            errors.append("ALERT_WEBHOOK_URL required when webhook alerts enabled")
        
        return errors
    
    @classmethod
    def get_connection_string(cls):
        """Get database connection string"""
        return (f"mysql+pymysql://{cls.DB_USER}:{cls.DB_PASSWORD}"
                f"@{cls.DB_HOST}:{cls.DB_PORT}/{cls.DB_NAME}")

# Create default instance
env_config = EnvironmentConfig()
