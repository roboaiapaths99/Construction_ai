"""
Application settings for AI Construction System
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings configuration"""
    
    # Application Settings
    APP_NAME = os.getenv("APP_NAME", "AI Construction Safety System")
    APP_VERSION = os.getenv("APP_VERSION", "1.0.0")
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Server Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8002))
    WORKERS = int(os.getenv("WORKERS", 1))
    RELOAD = os.getenv("RELOAD", "True").lower() == "true"
    
    # CORS Settings
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
    CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "True").lower() == "true"
    CORS_ALLOW_METHODS = os.getenv("CORS_ALLOW_METHODS", "*").split(",")
    CORS_ALLOW_HEADERS = os.getenv("CORS_ALLOW_HEADERS", "*").split(",")
    
    # File Upload Settings
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
    ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/images")
    
    # Video Settings
    VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")  # Default webcam
    VIDEO_FPS = int(os.getenv("VIDEO_FPS", 30))
    VIDEO_RESOLUTION = os.getenv("VIDEO_RESOLUTION", "1280x720")
    MAX_RECORDING_DURATION = int(os.getenv("MAX_RECORDING_DURATION", 300))  # 5 minutes
    
    # API Settings
    API_PREFIX = os.getenv("API_PREFIX", "/api/v1")
    API_TIMEOUT = int(os.getenv("API_TIMEOUT", 30))
    RATE_LIMIT = os.getenv("RATE_LIMIT", "100/minute")
    
    # Security Settings
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 30))
    
    # Logging Settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_FILE = os.getenv("LOG_FILE", "logs/server.log")
    LOG_MAX_SIZE = int(os.getenv("LOG_MAX_SIZE", 10 * 1024 * 1024))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))
    
    # Monitoring Settings
    METRICS_ENABLED = os.getenv("METRICS_ENABLED", "True").lower() == "true"
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", 30))
    PERFORMANCE_MONITORING = os.getenv("PERFORMANCE_MONITORING", "True").lower() == "true"
    
    # Cache Settings
    CACHE_TTL = int(os.getenv("CACHE_TTL", 300))  # 5 minutes
    CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", 1000))
    
    # WebSocket Settings
    WS_ENABLED = os.getenv("WS_ENABLED", "True").lower() == "true"
    WS_HEARTBEAT_INTERVAL = int(os.getenv("WS_HEARTBEAT_INTERVAL", 30))
    WS_MAX_CONNECTIONS = int(os.getenv("WS_MAX_CONNECTIONS", 100))
    
    @staticmethod
    def get_cors_config():
        """Get CORS configuration"""
        return {
            "allow_origins": Settings.CORS_ORIGINS,
            "allow_credentials": Settings.CORS_ALLOW_CREDENTIALS,
            "allow_methods": Settings.CORS_ALLOW_METHODS,
            "allow_headers": Settings.CORS_ALLOW_HEADERS
        }
    
    @staticmethod
    def get_app_info():
        """Get application information"""
        return {
            "name": Settings.APP_NAME,
            "version": Settings.APP_VERSION,
            "environment": Settings.ENVIRONMENT,
            "debug": Settings.DEBUG
        }
    
    @staticmethod
    def validate_settings():
        """Validate critical settings"""
        errors = []
        
        # Validate required settings
        if Settings.SECRET_KEY == "your-secret-key-change-in-production" and Settings.ENVIRONMENT == "production":
            errors.append("SECRET_KEY must be changed in production")
        
        if Settings.PORT < 1 or Settings.PORT > 65535:
            errors.append("PORT must be between 1 and 65535")
        
        if Settings.JWT_EXPIRE_MINUTES < 1:
            errors.append("JWT_EXPIRE_MINUTES must be at least 1")
        
        return errors

# Development settings
class DevelopmentSettings(Settings):
    """Development environment settings"""
    DEBUG = True
    RELOAD = True
    LOG_LEVEL = "DEBUG"

# Production settings
class ProductionSettings(Settings):
    """Production environment settings"""
    DEBUG = False
    RELOAD = False
    LOG_LEVEL = "WARNING"
    
    # Production security settings
    CORS_ORIGINS = ["https://yourdomain.com"]
    RATE_LIMIT = "50/minute"

# Test settings
class TestSettings(Settings):
    """Test environment settings"""
    DEBUG = True
    TESTING = True
    DATABASE_URL = "sqlite:///test.db"

# Settings factory
def get_settings():
    """Get settings based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    elif env == "test":
        return TestSettings()
    else:
        return DevelopmentSettings()

# Global settings instance
settings = get_settings()

if __name__ == "__main__":
    # Validate settings
    print("🔍 Validating application settings...")
    errors = Settings.validate_settings()
    
    if errors:
        print("❌ Settings validation failed:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ Settings validation passed")
    
    # Display current settings
    print("\n📋 Current Settings:")
    print(f"  App Name: {Settings.APP_NAME}")
    print(f"  Version: {Settings.APP_VERSION}")
    print(f"  Environment: {Settings.ENVIRONMENT}")
    print(f"  Debug: {Settings.DEBUG}")
    print(f"  Host: {Settings.HOST}:{Settings.PORT}")
    print(f"  Log Level: {Settings.LOG_LEVEL}")
    print(f"  Database: {os.getenv('DB_NAME', 'safety_ai')}")
