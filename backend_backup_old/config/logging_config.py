"""Comprehensive logging configuration for production"""
import logging
import logging.handlers
import os
from datetime import datetime

class ProductionLogger:
    """Production-grade logging setup"""
    
    def __init__(self, name: str, log_dir: str = "logs"):
        self.logger = logging.getLogger(name)
        self.log_dir = log_dir
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging with rotation and formatting"""
        # Create logs directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set logging level
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler - Rotating
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.log_dir, f"app_{today}.log")
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Error file handler
        error_file = os.path.join(self.log_dir, f"errors_{today}.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
    
    def get_logger(self):
        """Get configured logger"""
        return self.logger

# Create application logger
app_logger = ProductionLogger("ai_construction_system").get_logger()

def log_event(level: str, message: str, extra_data: dict = None):
    """Log an event with optional extra data"""
    if extra_data:
        message = f"{message} | Extra: {extra_data}"
    
    if level == "info":
        app_logger.info(message)
    elif level == "warning":
        app_logger.warning(message)
    elif level == "error":
        app_logger.error(message)
    elif level == "debug":
        app_logger.debug(message)
    elif level == "critical":
        app_logger.critical(message)
