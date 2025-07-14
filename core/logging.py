import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from pythonjsonlogger import jsonlogger
from .config import settings

def setup_logging():
    """Setup logging configuration for the application."""
    
    # Creating the logs directory if it doesn't exist
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if not settings.debug else logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s' 
)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler for general logs
    log_file = settings.logs_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # JSON handler for structured logging
    json_file = settings.logs_dir / f"app_json_{datetime.now().strftime('%Y%m%d')}.log"
    json_handler = logging.handlers.RotatingFileHandler(
        json_file, maxBytes=10*1024*1024, backupCount=5
    )
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s'
    )
    json_handler.setFormatter(json_formatter)
    logger.addHandler(json_handler)
    
    # Error handler for critical errors
    error_file = settings.logs_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file, maxBytes=10*1024*1024, backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)