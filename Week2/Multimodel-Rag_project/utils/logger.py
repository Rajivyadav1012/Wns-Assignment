"""
Logging configuration for the application
Step 2: Logging & Observability
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from config.settings import settings

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: grey + format_string + reset,
        logging.INFO: blue + format_string + reset,
        logging.WARNING: yellow + format_string + reset,
        logging.ERROR: red + format_string + reset,
        logging.CRITICAL: bold_red + format_string + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(
    name: str,
    log_level: Optional[str] = None,
    log_file: bool = True
) -> logging.Logger:
    """
    Setup and configure logger
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Whether to write logs to file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level
    level = log_level or settings.LOG_LEVEL
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console Handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)
    
    # File Handler (if enabled)
    if log_file:
        log_filename = f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        log_path = settings.LOGS_PATH / log_filename
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

# Create main application logger
app_logger = setup_logger("multimodal_rag")

# Test function
def test_logger():
    """Test logger functionality"""
    logger = setup_logger("test_logger")
    
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")
    
    print("\nLogger test complete! Check logs/ directory for log files.")

if __name__ == "__main__":
    test_logger()