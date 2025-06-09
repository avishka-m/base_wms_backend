"""Logging configuration for the WMS Chatbot"""

import logging
import sys
from pathlib import Path

def setup_logging(
    level: int = logging.INFO,
    log_file: str = "chatbot.log",
    format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level
        log_file: Path to log file
        format_string: Format string for log messages
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    if log_path.parent.name and not log_path.parent.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file)
        ]
    )
    
    # Create and return logger
    logger = logging.getLogger("wms_chatbot")
    logger.setLevel(level)
    
    return logger

# Create default logger instance
logger = setup_logging() 