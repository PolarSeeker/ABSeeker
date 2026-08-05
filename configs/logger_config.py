import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

def init_logging(level=logging.INFO):
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f"{log_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level) 
    
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        delay=False,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(file_handler)
    
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger('numexpr.utils').setLevel(logging.WARNING)
    return root_logger