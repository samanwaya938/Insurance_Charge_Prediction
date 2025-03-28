import os
import logging
from datetime import datetime

# Configure logging only once
if not logging.root.handlers:
    LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)
    
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        filemode="a",  # Changed to append mode
        format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    logging.info("=" * 50)
    logging.info("Logging initialized")
    logging.info("=" * 50)