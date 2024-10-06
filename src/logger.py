import logging
import os
from datetime import datetime

# Corrected the typo from strtime to strftime
LOG_FILE = f"{datetime.now().strftime('%d_%m_%y_%H_%M_%S')}.log"

# Create the logs directory in the current working directory
logs_path = os.path.join(os.getcwd(), "logs")

# Create the directory if it doesn't exist
os.makedirs(logs_path, exist_ok = True)

# Create the complete log file path
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Set up logging configuration
logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)
