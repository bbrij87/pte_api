import logging
import os
from logging.handlers import RotatingFileHandler

# Directory for all log files
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# General error log file
error_log_file = os.path.join(log_dir, "api_errors.log")

# Set up the general error logger
error_logger = logging.getLogger("error_logger")
error_logger.setLevel(logging.ERROR)
error_handler = RotatingFileHandler(
    error_log_file, maxBytes=1e6, backupCount=5)  # 1 MB per file, 5 backups
error_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'))
error_logger.addHandler(error_handler)


def get_logger(api_name):
    """
    Creates or retrieves a logger specific to an API, with log rotation.
    :param api_name: Name of the API for which to create a log file (e.g., 'speak_words').
    :return: Configured logger for the API.
    """
    # Define the log file for the specific API
    api_log_file = os.path.join(log_dir, f"{api_name}.log")

    # Set up the logger for the API
    logger = logging.getLogger(api_name)
    logger.setLevel(logging.INFO)

    # Check if handler already exists to prevent duplicate handlers
    if not logger.hasHandlers():
        # Rotating file handler for the specific API log
        handler = RotatingFileHandler(
            api_log_file, maxBytes=1e6, backupCount=5)  # 1 MB per file, 5 backups
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    return logger


def log_error(message):
    """
    Logs a message to the general error log.
    :param message: The error message to log.
    """
    error_logger.error(message)


def log_info(api_name, message):
    """
    Logs an informational message to a specific API's log file.
    :param api_name: Name of the API for which to log (e.g., 'speak_words').
    :param message: The message to log.
    """
    logger = get_logger(api_name)
    logger.info(message)
