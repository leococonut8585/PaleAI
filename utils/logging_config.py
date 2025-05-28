import os
import logging


def configure_logging():
    """Configure root logger from LOG_LEVEL environment variable with a more detailed format."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    
    # Define a more detailed log format
    log_format = "%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s"
    
    logging.basicConfig(level=level, format=log_format)

    # Example of how to get a logger in other modules:
    # import logging
    # logger = logging.getLogger(__name__)
    # logger.info("This is an info message.")
    # logger.debug("This is a debug message.")

