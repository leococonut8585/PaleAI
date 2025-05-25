import os
import logging


def configure_logging():
    """Configure root logger from LOG_LEVEL environment variable."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level)

