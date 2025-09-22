from __future__ import annotations
import logging
from logging.config import dictConfig

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s %(levelname)s %(name)s - %(message)s"},
        "uvicorn": {"format": "%(asctime)s %(levelname)s %(name)s - %(message)s"},
        "access":  {"format": "%(asctime)s %(levelname)s %(client_addr)s - '%(request_line)s' %(status_code)s"},
    },
    "handlers": {
        "default": {"class": "logging.StreamHandler", "formatter": "default"},
        "uvicorn": {"class": "logging.StreamHandler", "formatter": "uvicorn"},
        "access":  {"class": "logging.StreamHandler", "formatter": "access"},
    },
    "loggers": {
        "": {"handlers": ["default"], "level": "INFO"},
        "uvicorn": {"handlers": ["uvicorn"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        "fastapi": {"level": "INFO"},
        "app": {"level": "INFO"},
    },
}

def setup_logging() -> None:
    dictConfig(LOGGING_CONFIG)
    logging.getLogger("app").info("Logging configured.")
