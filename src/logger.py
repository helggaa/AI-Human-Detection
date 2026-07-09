"""
Project Logger
==============

Central logging configuration for the project.
"""

from __future__ import annotations

import logging

from src.config import LOG_DIR

LOG_FILE = LOG_DIR / "project.log"

LOGGER_NAME = "AIHumanDetection"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger() -> logging.Logger:
    """
    Return the project logger.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(LOGGER_NAME)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        LOG_FORMAT,
        DATE_FORMAT,
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
        delay=True,
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()

__all__ = [
    "get_logger",
    "logger",
]
