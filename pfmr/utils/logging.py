"""pfmr.utils.logging — consistent logger factory."""
import logging
import os


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(levelname)-8s %(name)s — %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    level = os.environ.get("PFMR_LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger
