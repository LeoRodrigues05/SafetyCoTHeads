"""Minimal logging setup."""
from __future__ import annotations
import logging
import sys


def get_logger(name: str = "safety_cot_heads", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(h)
    logger.setLevel(level)
    logger.propagate = False
    return logger
