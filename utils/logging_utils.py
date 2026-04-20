"""
utils/logging_utils.py

Minimal structured logging setup for consistent logs across modules.
"""

from __future__ import annotations

import logging
from typing import Optional


def setup_logging(level: str = "INFO") -> None:
    if logging.getLogger().handlers:
        # Avoid duplicate handlers when called multiple times.
        return

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else __name__)
