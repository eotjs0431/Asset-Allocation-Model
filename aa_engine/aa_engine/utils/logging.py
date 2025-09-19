"""Structured logging — minimal."""

from __future__ import annotations
import logging
import sys


def get_logger(name: str | None = None) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(name or "aa-engine")
