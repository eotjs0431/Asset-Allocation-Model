"""Deterministic seeds — stub."""

from __future__ import annotations
import numpy as np
import random
import os


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
