"""Constrained optimizers for portfolio weights — stub."""

from __future__ import annotations
import numpy as np


def optimize_weights(
    mu: np.ndarray, Sigma: np.ndarray, *, long_only: bool = True, w_cap: float | None = None
) -> np.ndarray:
    raise NotImplementedError
