from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass(frozen=True)
class ExpandingCVConfig:
    n_folds: int = 5
    embargo_days: int = 5  # business days


def expanding_cv_splits(
    dates: pd.DatetimeIndex,
    cfg: ExpandingCVConfig = ExpandingCVConfig(),
) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Create expanding-window CV splits on a single-ticker date index.
    - Validation windows are contiguous, equally sized chunks of the timeline tail.
    - Training for fold k uses all samples strictly before the k-th validation window,
      with an embargo that removes the last `embargo_days` business days before val_start.

    Returns:
        List of (train_dates, val_dates), both as DatetimeIndex subsets of `dates`.
    """
    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError("`dates` must be a DatetimeIndex (single ticker).")
    dates = dates.sort_values().unique()

    n = len(dates)
    if cfg.n_folds < 1 or n < cfg.n_folds * 10:
        # Heuristic: need ~10 samples per fold at minimum
        raise ValueError(f"Not enough samples ({n}) for {cfg.n_folds} folds.")

    # Split tail of data into n_folds contiguous validation chunks of (approximately) equal size
    fold_sizes = [n // cfg.n_folds] * cfg.n_folds
    for i in range(n % cfg.n_folds):
        fold_sizes[i] += 1

    # Build cumulative edges
    edges = [0]
    for fs in fold_sizes:
        edges.append(edges[-1] + fs)

    splits: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    for k in range(cfg.n_folds):
        val_start_idx = edges[k]
        val_end_idx = edges[k + 1]
        val_dates = dates[val_start_idx:val_end_idx]

        if len(val_dates) == 0:
            continue

        # Embargo: training ends embargo_days before validation start
        embargo = pd.tseries.offsets.BDay(cfg.embargo_days)
        train_end_date = val_dates[0] - embargo if cfg.embargo_days > 0 else val_dates[0]
        train_dates = dates[dates < train_end_date]

        if len(train_dates) == 0:
            # Skip fold if no training data yet (can happen for large embargo)
            continue

        splits.append((pd.DatetimeIndex(train_dates), pd.DatetimeIndex(val_dates)))

    if not splits:
        raise ValueError("No valid CV splits could be created; check data length and embargo.")
    return splits
