from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    """Configuration for v1 price features."""

    returns_windows: tuple[int, ...] = (5, 20, 60)
    vol_windows: tuple[int, ...] = (20, 60)
    momentum_windows: tuple[int, ...] = (20, 60, 120)
    price_col: str = "AdjClose"


# ----------------------------- core safe helpers -------------------------------------------


def _ensure_panel(panel: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    Ensure `panel` is MultiIndex [date, ticker] with the given price column present.
    No copies unless needed; index is sorted ascending by date then ticker.
    """
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.names != ["date", "ticker"]:
        raise ValueError("panel must be indexed by MultiIndex ['date', 'ticker']")
    if price_col not in panel.columns:
        raise KeyError(f"panel is missing required column '{price_col}'")
    # sort and enforce DateTimeIndex
    if not isinstance(panel.index.get_level_values("date"), pd.DatetimeIndex):
        panel = panel.copy()
        panel.index = panel.index.set_levels(
            [pd.to_datetime(panel.index.levels[0]).tz_localize(None), panel.index.levels[1]]
        )
    return panel.sort_index()


def _by_ticker(panel: pd.DataFrame) -> pd.core.groupby.generic.DataFrameGroupBy:
    return panel.groupby(level="ticker", sort=False, group_keys=False)


# ----------------------------- primitives (pure) -------------------------------------------


def daily_log_returns(panel: pd.DataFrame, price_col: str = "AdjClose") -> pd.Series:
    """
    Daily log return per ticker: ln(P_t / P_{t-1})
    Returns a Series aligned to panel index (MultiIndex).
    """
    p = _ensure_panel(panel, price_col)
    lr = _by_ticker(p)[price_col].apply(lambda s: np.log(s) - np.log(s.shift(1)))
    lr.name = "ret_1d_log"
    return lr


def rolling_returns(panel: pd.DataFrame, window: int, price_col: str = "AdjClose") -> pd.Series:
    """
    Rolling log return over `window` days per ticker: ln(P_t / P_{t-window})
    (Not a sum of daily returns to avoid accumulation drift when NaNs exist.)
    """
    p = _ensure_panel(panel, price_col)
    rr = _by_ticker(p)[price_col].apply(lambda s: np.log(s) - np.log(s.shift(window)))
    rr.name = f"ret_{window}d_log"
    return rr


def rolling_volatility(panel: pd.DataFrame, window: int, price_col: str = "AdjClose") -> pd.Series:
    """
    Rolling volatility of daily log returns (stdev over `window`) per ticker.
    """
    p = _ensure_panel(panel, price_col)
    r1 = daily_log_returns(p, price_col=price_col)
    vol = _by_ticker(r1.to_frame()).apply(
        lambda df: df["ret_1d_log"].rolling(window, min_periods=window).std()
    )
    vol.name = f"vol_{window}d"
    return vol


def moving_average(panel: pd.DataFrame, window: int, price_col: str = "AdjClose") -> pd.Series:
    """
    Simple moving average of price over `window`.
    """
    p = _ensure_panel(panel, price_col)
    ma = _by_ticker(p)[price_col].apply(lambda s: s.rolling(window, min_periods=window).mean())
    ma.name = f"ma_{window}d"
    return ma


def momentum_ma_ratio(
    panel: pd.DataFrame, fast: int, slow: int, price_col: str = "AdjClose"
) -> pd.Series:
    """
    Momentum feature: fast MA / slow MA - 1  (fractional premium).
    """
    if fast >= slow:
        raise ValueError("`fast` must be < `slow`")
    ma_f = moving_average(panel, fast, price_col=price_col)
    ma_s = moving_average(panel, slow, price_col=price_col)
    out = (ma_f / ma_s) - 1.0
    out.name = f"mom_ma_ratio_{fast}_{slow}"
    return out


def zscore(panel: pd.DataFrame, window: int, price_col: str = "AdjClose") -> pd.Series:
    """
    Z-score of price relative to rolling mean over `window`: (P - MA) / rolling std.
    Uses rolling std of price changes to avoid division by near-zero.
    """
    p = _ensure_panel(panel, price_col)
    roll_mean = _by_ticker(p)[price_col].apply(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    # use rolling std of price differences for robustness
    roll_std = _by_ticker(p)[price_col].apply(
        lambda s: s.diff().rolling(window, min_periods=window).std()
    )
    z = (p[price_col] - roll_mean) / roll_std.replace(0.0, np.nan)
    z.name = f"z_{window}d"
    return z


# ----------------------------- target construction (pure) ----------------------------------


def make_forward_return_target(
    panel: pd.DataFrame,
    horizon_days: int,
    price_col: str = "AdjClose",
) -> pd.Series:
    """
    Forward log return target: y_t = ln(P_{t+H} / P_t) per ticker.
    Implemented by shifting price -H and computing ln ratio (leakage-safe).
    """
    p = _ensure_panel(panel, price_col)

    def _fwd(s: pd.Series) -> pd.Series:
        return np.log(s.shift(-horizon_days)) - np.log(s)

    y = _by_ticker(p)[price_col].apply(_fwd)
    y.name = f"y_fwd_log_{horizon_days}d"
    return y


# ----------------------------- feature matrix assembly (pure) ------------------------------


def build_features(
    panel: pd.DataFrame,
    spec: FeatureSpec | Mapping[str, Iterable[int]] = FeatureSpec(),
) -> pd.DataFrame:
    """
    Build the v1 feature matrix given a price panel and a FeatureSpec.
    Returns a DataFrame indexed by MultiIndex [date, ticker] with flat column names.

    Policy:
      - No forward-fill across non-trading days.
      - All features computed per ticker independently.
      - Caller is responsible for aligning with target and dropping NaNs as needed.
    """
    if isinstance(spec, Mapping):
        # Coerce types explicitly for mypy
        returns_windows = tuple(int(x) for x in spec.get("returns_windows", (5, 20, 60)))
        vol_windows = tuple(int(x) for x in spec.get("vol_windows", (20, 60)))
        momentum_windows = tuple(int(x) for x in spec.get("momentum_windows", (20, 60, 120)))
        price_col = str(spec.get("price_col", "AdjClose"))

        spec = FeatureSpec(
            returns_windows=returns_windows,
            vol_windows=vol_windows,
            momentum_windows=momentum_windows,
            price_col=price_col,
        )

    p = _ensure_panel(panel, spec.price_col)

    cols: dict[str, pd.Series] = {}

    # Rolling log returns
    for w in spec.returns_windows:
        cols[f"ret_{w}d_log"] = rolling_returns(p, w, price_col=spec.price_col)

    # Volatility
    for w in spec.vol_windows:
        cols[f"vol_{w}d"] = rolling_volatility(p, w, price_col=spec.price_col)

    # Momentum (MA ratios) and Z-scores
    # Use adjacent pairs for ratios where possible; also include individual z-scores
    mws = sorted(set(spec.momentum_windows))
    for i in range(len(mws) - 1):
        fast, slow = mws[i], mws[i + 1]
        cols[f"mom_ma_ratio_{fast}_{slow}"] = momentum_ma_ratio(
            p, fast, slow, price_col=spec.price_col
        )
    for w in mws:
        cols[f"z_{w}d"] = zscore(p, w, price_col=spec.price_col)

    # Assemble
    feat = pd.concat(cols.values(), axis=1)
    feat.columns = list(cols.keys())
    # Preserve index name & order
    feat = feat.sort_index()
    return feat


# ----------------------------- convenience utilities ---------------------------------------


def align_features_and_target(
    features: pd.DataFrame,
    target: pd.Series,
    min_history_days: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Align features and target on the same MultiIndex, dropping rows with NaNs in either.
    Optionally enforce a minimum lookback length by dropping early rows per ticker.

    Args:
        features: DataFrame [date, ticker] -> feature columns
        target:   Series    [date, ticker]
        min_history_days: if set, drop the first N rows per ticker (warm-up).

    Returns:
        (X, y) aligned and NaN-free.
    """
    if not isinstance(features.index, pd.MultiIndex) or not isinstance(target.index, pd.MultiIndex):
        raise ValueError("features and target must both be MultiIndex ['date', 'ticker']")

    Xy = features.join(target.to_frame("_y"), how="inner")
    if min_history_days is not None and min_history_days > 0:

        def _drop_head(g: pd.DataFrame) -> pd.DataFrame:
            return g.iloc[min_history_days:]

        Xy = Xy.groupby(level="ticker", group_keys=False).apply(_drop_head)

    Xy = Xy.dropna(how="any")
    y = Xy.pop("_y")
    return Xy, y
