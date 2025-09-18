"""Price-based feature transforms (pure functions)."""
from __future__ import annotations
import pandas as pd

def rolling_returns(df: pd.DataFrame, window: int, price_col: str = "Close") -> pd.DataFrame:
    """Compute log-return over a rolling window; pure function.
    Expects df indexed by date with a column per asset or a MultiIndex [date, ticker].
    """
    raise NotImplementedError

def rolling_volatility(df: pd.DataFrame, window: int, price_col: str = "Close") -> pd.DataFrame:
    """Rolling standard deviation of returns; pure function."""
    raise NotImplementedError
