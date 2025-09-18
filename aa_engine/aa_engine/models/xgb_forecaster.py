"""XGBoost forecaster (skeleton)."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class ForecastResult:
    preds: pd.DataFrame  # index=date, columns=ticker (target horizon returns)
    diagnostics: dict

def fit_predict_xgb(features: pd.DataFrame, target: pd.Series, config: dict) -> ForecastResult:
    """Train XGB on expanding CV and return OOS predictions (stub)."""
    raise NotImplementedError
