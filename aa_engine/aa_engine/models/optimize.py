from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

import json
import math
import time

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("Please `pip install xgboost` to use the optimizer.") from e

from .tuning import ExpandingCVConfig, expanding_cv_splits

# Reuse helpers from the forecaster (keeps behavior identical).
try:
    # mypy: ignore-errors (impls may have looser types)
    from .xgb_forecaster import _normalize_booster_params as _normalize_booster_params_impl  # type: ignore[attr-defined]
    from .xgb_forecaster import _predict_booster as _predict_booster_impl  # type: ignore[attr-defined]
except Exception:
    _normalize_booster_params_impl = None  # type: ignore[assignment]
    _predict_booster_impl = None  # type: ignore[assignment]

    def _predict_booster(
        booster: xgb.Booster, dmat: xgb.DMatrix, num_boost_round: int
    ) -> np.ndarray:
        """Stable wrapper around the project predictor."""
        if _predict_booster_impl is not None:
            return _predict_booster_impl(booster, dmat, num_boost_round)  # type: ignore[misc]
        # --- fallback ---
        try:
            best_iter = getattr(booster, "best_iteration", None)
            if isinstance(best_iter, int) and best_iter >= 0:
                return booster.predict(dmat, iteration_range=(0, best_iter + 1))
        except Exception:
            pass
        try:
            ntree_limit = getattr(booster, "best_ntree_limit", None) or num_boost_round
            return booster.predict(dmat, ntree_limit=ntree_limit)
        except Exception:
            pass
        return booster.predict(dmat)

    def _normalize_params(config: Mapping[str, Any]) -> Tuple[Dict[str, Any], int, int]:
        """Stable wrapper around the project normalizer."""
        if _normalize_booster_params_impl is not None:
            # Ensure Mapping works with any impl that expects dict
            return _normalize_booster_params_impl(dict(config))  # type: ignore[misc]
        # --- fallback kept consistent with forecaster ---
        mcfg = dict(config.get("model", {}) or {})
        user = dict(mcfg.get("params", {}) or {})

        num_boost_round = int(user.pop("n_estimators", 500))
        params: Dict[str, Any] = {
            "objective": "reg:squarederror",
            "eta": user.pop("learning_rate", 0.05),
            "max_depth": user.pop("max_depth", 4),
            "subsample": user.pop("subsample", 0.8),
            "colsample_bytree": user.pop("colsample_bytree", 0.8),
            "nthread": user.pop("n_jobs", -1),
            "seed": int(config.get("seed", 42)),
            "eval_metric": user.pop("eval_metric", "rmse"),
        }
        if "random_state" in user:
            params["seed"] = int(user.pop("random_state"))
        user.pop("early_stopping_rounds", None)
        user.pop("verbosity", None)
        params.update(user)

        early_stop = int(mcfg.get("early_stopping_rounds", 50))
        return params, num_boost_round, early_stop


@dataclass(frozen=True)
class TrialResult:
    trial: int
    params: Dict[str, Any]
    score: float  # lower is better (aggregated)
    per_ticker_rmse: Dict[str, float]
    per_ticker_nrmse: Dict[str, float]
    n_val_obs: int


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    leaderboard: pd.DataFrame  # trial, score, n_val_obs, params_json, rmse_*, nrmse_*
    diagnostics: Dict[str, Any]


# ------------------------------ sampling utilities -----------------------------------------


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def _sample_from_space(space: Mapping[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    """
    Draw a single parameter set from the config-defined search space.

    Supported forms per key:
    - [low, high]        -> uniform float in [low, high]
    - {"low": a, "high": b, "type": "int"} -> integer uniform
    - {"low": a, "high": b, "type": "log"} -> log-uniform float in [a, b]
    - [v1, v2, v3, ...]  -> random choice from list (categorical)
    """
    out: Dict[str, Any] = {}
    for k, spec in space.items():
        if isinstance(spec, dict):
            low = spec.get("low")
            high = spec.get("high")
            tp = str(spec.get("type", "float"))
            if low is None or high is None:
                raise ValueError(f"search_space['{k}'] requires 'low' and 'high'")
            if tp == "int":
                lo_i, hi_i = int(low), int(high)
                if hi_i < lo_i:
                    lo_i, hi_i = hi_i, lo_i
                out[k] = int(rng.integers(lo_i, hi_i + 1))
            elif tp == "log":
                lo_f, hi_f = float(low), float(high)
                if lo_f <= 0 or hi_f <= 0:
                    raise ValueError(f"log-uniform range for '{k}' must be positive")
                val = math.exp(rng.uniform(math.log(lo_f), math.log(hi_f)))
                out[k] = float(val)
            else:
                lo_f, hi_f = float(low), float(high)
                out[k] = float(rng.uniform(lo_f, hi_f))
        elif (
            isinstance(spec, (list, tuple))
            and len(spec) == 2
            and all(isinstance(x, (int, float)) for x in spec)
        ):
            lo_f, hi_f = float(spec[0]), float(spec[1])
            if hi_f < lo_f:
                lo_f, hi_f = hi_f, lo_f
            out[k] = float(rng.uniform(lo_f, hi_f))
        elif isinstance(spec, (list, tuple)) and len(spec) > 0:
            idx = int(rng.integers(0, len(spec)))
            out[k] = spec[idx]
        else:
            out[k] = spec
    return out


# ------------------------------ scoring core -----------------------------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _score_param_set_for_ticker(
    df_xy: pd.DataFrame,  # columns: features..., "_y"
    params: Dict[str, Any],
    num_boost_round: int,
    early_stop: int,
    cv_cfg: ExpandingCVConfig,
) -> Tuple[float, int]:
    """
    Train/evaluate one parameter set on a single ticker via expanding CV.
    Returns (rmse_weighted_by_obs, total_validation_observations).
    """
    dates = df_xy.index
    splits = expanding_cv_splits(dates, cfg=cv_cfg)

    feat_cols = [c for c in df_xy.columns if c != "_y"]
    feature_names = [str(c) for c in feat_cols]

    rmse_sum = 0.0
    n_obs = 0

    for train_dates, val_dates in splits:
        X_train, y_train = df_xy.loc[train_dates, feat_cols], df_xy.loc[train_dates, "_y"]
        X_val, y_val = df_xy.loc[val_dates, feat_cols], df_xy.loc[val_dates, "_y"]
        if len(X_train) == 0 or len(X_val) == 0:
            continue

        dtrain = xgb.DMatrix(
            X_train.values,
            label=y_train.values,
            feature_names=feature_names,
            nthread=params.get("nthread", -1),
        )
        dval = xgb.DMatrix(
            X_val.values,
            label=y_val.values,
            feature_names=feature_names,
            nthread=params.get("nthread", -1),
        )

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=early_stop if early_stop > 0 else None,
            verbose_eval=False,
        )

        y_pred = _predict_booster(booster, dval, num_boost_round)
        rmse_fold = _rmse(y_val.values, y_pred)
        rmse_sum += rmse_fold * len(y_val)
        n_obs += len(y_val)

    if n_obs == 0:
        return (float("nan"), 0)
    return (rmse_sum / n_obs, n_obs)


def _prepare_xy_for_ticker(
    features: pd.DataFrame,
    target: pd.Series,
    ticker: str,
    min_history_days: int | None = 120,
) -> pd.DataFrame:
    """Slice and align X/y for a ticker; drop rows with NaNs; apply warmup trim."""
    X = features.xs(ticker, level="ticker").sort_index()
    y = target.xs(ticker, level="ticker").sort_index()
    df = X.join(y.to_frame("_y"), how="inner").dropna(how="any")
    if min_history_days and min_history_days > 0 and df.shape[0] > min_history_days:
        df = df.iloc[min_history_days:]
    return df


# ------------------------------ public API --------------------------------------------------


def tune_xgb_params(
    features: pd.DataFrame,  # [date, ticker] → features
    target: pd.Series,  # [date, ticker] → forward log return
    tuning_cfg: Mapping[str, Any],
    cv_cfg: ExpandingCVConfig,
) -> TuningResult:
    """
    Random-search XGBoost params using the same expanding CV + embargo as production.
    Aggregates (N)RMSE across specified tickers with equal or n_obs weighting.
    """
    if not isinstance(features.index, pd.MultiIndex) or not isinstance(target.index, pd.MultiIndex):
        raise ValueError("features and target must both be MultiIndex ['date', 'ticker'].")

    # Aggregation settings
    agg = dict(tuning_cfg.get("aggregation", {}))
    unit = str(agg.get("unit", "rmse")).lower()  # 'rmse' or 'nrmse'
    weight_mode = str(agg.get("weight", "n_obs")).lower()  # 'equal' or 'n_obs'
    reducer = str(agg.get("reducer", "mean")).lower()  # currently 'mean' only

    seed = int(tuning_cfg.get("seed", 42))
    n_trials = int(tuning_cfg.get("n_trials", 20))
    space = dict(tuning_cfg.get("search_space", {}))
    calib_tickers: List[str] = list(tuning_cfg.get("tickers", [])) or sorted(
        set(features.index.get_level_values("ticker"))
        & set(target.index.get_level_values("ticker"))
    )
    min_history_days = int(tuning_cfg.get("min_history_days", 120))

    # Base config passed to normalizer (so seed, early_stopping from tuning_cfg can apply)
    base_cfg: Dict[str, Any] = {
        "seed": seed,
        "model": {
            "params": {},
            "early_stopping_rounds": int(tuning_cfg.get("early_stopping_rounds", 50)),
        },
    }

    rng = _rng(seed)
    trials: List[TrialResult] = []

    # Pre-slice XY per ticker once
    per_ticker_xy: Dict[str, pd.DataFrame] = {}
    for tkr in calib_tickers:
        try:
            per_ticker_xy[tkr] = _prepare_xy_for_ticker(
                features, target, tkr, min_history_days=min_history_days
            )
        except KeyError:
            continue

    t_start = time.time()
    for trial in range(1, n_trials + 1):
        sampled = _sample_from_space(space, rng)

        # Inject sampled params into base_cfg and normalize for Booster API
        cfg_trial = dict(base_cfg)
        cfg_trial["model"] = dict(base_cfg["model"])
        cfg_trial["model"]["params"] = dict(sampled)

        params, num_boost_round, early_stop = _normalize_params(cfg_trial)

        per_ticker_rmse: Dict[str, float] = {}
        per_ticker_nrmse: Dict[str, float] = {}
        total_w = 0.0
        weighted_sum = 0.0

        for tkr, df_xy in per_ticker_xy.items():
            if df_xy.empty or df_xy.shape[0] < max(40, cv_cfg.n_folds * 10):
                continue

            rmse_t, n_obs = _score_param_set_for_ticker(
                df_xy, params, num_boost_round, early_stop, cv_cfg
            )
            if n_obs <= 0 or math.isnan(rmse_t):
                continue

            # Per-ticker normalization
            std_y = float(df_xy["_y"].std(ddof=1) or 0.0)
            nrmse_t = rmse_t / (std_y if std_y > 1e-12 else 1.0)

            per_ticker_rmse[tkr] = rmse_t
            per_ticker_nrmse[tkr] = nrmse_t

            # Aggregation unit & weight
            val = nrmse_t if unit == "nrmse" else rmse_t
            w = 1.0 if weight_mode == "equal" else float(n_obs)
            weighted_sum += val * w
            total_w += w

        score = (weighted_sum / total_w) if total_w > 0 else float("inf")

        trials.append(
            TrialResult(
                trial=trial,
                params=sampled,
                score=score,
                per_ticker_rmse=per_ticker_rmse,
                per_ticker_nrmse=per_ticker_nrmse,
                n_val_obs=int(
                    sum(len(df_xy) for df_xy in per_ticker_xy.values() if not df_xy.empty)
                ),
            )
        )

    # Build leaderboard DataFrame
    rows: List[Dict[str, Any]] = []
    all_tickers = sorted(per_ticker_xy.keys())
    for tr in trials:
        row: Dict[str, Any] = {
            "trial": tr.trial,
            "score": tr.score,
            "n_val_obs": tr.n_val_obs,
            "params_json": json.dumps(tr.params, sort_keys=True),
        }
        for t in all_tickers:
            row[f"rmse_{t}"] = tr.per_ticker_rmse.get(t, np.nan)
            row[f"nrmse_{t}"] = tr.per_ticker_nrmse.get(t, np.nan)
        rows.append(row)

    leaderboard = (
        pd.DataFrame(rows).sort_values(["score", "n_val_obs", "trial"]).reset_index(drop=True)
    )

    # Best params (handle no-valid-trial corner case)
    if leaderboard.empty or not np.isfinite(leaderboard.loc[0, "score"]):
        best_params: Dict[str, Any] = {}
    else:
        best_params = json.loads(leaderboard.loc[0, "params_json"])

    diagnostics = {
        "seed": seed,
        "n_trials": n_trials,
        "elapsed_sec": time.time() - t_start,
        "aggregation": {"unit": unit, "weight": weight_mode, "reducer": reducer},
        "calibration_tickers": all_tickers,
        "cv": cv_cfg.__dict__,
        "space": space,
    }
    return TuningResult(best_params=best_params, leaderboard=leaderboard, diagnostics=diagnostics)
