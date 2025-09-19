# aa_engine/models/xgb_forecaster.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import xgboost as xgb
except ImportError as e:
    raise ImportError("Please `pip install xgboost` to use the XGB forecaster.") from e

from .tuning import ExpandingCVConfig, expanding_cv_splits
from ..utils.seeds import set_global_seed


@dataclass
class ForecastResult:
    preds: pd.DataFrame  # index=date, columns=ticker (OOS predictions)
    diagnostics: dict  # metrics per ticker/fold, params, feature importance, etc.


def _normalize_booster_params(config: dict) -> tuple[dict, int, int]:
    """Map sklearn-style keys to Booster API, drop wrapper-only keys, and extract num_boost_round."""
    mcfg = config.get("model", {}) or {}
    user = dict(mcfg.get("params", {}) or {})

    # Extract num_boost_round and remove n_estimators from params to avoid warnings
    num_boost_round = int(user.pop("n_estimators", 500))

    # Base params (Booster names)
    params: Dict[str, object] = {
        "objective": "reg:squarederror",
        "eta": user.pop("learning_rate", 0.05),
        "max_depth": user.pop("max_depth", 4),
        "subsample": user.pop("subsample", 0.8),
        "colsample_bytree": user.pop("colsample_bytree", 0.8),
        "nthread": user.pop("n_jobs", -1),
        "seed": int(config.get("seed", 42)),
        "eval_metric": user.pop("eval_metric", "rmse"),
    }

    # Map/clean common wrapper keys
    if "random_state" in user:
        # prefer Booster's 'seed'
        params["seed"] = int(user.pop("random_state"))

    # Remove keys that are wrapper-only or handled elsewhere
    for k in ("early_stopping_rounds", "verbosity"):
        user.pop(k, None)

    # Merge any remaining Booster-compatible keys
    params.update(user)

    early_stop = int(mcfg.get("early_stopping_rounds", 50))
    return params, num_boost_round, early_stop


# ---------- metrics ----------
def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "hit": np.nan}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if y_true.size >= 2 else np.nan
    hit = float(np.mean(np.sign(y_true) == np.sign(y_pred))) if y_true.size > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2, "hit": hit}


# ---------- booster utils (version-proof) ----------
def _predict_booster(booster: xgb.Booster, dmat: xgb.DMatrix, num_boost_round: int) -> np.ndarray:
    """Predict safely across xgboost versions."""
    # Newer APIs: iteration_range uses best_iteration
    try:
        best_iter = getattr(booster, "best_iteration", None)
        if isinstance(best_iter, int) and best_iter >= 0:
            return booster.predict(dmat, iteration_range=(0, best_iter + 1))
    except Exception:
        pass
    # Older APIs: ntree_limit via best_ntree_limit
    try:
        ntree_limit = getattr(booster, "best_ntree_limit", None)
        if not ntree_limit:
            ntree_limit = num_boost_round
        return booster.predict(dmat, ntree_limit=ntree_limit)
    except Exception:
        pass
    # Fallback
    return booster.predict(dmat)


def _gain_importance(booster: xgb.Booster) -> Dict[str, float]:
    """Gain-based feature importance; keys are feature names passed to DMatrix."""
    try:
        return booster.get_score(importance_type="gain")
    except Exception:
        try:
            return booster.get_fscore()
        except Exception:
            return {}


# ---------- main API ----------
def fit_predict_xgb(features: pd.DataFrame, target: pd.Series, config: dict) -> "ForecastResult":
    if not isinstance(features.index, pd.MultiIndex) or not isinstance(target.index, pd.MultiIndex):
        raise ValueError("features and target must both be MultiIndex ['date', 'ticker'].")

    set_global_seed(int(config.get("seed", 42)))

    # CV config
    cv_cfg = config.get("cv", {}) or {}
    cv = ExpandingCVConfig(
        n_folds=int(cv_cfg.get("n_folds", 5)),
        embargo_days=int(cv_cfg.get("embargo_days", 5)),
    )

    # >>> Core param normalization (prevents 'n_estimators not used')
    params, num_boost_round, early_stop = _normalize_booster_params(config)

    tickers = sorted(
        set(features.index.get_level_values("ticker"))
        & set(target.index.get_level_values("ticker"))
    )
    all_preds: List[pd.Series] = []
    diag: Dict[str, dict] = {"per_ticker": {}, "params": params, "cv": cv.__dict__}

    for tkr in tickers:
        X = features.xs(tkr, level="ticker").sort_index()
        y = target.xs(tkr, level="ticker").sort_index()
        df = X.join(y.to_frame("_y"), how="inner").dropna(how="any")
        if df.empty or df.shape[0] < max(40, cv.n_folds * 10):
            continue

        splits = expanding_cv_splits(df.index, cfg=cv)
        feature_names = [str(c) for c in X.columns]

        # >>> collect fold preds in a list (prevents pandas FutureWarning)
        oos_parts: list[pd.Series] = []
        fold_metrics: list[dict] = []
        feat_importance: dict[str, float] = {}

        for train_dates, val_dates in splits:
            X_train, y_train = df.loc[train_dates].drop(columns=["_y"]), df.loc[train_dates, "_y"]
            X_val, y_val = df.loc[val_dates].drop(columns=["_y"]), df.loc[val_dates, "_y"]
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
                verbose_eval=False,  # not silencing warnings; just no progress bar
            )

            # version-proof prediction
            y_pred = _predict_booster(booster, dval, num_boost_round)
            oos_parts.append(pd.Series(y_pred, index=val_dates))

            fold_metrics.append(_compute_metrics(y_val.values, y_pred))
            for k, v in _gain_importance(booster).items():
                feat_importance[k] = feat_importance.get(k, 0.0) + float(v)

        if not oos_parts:
            continue
        y_oos_pred = pd.concat(oos_parts).sort_index()

        y_true_all = df.loc[y_oos_pred.index, "_y"].values
        m = _compute_metrics(y_true_all, y_oos_pred.values)
        diag["per_ticker"][tkr] = {
            "folds": fold_metrics,
            "oos_metrics": m,
            "n_obs": int(df.shape[0]),
            "feature_importance": sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[
                :30
            ],
        }

        y_oos_pred.name = tkr
        all_preds.append(y_oos_pred)

    preds_df = pd.concat(all_preds, axis=1).sort_index() if all_preds else pd.DataFrame()
    return ForecastResult(preds=preds_df, diagnostics=diag)
