from __future__ import annotations
import argparse
import json
import datetime as dt
from pathlib import Path
import yaml
import pandas as pd

from aa_engine.data.loaders import load_price_panel
from aa_engine.features.price import (
    FeatureSpec,
    build_features,
    make_forward_return_target,
    align_features_and_target,
)
from aa_engine.models.tuning import ExpandingCVConfig
from aa_engine.models.optimize import tune_xgb_params
from aa_engine.models.xgb_forecaster import fit_predict_xgb


DEFAULT_TICKERS = [
    "SPY",
    "QQQ",
    "AGG",
    "BND",
    "HYG",
    "EMB",
    "VWOB",
    "EWY",
    "IWM",
    "VWO",
    "IEMG",
    "CMOD.L",
    "GSG",
    "GLD",
    "RWO",
    "IFGL",
    "148070.KS",
    "385560.KS",
    "278530.KS",
]


def _load_yaml_any_encoding(path: Path) -> dict:
    """Load YAML robustly; prefer UTF-8 on Windows."""
    if not path.exists():
        return {}
    for enc in ("utf-8", "utf-8-sig", "cp949", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                data = yaml.safe_load(f)
                return data or {}
        except UnicodeDecodeError:
            continue
    # Last resort: decode as UTF-8 with replacement
    with open(path, "rb") as fb:
        text = fb.read().decode("utf-8", errors="replace")
    return yaml.safe_load(text) or {}


def _now_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    ap = argparse.ArgumentParser(description="Part 1: Tune (optional) and run XGB forecaster")
    ap.add_argument("--config", default="configs/forecaster.yaml", help="YAML config path")
    ap.add_argument(
        "--no-tune", action="store_true", help="Skip tuning and use config model.params as-is"
    )
    ap.add_argument(
        "--tune-all", action="store_true", help="Tune on the full --tickers list (overrides YAML)"
    )
    ap.add_argument("--trials", type=int, default=None, help="Override tuning.n_trials")
    ap.add_argument("--start", default=None, help="Override data start date (YYYY-MM-DD)")
    ap.add_argument("--tickers", nargs="*", default=None, help="Override ticker list")
    args = ap.parse_args()

    # --- Load config (best-effort) ---
    cfg_path = Path(args.config)
    cfg = _load_yaml_any_encoding(cfg_path)

    data_start = args.start or cfg.get("data", {}).get("start", "2015-01-01")
    tickers = args.tickers or cfg.get("tickers", DEFAULT_TICKERS)

    # --- Load prices ---
    panel = load_price_panel(tickers, start=data_start, update=True)

    # --- Features & target ---
    spec = FeatureSpec()
    X = build_features(panel, spec)
    y = make_forward_return_target(panel, horizon_days=int(cfg.get("horizon_days", 20)))
    X, y = align_features_and_target(X, y, min_history_days=int(cfg.get("min_history_days", 120)))

    # --- CV config ---
    cv_cfg = ExpandingCVConfig(
        n_folds=int(cfg.get("cv", {}).get("n_folds", 3)),
        embargo_days=int(cfg.get("cv", {}).get("embargo_days", 5)),
    )

    # --- Artifacts root ---
    run_tag = _now_tag()
    outdir = _ensure_dir(Path(f"artifacts/forecaster/{run_tag}"))

    # --- Tuning (optional) ---
    model_params = dict(cfg.get("model", {}).get("params", {}))
    early_stop = int(
        cfg.get("model", {}).get(
            "early_stopping_rounds", cfg.get("tuning", {}).get("early_stopping_rounds", 50)
        )
    )
    if not args.no_tune and cfg.get("tuning", {}).get("enabled", True):
        tune_cfg = dict(cfg.get("tuning", {}))
        if args.trials is not None:
            tune_cfg["n_trials"] = args.trials
        # Aggregation defaults if not provided
        tune_cfg.setdefault("aggregation", {"unit": "nrmse", "weight": "equal", "reducer": "mean"})
        # Calibrate on full universe if requested
        if args.tune_all:
            tune_cfg["tickers"] = tickers
        else:
            tune_cfg.setdefault("tickers", ["SPY", "QQQ", "AGG"])

        tres = tune_xgb_params(X, y, tune_cfg, cv_cfg)

        # persist tuning artifacts
        tres.leaderboard.to_csv(outdir / "tuning_leaderboard.csv", index=False)
        with open(outdir / "tuning_best_params.json", "w") as f:
            json.dump(tres.best_params, f, indent=2)
        with open(outdir / "tuning_diagnostics.json", "w") as f:
            json.dump(tres.diagnostics, f, indent=2)

        # merge best into model params for the final fit
        model_params.update(tres.best_params)

    # --- Final training run with chosen params ---
    run_cfg = {
        "seed": int(cfg.get("seed", 42)),
        "cv": {"n_folds": cv_cfg.n_folds, "embargo_days": cv_cfg.embargo_days},
        "model": {"params": model_params, "early_stopping_rounds": early_stop},
    }
    result = fit_predict_xgb(X, y, run_cfg)

    # --- Save outputs ---
    result.preds.to_csv(outdir / "oos_predictions.csv")
    with open(outdir / "diagnostics.json", "w") as f:
        json.dump(result.diagnostics, f, indent=2)

    # quick summary to console
    print(f"\nSaved artifacts → {outdir}")
    print("\nPer-ticker OOS metrics:")
    cols = ["rmse", "mae", "r2", "hit"]
    summary = []
    for tkr, info in result.diagnostics.get("per_ticker", {}).items():
        m = info.get("oos_metrics", {})
        summary.append({"ticker": tkr, **{k: m.get(k) for k in cols}})
    if summary:
        df = pd.DataFrame(summary).set_index("ticker").sort_index()
        print(df.round(6))
        df.to_csv(outdir / "oos_metrics.csv")


if __name__ == "__main__":
    main()
