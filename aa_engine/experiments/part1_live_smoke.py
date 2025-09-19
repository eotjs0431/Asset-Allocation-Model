from aa_engine.data.loaders import load_price_panel
from aa_engine.features.price import (
    FeatureSpec,
    build_features,
    make_forward_return_target,
    align_features_and_target,
)
from aa_engine.models.xgb_forecaster import fit_predict_xgb

if __name__ == "__main__":
    tickers = [
        "SPY",
        "QQQ",
        "AGG",  # US
        "148070.KS",
        "385560.KS",
        "278530.KS",  # KRX
        "CMOD.L",  # London
    ]
    panel = load_price_panel(tickers, update=True, start="2015-01-01")

    spec = FeatureSpec()
    X = build_features(panel, spec)
    y = make_forward_return_target(panel, horizon_days=20)
    X_aligned, y_aligned = align_features_and_target(X, y, min_history_days=120)

    cfg = {
        "seed": 42,
        "cv": {"n_folds": 3, "embargo_days": 5},
        "model": {"params": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 4}},
    }
    result = fit_predict_xgb(X_aligned, y_aligned, cfg)

    print(result.preds.tail())
    for t, info in result.diagnostics["per_ticker"].items():
        print(t, info["oos_metrics"])
