from aa_engine.data.loaders import load_price_panel
from aa_engine.features.price import (
    FeatureSpec,
    build_features,
    make_forward_return_target,
    align_features_and_target,
)
from aa_engine.models.xgb_forecaster import fit_predict_xgb

# --- choose a small set first for speed, then expand ---
TICKERS = [
    "SPY",
    "QQQ",
    "AGG",  # quick smoke
    "148070.KS",
    "385560.KS",
    "278530.KS",
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
]

# 1) Load prices (AdjClose as TRI proxy)
panel = load_price_panel(tickers=TICKERS, update=True, start="2015-01-01")

# 2) Build features & target
spec = FeatureSpec()
X = build_features(panel, spec)
y = make_forward_return_target(panel, horizon_days=20)
X_aligned, y_aligned = align_features_and_target(X, y, min_history_days=120)

# 3) Fit per-ticker XGB with expanding CV
cfg = {
    "seed": 42,
    "cv": {"n_folds": 3, "embargo_days": 5},
    "model": {"params": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4}},
}
result = fit_predict_xgb(X_aligned, y_aligned, cfg)

print(result.preds.tail())
for tkr, info in result.diagnostics["per_ticker"].items():
    print(tkr, info["oos_metrics"])
