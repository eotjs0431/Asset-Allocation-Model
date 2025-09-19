"""AA Engine package.

Layers:
- data: loaders, calendars (PIT-safe)
- features: pure feature functions
- models: forecasters + tuning + diagnostics
- portfolio: Black–Litterman, optimizers
- backtest: execution, costs, metrics
- utils: io, time, logging, seeds
"""

__all__ = ["config", "data", "features", "models", "portfolio", "backtest", "utils"]
