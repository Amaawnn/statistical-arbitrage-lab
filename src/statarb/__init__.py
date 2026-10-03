"""Research tools for regime-aware statistical arbitrage."""

from .backtest import PairConfig, backtest_pair
from .cointegration import CointegrationResult, adf_diagnostic, engle_granger_test, estimate_hedge_ratio, half_life
from .metrics import equity_curve, performance_report
from .walkforward import WalkForwardResult, walk_forward

__all__ = [
    "CointegrationResult", "PairConfig", "WalkForwardResult", "adf_diagnostic",
    "backtest_pair", "engle_granger_test", "equity_curve", "estimate_hedge_ratio",
    "half_life", "performance_report", "walk_forward",
]
