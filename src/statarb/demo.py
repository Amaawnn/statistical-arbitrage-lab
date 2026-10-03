"""Deterministic end-to-end demonstration on synthetic cointegrated prices."""

import numpy as np
import pandas as pd

from .backtest import PairConfig, backtest_pair
from .cointegration import engle_granger_test, half_life
from .metrics import performance_report
from .walkforward import walk_forward


def synthetic_prices(n: int = 1800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    common = np.cumsum(rng.normal(0, 0.008, n))
    residual = np.zeros(n)
    for i in range(1, n):
        residual[i] = 0.92 * residual[i - 1] + rng.normal(0, 0.012)
    log_x = np.log(100) + common
    log_y = np.log(50) + 0.15 + 1.25 * common + residual
    return pd.DataFrame({"ASSET_A": np.exp(log_y), "ASSET_B": np.exp(log_x)}, index=dates)


def main() -> None:
    prices = synthetic_prices()
    y, x = prices.ASSET_A, prices.ASSET_B
    config = PairConfig(formation_window=252, z_window=60, entry_z=2.0, exit_z=0.5,
                        cost_bps=5.0, regime_vol_quantile=None)
    diagnostics = engle_granger_test(y.iloc[:900], x.iloc[:900])
    print("Training-window Engle-Granger diagnostics:")
    print(diagnostics)
    print(f"Residual half-life: {half_life(np.log(y.iloc[:900]) - diagnostics.intercept - diagnostics.beta * np.log(x.iloc[:900])):.1f} bars")
    result = backtest_pair(y, x, config=config)
    print("\nFull-sample illustrative backtest (in-sample; not a performance claim):")
    print(performance_report(result.strategy_returns))
    wf = walk_forward(y, x, train_size=756, test_size=126, config=config)
    print("\nWalk-forward out-of-sample metrics:")
    print(performance_report(wf.out_of_sample_returns))
    print("\nFold diagnostics:")
    print(wf.fold_diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
