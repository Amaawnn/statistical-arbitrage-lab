"""Rolling-origin out-of-sample evaluation with train-only pair selection."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import PairConfig, backtest_pair
from .cointegration import engle_granger_test, half_life


@dataclass
class WalkForwardResult:
    out_of_sample_returns: pd.Series
    fold_diagnostics: pd.DataFrame
    fold_results: list[pd.DataFrame]


def walk_forward(y: pd.Series, x: pd.Series, *, train_size: int, test_size: int,
                 config: PairConfig | None = None, pvalue_threshold: float = 0.05,
                 adf_threshold: float | None = None,
                 max_half_life: float | None = None,
                 skip_failed_folds: bool = True) -> WalkForwardResult:
    """Fit/test sequential folds; eligibility diagnostics use training data only.

    Each test segment gets a warm-up prefix from its training data to initialize
    trailing estimates, but only test-segment returns are included in output.
    Optional residual ADF significance and maximum half-life gates are evaluated
    on each fold's training residual before including its test returns.
    """
    cfg = config or PairConfig()
    pair = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().astype(float)
    if len(pair) < train_size + 1 or train_size < max(20, cfg.formation_window):
        raise ValueError("Insufficient data or train_size is shorter than the formation window.")
    returns, diagnostics, results = [], [], []
    fold = 0
    start = 0
    while start + train_size < len(pair):
        train_end = start + train_size
        test_end = min(train_end + test_size, len(pair))
        train, test = pair.iloc[start:train_end], pair.iloc[train_end:test_end]
        if test.empty:
            break
        diag = engle_granger_test(train.y, train.x)
        training_spread = np.log(train.y) - diag.intercept - diag.beta * np.log(train.x)
        hl = half_life(training_spread)
        adf_ok = adf_threshold is None or diag.residual_adf_pvalue < adf_threshold
        half_life_ok = max_half_life is None or hl <= max_half_life
        accepted = diag.pvalue < pvalue_threshold and adf_ok and half_life_ok
        record = {"fold": fold, "train_start": train.index[0], "train_end": train.index[-1],
                  "test_start": test.index[0], "test_end": test.index[-1], "coint_pvalue": diag.pvalue,
                  "coint_statistic": diag.statistic, "adf_statistic": diag.residual_adf_statistic,
                  "adf_pvalue": diag.residual_adf_pvalue, "half_life": hl,
                  "beta_train": diag.beta, "accepted": accepted}
        if accepted:
            # Seed rolling calculations with past data; retain only the OOS interval.
            warm_start = max(0, train_end - max(cfg.formation_window, cfg.z_window, cfg.regime_vol_window) - cfg.regime_vol_lookback)
            window = pair.iloc[warm_start:test_end]
            result = backtest_pair(window.y, window.x, config=cfg)
            oos = result.loc[test.index, "strategy_returns"]
            returns.append(oos)
            results.append(result.loc[test.index].copy())
            record["oos_observations"] = len(oos)
        else:
            record["oos_observations"] = 0
            if not skip_failed_folds:
                returns.append(pd.Series(0.0, index=test.index))
        diagnostics.append(record)
        fold += 1
        start += test_size
    combined = pd.concat(returns).sort_index() if returns else pd.Series(dtype=float, name="strategy_returns")
    combined.name = "strategy_returns"
    return WalkForwardResult(combined, pd.DataFrame(diagnostics), results)
