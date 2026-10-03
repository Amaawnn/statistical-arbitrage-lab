"""Rolling-origin out-of-sample evaluation with train-only pair selection."""

from dataclasses import dataclass

import pandas as pd

from .backtest import PairConfig, backtest_pair
from .cointegration import engle_granger_test


@dataclass
class WalkForwardResult:
    out_of_sample_returns: pd.Series
    fold_diagnostics: pd.DataFrame
    fold_results: list[pd.DataFrame]


def walk_forward(y: pd.Series, x: pd.Series, *, train_size: int, test_size: int,
                 config: PairConfig | None = None, pvalue_threshold: float = 0.05,
                 skip_failed_folds: bool = True) -> WalkForwardResult:
    """Fit/test sequential folds; cointegration screening uses training data only.

    Each test segment gets a warm-up prefix from its training data to initialize
    trailing estimates, but only test-segment returns are included in output.
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
        accepted = diag.pvalue < pvalue_threshold
        record = {"fold": fold, "train_start": train.index[0], "train_end": train.index[-1],
                  "test_start": test.index[0], "test_end": test.index[-1], "coint_pvalue": diag.pvalue,
                  "coint_statistic": diag.statistic, "beta_train": diag.beta, "accepted": accepted}
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
