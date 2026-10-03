"""Pair signal generation and transparent next-session execution simulation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cointegration import estimate_hedge_ratio


@dataclass(frozen=True)
class PairConfig:
    formation_window: int = 252
    z_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    cost_bps: float = 5.0
    regime_vol_window: int = 60
    regime_vol_lookback: int = 252
    regime_vol_quantile: float | None = 0.9

    def __post_init__(self):
        if self.formation_window < 20 or self.z_window < 5:
            raise ValueError("formation_window must be >=20 and z_window >=5.")
        if self.entry_z <= self.exit_z or self.exit_z < 0:
            raise ValueError("Require entry_z > exit_z >= 0.")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative.")
        if self.regime_vol_quantile is not None and not 0 < self.regime_vol_quantile < 1:
            raise ValueError("regime_vol_quantile must be between 0 and 1.")


def _rolling_ols(y: pd.Series, x: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    log_y, log_x = np.log(y), np.log(x)
    beta = pd.Series(np.nan, index=y.index, dtype=float)
    intercept = beta.copy()
    for end in range(window, len(y) + 1):
        start = end - window
        if end < len(y):
            b, a = estimate_hedge_ratio(y.iloc[start:end], x.iloc[start:end])
            beta.iloc[end] = b
            intercept.iloc[end] = a
    return beta, intercept


def backtest_pair(y: pd.Series, x: pd.Series, *, config: PairConfig | None = None,
                  hedge_ratio: float | None = None, intercept: float | None = None) -> pd.DataFrame:
    """Backtest one pair. Signals at t close execute at t+1 close/open proxy.

    With a supplied fixed hedge ratio, the residual center/intercept is estimated
    on the trailing formation window. Otherwise OLS is refit on a rolling window.
    Returns are fractional close-to-close price returns after the position has
    been shifted one row; transaction cost is charged per leg on notional change.
    """
    cfg = config or PairConfig()
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().astype(float)
    frame = frame[(frame > 0).all(axis=1)]
    if len(frame) <= max(cfg.formation_window, cfg.z_window) + 2:
        raise ValueError("Not enough aligned positive prices for configured windows.")
    if hedge_ratio is not None:
        beta = pd.Series(float(hedge_ratio), index=frame.index)
        if intercept is None:
            a = pd.Series(np.nan, index=frame.index)
            for i in range(cfg.formation_window, len(frame)):
                fit_y = np.log(frame.y.iloc[i-cfg.formation_window:i]) - beta.iloc[i] * np.log(frame.x.iloc[i-cfg.formation_window:i])
                a.iloc[i] = fit_y.mean()
        else:
            a = pd.Series(float(intercept), index=frame.index)
    else:
        beta, a = _rolling_ols(frame.y, frame.x, cfg.formation_window)

    spread = np.log(frame.y) - (a + beta * np.log(frame.x))
    mean = spread.rolling(cfg.z_window, min_periods=cfg.z_window).mean()
    std = spread.rolling(cfg.z_window, min_periods=cfg.z_window).std(ddof=1).replace(0, np.nan)
    z = (spread - mean) / std
    spread_vol = spread.diff().rolling(cfg.regime_vol_window, min_periods=cfg.regime_vol_window).std()
    if cfg.regime_vol_quantile is None:
        regime_ok = pd.Series(True, index=frame.index)
    else:
        threshold = spread_vol.shift(1).rolling(cfg.regime_vol_lookback, min_periods=max(20, cfg.regime_vol_window)).quantile(cfg.regime_vol_quantile)
        regime_ok = (spread_vol <= threshold).fillna(False)

    desired = np.zeros(len(frame), dtype=float)
    state = 0.0
    for i, value in enumerate(z.to_numpy()):
        if not np.isfinite(value) or not bool(regime_ok.iloc[i]):
            state = 0.0
        elif state == 0 and value >= cfg.entry_z:
            state = -1.0
        elif state == 0 and value <= -cfg.entry_z:
            state = 1.0
        elif state == -1 and value <= cfg.exit_z:
            state = 0.0
        elif state == 1 and value >= -cfg.exit_z:
            state = 0.0
        desired[i] = state

    result = frame.copy()
    result["beta"] = beta
    result["intercept"] = a
    result["spread"] = spread
    result["zscore"] = z
    result["regime_ok"] = regime_ok
    result["signal"] = desired
    result["position"] = result["signal"].shift(1).fillna(0.0)
    result["y_return"] = frame.y.pct_change().fillna(0.0)
    result["x_return"] = frame.x.pct_change().fillna(0.0)
    # Normalize each leg to gross exposure 1 + |beta|. Signed position sets direction.
    denom = 1.0 + result.beta.abs()
    result["y_weight"] = result.position / denom
    result["x_weight"] = -result.position * result.beta / denom
    result["gross_return"] = result.y_weight * result.y_return + result.x_weight * result.x_return
    y_turn = result.y_weight.diff().abs().fillna(result.y_weight.abs())
    x_turn = result.x_weight.diff().abs().fillna(result.x_weight.abs())
    rate = cfg.cost_bps / 10_000.0
    result["y_cost"] = y_turn * rate
    result["x_cost"] = x_turn * rate
    result["transaction_cost"] = result.y_cost + result.x_cost
    result["strategy_returns"] = result.gross_return - result.transaction_cost
    result["turnover"] = y_turn + x_turn
    return result
