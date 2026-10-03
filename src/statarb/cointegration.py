"""Cointegration tests, hedge-ratio estimation, and mean-reversion diagnostics."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


@dataclass(frozen=True)
class CointegrationResult:
    statistic: float
    pvalue: float
    critical_values: tuple[float, float, float]
    beta: float
    intercept: float
    residual_adf_statistic: float
    residual_adf_pvalue: float

    def to_dict(self) -> dict:
        return asdict(self)


def _aligned_log_prices(y: pd.Series, x: pd.Series) -> tuple[pd.Series, pd.Series]:
    pair = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(pair) < 20:
        raise ValueError("At least 20 aligned observations are required.")
    if (pair <= 0).any().any():
        raise ValueError("Prices must be positive to compute log prices.")
    return np.log(pair["y"].astype(float)), np.log(pair["x"].astype(float))


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """OLS fit log(y) = intercept + beta * log(x) + residual."""
    log_y, log_x = _aligned_log_prices(y, x)
    fit = sm.OLS(log_y, sm.add_constant(log_x)).fit()
    return float(fit.params.iloc[1]), float(fit.params.iloc[0])


def adf_diagnostic(series: pd.Series, *, maxlag: int | None = None, regression: str = "c") -> dict:
    """ADF unit-root diagnostic for a supplied spread or residual series."""
    clean = pd.Series(series).dropna().astype(float)
    if len(clean) < 20:
        raise ValueError("At least 20 observations are required for an ADF test.")
    stat, pvalue, used_lag, nobs, critical, icbest = adfuller(clean, maxlag=maxlag, regression=regression, autolag="AIC")
    return {"statistic": float(stat), "pvalue": float(pvalue), "used_lag": int(used_lag), "nobs": int(nobs),
            "critical_values": {k: float(v) for k, v in critical.items()}, "icbest": float(icbest)}


def engle_granger_test(y: pd.Series, x: pd.Series, *, trend: str = "c", maxlag: int | None = None) -> CointegrationResult:
    """Engle–Granger two-step test with OLS log-price hedge ratio and residual ADF."""
    log_y, log_x = _aligned_log_prices(y, x)
    fit = sm.OLS(log_y, sm.add_constant(log_x)).fit()
    beta, intercept = float(fit.params.iloc[1]), float(fit.params.iloc[0])
    stat, pvalue, critical = coint(log_y, log_x, trend=trend, maxlag=maxlag, autolag="aic")
    resid_stat, resid_pvalue, *_ = adfuller(fit.resid, regression="n", autolag="AIC")
    return CointegrationResult(float(stat), float(pvalue), tuple(float(v) for v in critical), beta, intercept,
                               float(resid_stat), float(resid_pvalue))


def half_life(spread: pd.Series) -> float:
    """Estimate AR(1) mean-reversion half-life; infinity when no reversion is estimated."""
    s = pd.Series(spread).dropna().astype(float)
    delta = s.diff().dropna()
    lagged = s.shift(1).dropna()
    delta, lagged = delta.align(lagged, join="inner")
    if len(delta) < 3:
        return float("nan")
    slope = float(sm.OLS(delta, sm.add_constant(lagged)).fit().params.iloc[1])
    return float(-np.log(2) / slope) if slope < 0 else float("inf")
