"""Performance statistics for periodic strategy returns."""

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series, *, initial_equity: float = 1.0) -> pd.Series:
    """Compound periodic simple returns into an equity curve."""
    r = pd.Series(returns).fillna(0.0).astype(float)
    if (r <= -1).any():
        raise ValueError("Returns must be greater than -100%.")
    return initial_equity * (1.0 + r).cumprod()


def performance_report(returns: pd.Series, *, periods_per_year: int = 252,
                       risk_free_rate: float = 0.0,
                       turnover: pd.Series | None = None) -> dict[str, float]:
    """Return annualized Sharpe, CAGR, maximum drawdown and related metrics."""
    r = pd.Series(returns).dropna().astype(float)
    if r.empty:
        return {k: float("nan") for k in ["total_return", "cagr", "annualized_volatility", "sharpe", "max_drawdown", "turnover"]} | {"observations": 0}
    eq = equity_curve(r)
    years = len(r) / periods_per_year
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else float("nan")
    excess = r - risk_free_rate / periods_per_year
    vol = float(r.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 else float("nan")
    sharpe = float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year)) if len(r) > 1 and excess.std(ddof=1) > 0 else float("nan")
    drawdown = eq / eq.cummax() - 1.0
    turnover_total = float(pd.Series(turnover).fillna(0.0).abs().sum()) if turnover is not None else float("nan")
    return {"total_return": float(eq.iloc[-1] - 1), "cagr": cagr, "annualized_volatility": vol,
            "sharpe": sharpe, "max_drawdown": float(drawdown.min()), "turnover": turnover_total,
            "observations": int(len(r))}
