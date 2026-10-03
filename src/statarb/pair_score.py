"""Transparent composite quality score for a researched trading pair."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PairQualityResult:
    """A 0–100 pair score, its label, and the points behind the score."""

    score: float
    classification: str
    component_scores: dict[str, float]


def _validate_probability(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite p-value between 0 and 1.")
    return number


def _tier_points(value: float, thresholds: tuple[tuple[float, float], ...]) -> float:
    for cutoff, points in thresholds:
        if value <= cutoff:
            return points
    return 0.0


def calculate_pair_quality_score(
    cointegration_pvalue: float,
    adf_pvalue: float,
    half_life_bars: float,
    sharpe_ratio: float,
    maximum_drawdown: float,
) -> PairQualityResult:
    """Score a pair using fixed, interpretable tiers (no fitted model).

    Inputs use p-values in [0, 1], half-life in bars, annualized Sharpe, and
    maximum drawdown as a decimal return (for example, -0.12 for -12%).
    Missing/non-finite non-p-value diagnostics are treated as zero points; an
    infinite half-life is valid and receives zero mean-reversion points.
    """
    coint_p = _validate_probability("cointegration_pvalue", cointegration_pvalue)
    adf_p = _validate_probability("adf_pvalue", adf_pvalue)
    half_life = float(half_life_bars)
    sharpe = float(sharpe_ratio)
    drawdown = float(maximum_drawdown)
    if math.isnan(half_life) or half_life < 0:
        half_life = math.inf
    if not math.isfinite(sharpe):
        sharpe = -math.inf
    drawdown_available = math.isfinite(drawdown)
    if drawdown_available and (drawdown > 0 or drawdown < -1):
        raise ValueError("maximum_drawdown must be between -1 and 0, inclusive.")

    # Maximum points: cointegration 25, ADF 20, half-life 20,
    # Sharpe 20, and maximum drawdown 15 (total 100).
    components = {
        "Engle-Granger cointegration": _tier_points(coint_p, ((0.01, 25), (0.05, 20), (0.10, 12.5))),
        "Residual ADF": _tier_points(adf_p, ((0.01, 20), (0.05, 16), (0.10, 10))),
        "Mean-reversion half-life": _tier_points(half_life, ((5, 20), (20, 15), (60, 10), (120, 5))),
        "Walk-forward Sharpe": _tier_points(-sharpe, ((-2, 20), (-1, 15), (0, 10), (1, 5))),
        "Maximum drawdown": _tier_points(-drawdown, ((0.05, 15), (0.10, 12), (0.20, 8), (0.30, 4))) if drawdown_available else 0.0,
    }
    score = round(float(sum(components.values())), 1)
    if score >= 75:
        classification = "Strong Candidate"
    elif score >= 50:
        classification = "Moderate Candidate"
    else:
        classification = "Weak Candidate"
    return PairQualityResult(score, classification, components)
