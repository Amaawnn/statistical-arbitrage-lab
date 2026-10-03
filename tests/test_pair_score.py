import pytest

from statarb import PairQualityResult, calculate_pair_quality_score


def test_pair_score_returns_explainable_100_point_strong_candidate():
    result = calculate_pair_quality_score(
        cointegration_pvalue=0.005,
        adf_pvalue=0.008,
        half_life_bars=4,
        sharpe_ratio=2.4,
        maximum_drawdown=-0.04,
    )

    assert isinstance(result, PairQualityResult)
    assert result.score == 100
    assert result.classification == "Strong Candidate"
    assert sum(result.component_scores.values()) == result.score


def test_pair_score_assigns_expected_moderate_tier_points():
    result = calculate_pair_quality_score(0.05, 0.08, 30, 0.5, -0.15)

    assert result.score == 58
    assert result.classification == "Moderate Candidate"
    assert result.component_scores["Engle-Granger cointegration"] == 20
    assert result.component_scores["Maximum drawdown"] == 8


def test_pair_score_marks_weak_metrics_as_weak_candidate():
    result = calculate_pair_quality_score(0.4, 0.5, float("inf"), -1.2, -0.45)

    assert result.score == 0
    assert result.classification == "Weak Candidate"


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf")])
def test_pair_score_rejects_invalid_pvalues(value):
    with pytest.raises(ValueError, match="p-value"):
        calculate_pair_quality_score(value, 0.05, 10, 1.0, -0.1)


@pytest.mark.parametrize("drawdown", [0.01, -1.01])
def test_pair_score_rejects_invalid_drawdown(drawdown):
    with pytest.raises(ValueError, match="maximum_drawdown"):
        calculate_pair_quality_score(0.01, 0.01, 10, 1.0, drawdown)
