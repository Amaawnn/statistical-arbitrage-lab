from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from app import parse_price_csv, performance_cards
from statarb import PairConfig, make_synthetic_pair, walk_forward
from statarb.backtest import _rolling_ols


def test_make_synthetic_pair_is_seeded_and_has_requested_length():
    first = make_synthetic_pair(n=500, seed=17)
    second = make_synthetic_pair(n=500, seed=17)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 500
    assert (first > 0).all().all()


def test_parse_price_csv_validates_and_sorts_data():
    payload = b"Date,Asset_Y,Asset_X\n2024-01-02,11,21\n2024-01-01,10,20\n"
    prices = parse_price_csv(BytesIO(payload))
    assert prices.index.is_monotonic_increasing
    assert list(prices.columns) == ["Asset_Y", "Asset_X"]
    assert prices.iloc[0].to_list() == [10, 20]


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"Date,Asset_Y\n2024-01-01,10\n", "Missing: Asset_X"),
        (b"Date,Asset_Y,Asset_X\n2024-01-01,0,10\n", "must all be positive"),
        (b"Date,Asset_Y,Asset_X\nnot-a-date,10,20\n", "valid dates"),
        (b"Date,Asset_Y,Asset_X\n2024-01-01,nope,20\n", "numeric"),
    ],
)
def test_parse_price_csv_reports_invalid_input(payload, message):
    with pytest.raises(ValueError, match=message):
        parse_price_csv(BytesIO(payload))


def test_performance_cards_uses_supplied_walk_forward_returns_and_costs():
    returns = pd.Series([0.01, -0.005, 0.002])
    folds = [pd.DataFrame({"transaction_cost": [0.001, 0.002]})]
    cards = performance_cards(returns, folds)
    assert cards["trading_days"] == 3
    assert cards["win_rate"] == pytest.approx(2 / 3)
    assert cards["transaction_costs"] == pytest.approx(0.003)
    assert cards["total_return"] == pytest.approx(np.prod(1 + returns) - 1)


def test_rolling_ols_is_causal_and_recovers_a_linear_log_hedge_ratio():
    index = pd.RangeIndex(40)
    log_x = pd.Series(np.linspace(0.0, 1.0, len(index)), index=index)
    x = np.exp(log_x)
    y = np.exp(0.25 + 1.4 * log_x)
    beta, alpha = _rolling_ols(y, x, window=20)
    assert beta.iloc[:20].isna().all()
    assert beta.iloc[20] == pytest.approx(1.4)
    assert alpha.iloc[20] == pytest.approx(0.25)

    changed_y = y.copy()
    changed_y.iloc[25:] *= 4
    changed_beta, _ = _rolling_ols(changed_y, x, window=20)
    pd.testing.assert_series_equal(beta.iloc[:25], changed_beta.iloc[:25])


def test_walk_forward_rejects_fold_when_training_adf_gate_fails():
    prices = make_synthetic_pair(n=400, seed=31)
    result = walk_forward(
        prices.ASSET_A, prices.ASSET_B, train_size=300, test_size=100,
        config=PairConfig(formation_window=200, z_window=20),
        pvalue_threshold=1.0, adf_threshold=0.0,
    )
    assert not result.fold_diagnostics.iloc[0]["accepted"]
    assert result.fold_diagnostics.iloc[0]["adf_pvalue"] >= 0.0
    assert result.out_of_sample_returns.empty
