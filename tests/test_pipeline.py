import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (backtest, cointegration, data, kalman, metrics, ou, signals)


@pytest.fixture(scope="module")
def universe():
    return data.generate_synthetic_universe(n_days=2520, seed=7)


def test_synthetic_universe_shape(universe):
    prices, truth = universe
    assert prices.shape == (2520, 10)
    assert len(truth.cointegrated_pairs) == 3
    assert not prices.isna().any().any()
    assert (prices > 0).all().all()          # prices must be positive


def test_cointegration_ranks_true_pairs_first(universe):
    prices, truth = universe
    screen = cointegration.screen_pairs(prices)
    true_pairs = {frozenset(p) for p in truth.cointegrated_pairs}
    # the single most cointegrated pair by p-value must be a true one
    top = screen.iloc[0]
    assert frozenset((top["asset_a"], top["asset_b"])) in true_pairs
    # true constant-hedge pairs should have very small p-values
    assert top["eg_pvalue"] < 1e-6


def test_kalman_recovers_time_varying_hedge(universe):
    prices, truth = universe
    pair = next(p for p in truth.cointegrated_pairs if truth.time_varying[p])
    a, b = pair
    kf = kalman.kalman_hedge(prices[a], prices[b])
    true_beta = pd.Series(truth.true_beta[pair], index=prices.index)
    mask = kf.hedge_ratio.notna() & (np.arange(len(prices)) > 300)
    corr = np.corrcoef(kf.hedge_ratio[mask], true_beta[mask])[0, 1]
    assert corr > 0.9                        # tracks the drifting hedge


def test_ou_half_life_recovered_on_constant_pairs(universe):
    prices, truth = universe
    for pair in truth.cointegrated_pairs:
        if truth.time_varying[pair]:
            continue
        a, b = pair
        eg = cointegration.engle_granger(prices[a], prices[b])
        spread = prices[a].values - (eg["hedge_ratio"] * prices[b].values
                                     + eg["intercept"])
        fit = ou.fit_ou(spread)
        true_hl = truth.true_half_life[pair]
        # recovered half-life within 40% of the truth
        assert abs(fit.half_life - true_hl) / true_hl < 0.4


def test_backtest_has_no_lookahead(universe):
    prices, _ = universe
    a, b = "COINTA1", "COINTA2"
    kf = kalman.kalman_hedge(prices[a], prices[b])
    z = signals.rolling_zscore(kf.spread, window=40)
    pos = signals.zscore_positions(z)

    cut = 1500
    bt_full = backtest.backtest_pair(prices[a], prices[b], pos, kf.hedge_ratio)

    tampered = prices.copy()
    tampered.iloc[cut:, tampered.columns.get_loc(a)] *= 1.5   # future shock
    # hedge/pos held fixed (they are inputs); only later prices change
    bt_tamp = backtest.backtest_pair(tampered[a], tampered[b], pos, kf.hedge_ratio)

    # returns strictly before the cut must be identical
    np.testing.assert_allclose(
        bt_full.returns.iloc[:cut - 1].values,
        bt_tamp.returns.iloc[:cut - 1].values, atol=1e-12)


def test_deflated_sharpe_in_unit_interval():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 1500)
    sr_pp = metrics.sharpe_ratio(r) / np.sqrt(252)
    dsr = metrics.deflated_sharpe_ratio(sr_pp, sr_variance=0.02**2,
                                        n_trials=50, n_obs=len(r))
    assert 0.0 <= dsr["deflated_sharpe"] <= 1.0
    # more trials -> higher benchmark -> lower or equal deflated Sharpe
    dsr_more = metrics.deflated_sharpe_ratio(sr_pp, sr_variance=0.02**2,
                                             n_trials=500, n_obs=len(r))
    assert dsr_more["deflated_sharpe"] <= dsr["deflated_sharpe"] + 1e-9


def test_probabilistic_sharpe_monotonic_in_track_length():
    # longer track record -> more confident the Sharpe beats zero
    short = metrics.probabilistic_sharpe_ratio(0.05, 0.0, 250)
    long = metrics.probabilistic_sharpe_ratio(0.05, 0.0, 2500)
    assert long > short


def test_max_drawdown_sign_and_bounds():
    equity = pd.Series([1.0, 1.2, 0.9, 1.1, 0.8, 1.3])
    mdd = metrics.max_drawdown(equity)
    assert -1.0 <= mdd <= 0.0
    assert mdd == pytest.approx((0.8 - 1.2) / 1.2)
