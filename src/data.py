"""
Data sources for the pairs-trading study.

Two entry points:

* `generate_synthetic_universe`  -- builds a panel of price series with KNOWN
  ground truth: some pairs are genuinely cointegrated (with a spread that is a
  stationary Ornstein-Uhlenbeck process of known half-life, and in one case a
  deliberately time-varying hedge ratio), and the rest are independent random
  walks that are cointegrated with nothing. Because the truth is known, the
  cointegration test, the Kalman hedge-ratio filter, and the OU half-life
  estimator can all be *validated* -- do they recover what was put in? This is
  the same logic as validating a Monte Carlo pricer against a closed form:
  a method you cannot check against a known answer is a method you cannot trust.

* `load_prices_yfinance`  -- downloads real adjusted close prices so the
  identical pipeline can be run on live equities/ETFs. Requires internet and
  the `yfinance` package; it is the real-data entry point for anyone running
  the repo.

The synthetic generator returns both the price panel and a `GroundTruth` record
describing exactly which pairs are cointegrated and with what parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["GroundTruth", "generate_synthetic_universe", "load_prices_yfinance"]


@dataclass
class GroundTruth:
    """What was actually put into the synthetic data, for validation."""
    cointegrated_pairs: list                      # list of (ticker_a, ticker_b)
    true_beta: dict = field(default_factory=dict)   # pair -> true hedge-ratio path
    true_half_life: dict = field(default_factory=dict)  # pair -> OU half-life (days)
    time_varying: dict = field(default_factory=dict)    # pair -> bool


def _ou_spread(n, half_life, sigma_stat, rng):
    """A stationary AR(1)/OU spread with a given half-life and stationary std.

    AR(1): s_t = phi s_{t-1} + eps, with phi = 0.5 ** (1 / half_life) so the
    autocorrelation decays by half every `half_life` steps. The innovation std
    is set so the stationary standard deviation equals `sigma_stat`.
    """
    phi = 0.5 ** (1.0 / half_life)
    eps_sigma = sigma_stat * np.sqrt(1.0 - phi**2)
    s = np.zeros(n)
    s[0] = rng.normal(0, sigma_stat)
    innovations = rng.normal(0, eps_sigma, size=n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + innovations[t]
    return s


def _random_walk_price(n, s0, mu, sigma, rng):
    """A positive price series: geometric random walk (exp of a drifting walk)."""
    daily = rng.normal(mu, sigma, size=n)
    log_price = np.log(s0) + np.cumsum(daily)
    return np.exp(log_price)


def generate_synthetic_universe(n_days=2520, seed=7):
    """Build a synthetic universe with known cointegration structure.

    Returns
    -------
    prices : pd.DataFrame  (n_days x n_assets), business-day indexed
    truth  : GroundTruth
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-01", periods=n_days)
    cols = {}
    truth = GroundTruth(cointegrated_pairs=[])

    def add_cointegrated(name, beta_path, half_life, sigma_spread, base_kw):
        """Create a cointegrated pair: y = beta_t * x + stationary_spread."""
        x = _random_walk_price(n_days, rng=rng, **base_kw)
        spread = _ou_spread(n_days, half_life, sigma_spread, rng)
        y = beta_path * x + spread
        a, b = f"{name}1", f"{name}2"
        # Ticker1 is the DEPENDENT leg (y = beta*x + stationary spread) and
        # ticker2 the independent leg, so regressing ticker1 on ticker2 -- as
        # the Kalman filter and Engle-Granger both do -- recovers beta directly.
        cols[a] = y
        cols[b] = x
        pair = (a, b)
        truth.cointegrated_pairs.append(pair)
        truth.true_beta[pair] = beta_path
        truth.true_half_life[pair] = half_life
        truth.time_varying[pair] = bool(np.ptp(beta_path) > 1e-6)

    # Pair A: constant hedge ratio, fast mean reversion (half-life ~ 10 days).
    add_cointegrated("COINTA", np.full(n_days, 1.15), half_life=10,
                     sigma_spread=2.5, base_kw=dict(s0=100, mu=3e-4, sigma=0.012))

    # Pair B: TIME-VARYING hedge ratio -- a substantial structural drift (the
    # relationship strengthens then the leverage between the two names falls, as
    # happens when relative fundamentals shift over years). This is the case
    # that breaks a static OLS hedge and showcases the Kalman filter. The drift
    # is slow relative to the spread's ~18-day mean reversion, so a small-delta
    # filter can track it without chasing the tradeable signal.
    beta_tv = np.linspace(1.30, 0.50, n_days)    # steady structural drift
    add_cointegrated("COINTB", beta_tv, half_life=18,
                     sigma_spread=3.0, base_kw=dict(s0=80, mu=2e-4, sigma=0.013))

    # Pair C: constant hedge ratio, slower mean reversion (half-life ~ 30 days).
    add_cointegrated("COINTC", np.full(n_days, 0.95), half_life=30,
                     sigma_spread=4.0, base_kw=dict(s0=120, mu=1e-4, sigma=0.011))

    # Decoys: independent random walks, cointegrated with nothing. They form the
    # null distribution for the multiple-testing / deflated-Sharpe analysis.
    for i in range(1, 5):
        cols[f"INDEP{i}"] = _random_walk_price(
            n_days, s0=rng.uniform(50, 150), mu=rng.normal(2e-4, 1e-4),
            sigma=rng.uniform(0.010, 0.016), rng=rng)

    prices = pd.DataFrame(cols, index=dates)
    return prices, truth


def load_prices_yfinance(tickers, start="2015-01-01", end="2024-12-31",
                         price_field="Close"):
    """Download real adjusted-close prices for a list of tickers.

    Requires internet access and `pip install yfinance`. This is the entry point
    for running the identical pipeline on real equities/ETFs -- e.g. sector ETF
    pairs (XLE/XOP), dual-listed names, or an index against its constituents.

    Example
    -------
    >>> prices = load_prices_yfinance(["EWA", "EWC"], "2010-01-01", "2024-12-31")
    """
    import yfinance as yf  # imported lazily so the synthetic path needs no net

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False)
    prices = raw[price_field] if price_field in raw else raw["Close"]
    return prices.dropna(how="any")
