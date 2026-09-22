"""
Performance metrics -- including the multiple-testing correction that is the
intellectual core of this project.

The ordinary Sharpe ratio is misleading in exactly the situation a pairs study
creates: when you screen hundreds of candidate pairs and keep the best one, the
winner's Sharpe is inflated by selection. Searching many strategies for the best
Sharpe is the same statistical problem as searching many sky locations for a
signal -- the "look-elsewhere effect" -- and it must be corrected with a trials
penalty.

Two tools do that here:

* Probabilistic Sharpe Ratio (PSR) -- the probability that the true Sharpe
  exceeds a benchmark, accounting for track-record length and for the skew and
  fat tails of real strategy returns (a Sharpe on non-normal returns is itself
  biased).

* Deflated Sharpe Ratio (DSR) -- PSR evaluated against the benchmark you would
  expect the BEST of N random strategies to achieve by luck alone. It penalises
  exactly the multiple testing that pairs selection performs. (Bailey &
  Lopez de Prado, 2014.)

All Sharpe inputs below are PER-PERIOD (e.g. daily) unless annualised via
`annualise_sharpe`.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "sharpe_ratio", "annualise_sharpe", "sortino_ratio", "max_drawdown",
    "probabilistic_sharpe_ratio", "deflated_sharpe_ratio",
    "expected_max_sharpe", "summarise_returns",
]

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns, periods_per_year=252) -> float:
    """Annualised Sharpe ratio of a per-period return series."""
    r = np.asarray(returns, dtype=float)
    if r.std(ddof=1) == 0:
        return 0.0
    return np.sqrt(periods_per_year) * r.mean() / r.std(ddof=1)


def annualise_sharpe(per_period_sr, periods_per_year=252) -> float:
    return per_period_sr * np.sqrt(periods_per_year)


def sortino_ratio(returns, periods_per_year=252) -> float:
    """Like Sharpe but penalising only downside deviation."""
    r = np.asarray(returns, dtype=float)
    downside = r[r < 0]
    dd = downside.std(ddof=1) if downside.size > 1 else np.nan
    if not dd or np.isnan(dd):
        return np.nan
    return np.sqrt(periods_per_year) * r.mean() / dd


def max_drawdown(equity) -> float:
    """Worst peak-to-trough fractional drop of an equity curve."""
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return dd.min()


def probabilistic_sharpe_ratio(observed_sr, benchmark_sr, n_obs,
                               skew=0.0, kurt=3.0) -> float:
    """P(true Sharpe > benchmark), given track length and return non-normality.

    All Sharpes are per-period. `kurt` is non-excess kurtosis (normal = 3).
    """
    sr = observed_sr
    denom = np.sqrt(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2)
    z = (sr - benchmark_sr) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_variance, n_trials) -> float:
    """Expected maximum per-period Sharpe of N independent zero-skill trials.

    This is the benchmark the best of N random strategies clears by luck. Uses
    the extreme-value approximation from Bailey & Lopez de Prado (2014).
    """
    if n_trials <= 1:
        return 0.0
    v = np.sqrt(sr_variance)
    g = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(v * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(observed_sr, sr_variance, n_trials, n_obs,
                          skew=0.0, kurt=3.0) -> dict:
    """Deflated Sharpe Ratio: PSR against the expected-max-of-N benchmark.

    Parameters are per-period. Returns a dict with the benchmark it had to beat,
    and the DSR probability. DSR close to 1 => the track record is unlikely to
    be a multiple-testing artefact; close to 0 => it probably is.
    """
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    dsr = probabilistic_sharpe_ratio(observed_sr, sr0, n_obs, skew, kurt)
    return {"deflated_sharpe": dsr, "benchmark_sr_per_period": sr0,
            "observed_sr_per_period": observed_sr, "n_trials": n_trials}


def summarise_returns(returns, equity=None, periods_per_year=252) -> dict:
    """Standard performance summary for a return series."""
    r = np.asarray(returns, dtype=float)
    if equity is None:
        equity = np.cumprod(1.0 + r)
    ann_ret = (1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0
    return {
        "sharpe": sharpe_ratio(r, periods_per_year),
        "sortino": sortino_ratio(r, periods_per_year),
        "annual_return": ann_ret,
        "annual_vol": r.std(ddof=1) * np.sqrt(periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "n_periods": len(r),
    }
