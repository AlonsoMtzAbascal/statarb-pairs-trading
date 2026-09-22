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
    r = np.asarray(returns, dtype=float)
    if r.std(ddof=1) == 0:
        return 0.0
    return np.sqrt(periods_per_year) * r.mean() / r.std(ddof=1)


def annualise_sharpe(per_period_sr, periods_per_year=252) -> float:
    return per_period_sr * np.sqrt(periods_per_year)


def sortino_ratio(returns, periods_per_year=252) -> float:
    r = np.asarray(returns, dtype=float)
    downside = r[r < 0]
    dd = downside.std(ddof=1) if downside.size > 1 else np.nan
    if not dd or np.isnan(dd):
        return np.nan
    return np.sqrt(periods_per_year) * r.mean() / dd


def max_drawdown(equity) -> float:
    eq = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    return dd.min()


def probabilistic_sharpe_ratio(observed_sr, benchmark_sr, n_obs,
                               skew=0.0, kurt=3.0) -> float:
    sr = observed_sr
    denom = np.sqrt(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2)
    z = (sr - benchmark_sr) * np.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_variance, n_trials) -> float:
    if n_trials <= 1:
        return 0.0
    v = np.sqrt(sr_variance)
    g = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(v * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(observed_sr, sr_variance, n_trials, n_obs,
                          skew=0.0, kurt=3.0) -> dict:
    sr0 = expected_max_sharpe(sr_variance, n_trials)
    dsr = probabilistic_sharpe_ratio(observed_sr, sr0, n_obs, skew, kurt)
    return {"deflated_sharpe": dsr, "benchmark_sr_per_period": sr0,
            "observed_sr_per_period": observed_sr, "n_trials": n_trials}


def summarise_returns(returns, equity=None, periods_per_year=252) -> dict:
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
