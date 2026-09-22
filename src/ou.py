"""
Ornstein-Uhlenbeck modeling of the spread.

The spread of a cointegrated pair mean-reverts, and the OU process is the
canonical continuous-time model of mean reversion:

    ds_t = kappa (mu - s_t) dt + sigma dW_t

Its most useful summary is the HALF-LIFE of mean reversion, ln(2)/kappa -- the
expected time for a deviation to decay halfway back to the mean. The half-life
tells you the natural holding period of the trade and lets you size entries and
exits off the process's own timescale, instead of picking an arbitrary "enter at
2 sigma, exit at 0".

Estimation is by the exact discrete equivalent of OU, which is an AR(1):

    s_{t+1} = a + b s_t + eps,   with  b = exp(-kappa dt).

Regressing s_{t+1} on s_t gives b, and hence kappa = -ln(b)/dt and
half_life = ln(2)/kappa = -ln(2)/ln(b).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["OUFit", "fit_ou"]


@dataclass
class OUFit:
    kappa: float        # mean-reversion speed (per period)
    mu: float           # long-run mean of the spread
    sigma: float        # instantaneous volatility
    half_life: float    # ln(2) / kappa, in periods (days)


def fit_ou(spread, dt: float = 1.0) -> OUFit:
    """Fit an OU process to a spread series by AR(1) regression.

    Returns an OUFit; half_life is NaN if the series is not mean-reverting
    (estimated b >= 1), which is itself a useful diagnostic.
    """
    s = np.asarray(spread, dtype=float)
    s = s[~np.isnan(s)]
    s_t, s_next = s[:-1], s[1:]

    # OLS of s_{t+1} on s_t with intercept.
    X = np.vstack([s_t, np.ones_like(s_t)]).T
    b, a = np.linalg.lstsq(X, s_next, rcond=None)[0]
    resid = s_next - (b * s_t + a)

    if b <= 0 or b >= 1:
        half_life = np.nan
        kappa = np.nan
    else:
        kappa = -np.log(b) / dt
        half_life = np.log(2) / kappa

    mu = a / (1 - b) if b < 1 else np.nan
    sigma = resid.std(ddof=2) / np.sqrt(dt)
    return OUFit(kappa=kappa, mu=mu, sigma=sigma, half_life=half_life)
