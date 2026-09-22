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
