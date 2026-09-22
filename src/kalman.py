"""
Kalman filter for a dynamic (time-varying) hedge ratio.

A naive pairs trade estimates one hedge ratio by regressing the whole price
history of A on B. That is wrong in two ways: it uses future data to set the
hedge at every past date (lookahead bias), and it assumes the relationship is
constant when in reality hedge ratios drift as fundamentals change. Estimating
the hedge ratio *online* -- updating it each day using only information
available up to that day -- fixes both problems, and doing it as a proper
state-space filter is the clearest signal in the whole project that the author
is not a beginner.

Model (a scalar linear Gaussian state-space model)
--------------------------------------------------
State   hedge_t follows a random walk:  hedge_t = hedge_{t-1} + w_t,  w_t~N(0,Q)
Observe A_t given B_t:                   A_t = hedge_t * B_t + v_t,    v_t~N(0,R)

The hedge is modelled *without* an intercept: the (possibly non-zero, possibly
drifting) mean of the spread is handled downstream by the rolling
standardisation in `signals.rolling_zscore`, and dropping the intercept removes
the hedge/level collinearity that otherwise destabilises the estimate. The
one-step forecast error

    e_t = A_t - hedge_{t-1} * B_t

is the causal, tradeable spread -- it uses only yesterday's hedge, so there is
no lookahead. `delta` sets the process noise and thus how fast the hedge may
move (Chan, 2013); the filter is seeded by OLS on a warm-up window whose spread
is left NaN (never traded).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["KalmanHedge", "kalman_hedge"]


@dataclass
class KalmanHedge:
    """Output of the Kalman hedge filter, all indexed like the input prices."""
    hedge_ratio: pd.Series      # filtered time-varying hedge ratio
    spread: pd.Series           # causal spread e_t = A_t - hedge_{t-1} B_t
    spread_std: pd.Series       # sqrt(one-step forecast variance S_t), a diagnostic

    @property
    def forecast_zscore(self) -> pd.Series:
        """Spread scaled by forecast std -- a diagnostic, not the trade signal."""
        return self.spread / self.spread_std


def kalman_hedge(price_a: pd.Series, price_b: pd.Series, delta: float = 1e-9,
                 warmup: int = 250) -> KalmanHedge:
    """Filter a scalar time-varying hedge ratio of A on B.

    Parameters
    ----------
    delta  : process noise Q = delta/(1-delta). Larger -> hedge adapts faster.
             The default is deliberately small so the hedge tracks only the SLOW
             structural drift of the relationship (which evolves over years) and
             does NOT chase the fast day-to-day mean reversion of the spread
             (which reverts over days) -- if the hedge chases the spread it
             absorbs and destroys the very signal being traded. `run_robustness`
             shows this timescale-separation tradeoff explicitly.
    warmup : OLS seed-window length; spread is NaN over the warm-up (not traded).
    """
    a = price_a.values.astype(float)
    b = price_b.values.astype(float)
    n = len(a)
    warmup = min(warmup, max(20, n // 5))

    # --- seed the hedge and observation noise R from a warm-up OLS -----------
    hedge0 = float(np.polyfit(b[:warmup], a[:warmup], 1)[0])
    R = max(float((a[:warmup] - hedge0 * b[:warmup]).var(ddof=2)), 1e-6)

    Q = delta / (1.0 - delta)
    theta = hedge0
    P = R * 0.01                                  # tight prior around the seed

    hedge = np.full(n, np.nan)
    spread = np.full(n, np.nan)
    spread_sd = np.full(n, np.nan)

    for t in range(n):
        # --- Predict (random-walk state) ---
        P_pred = P + Q

        # --- Observe ---
        e = a[t] - theta * b[t]                   # forecast error == causal spread
        S = b[t] * P_pred * b[t] + R              # forecast variance

        # --- Update ---
        K = P_pred * b[t] / S                      # Kalman gain
        theta = theta + K * e
        P = P_pred - K * b[t] * P_pred

        hedge[t] = theta
        if t >= warmup:                            # exclude convergence transient
            spread[t] = e
            spread_sd[t] = np.sqrt(S)

    idx = price_a.index
    return KalmanHedge(
        hedge_ratio=pd.Series(hedge, index=idx),
        spread=pd.Series(spread, index=idx),
        spread_std=pd.Series(spread_sd, index=idx),
    )
