"""
Trading-signal generation from the standardised spread.

Given the z-score of the spread (from the Kalman filter), we take a classic
mean-reversion position: when the spread is unusually high the pair is expected
to converge, so we SHORT the spread; when unusually low we go LONG; and we flat
out once it has reverted near the mean. A wider entry band than exit band gives
hysteresis, so we are not whipsawed in and out around a single threshold.

Positions are generated causally: the position held *into* day t uses the signal
observed at the close of day t-1, so there is no lookahead.

Sign convention: position = +1 means LONG the spread (long asset A, short
hedge*B); -1 means SHORT the spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["rolling_zscore", "zscore_positions"]


def rolling_zscore(spread: pd.Series, window: int = 60, min_periods: int = 20) -> pd.Series:
    """Standardise a spread against its own trailing mean and std.

    Causal by construction: the mean and std at time t use only the trailing
    `window` observations up to and including t, so the resulting z-score can be
    acted on the next day with no lookahead. A rolling standardisation (rather
    than the Kalman forecast std) keeps the entry/exit thresholds interpretable
    in units of the spread's recent dispersion.
    """
    mean = spread.rolling(window, min_periods=min_periods).mean()
    std = spread.rolling(window, min_periods=min_periods).std(ddof=1)
    return (spread - mean) / std


def zscore_positions(zscore: pd.Series, entry: float = 2.0, exit: float = 0.5,
                     stop: float | None = None) -> pd.Series:
    """Convert a spread z-score into a {-1, 0, +1} position series (stateful).

    entry : |z| at which to open a position (fade the move).
    exit  : |z| at which to close back to flat.
    stop  : optional |z| beyond which to force flat (a divergence stop-loss).
    """
    z = zscore.values
    pos = np.zeros(len(z))
    state = 0
    for t in range(len(z)):
        zt = z[t]
        if np.isnan(zt):
            pos[t] = 0
            continue
        if state == 0:
            if zt >= entry:
                state = -1            # spread too high -> short it
            elif zt <= -entry:
                state = +1            # spread too low  -> long it
        else:
            # exit on reversion to the mean
            if abs(zt) <= exit:
                state = 0
            # optional divergence stop
            elif stop is not None and abs(zt) >= stop:
                state = 0
        pos[t] = state
    return pd.Series(pos, index=zscore.index)
