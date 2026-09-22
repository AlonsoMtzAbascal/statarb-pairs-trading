from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["rolling_zscore", "zscore_positions"]


def rolling_zscore(spread: pd.Series, window: int = 60, min_periods: int = 20) -> pd.Series:
    mean = spread.rolling(window, min_periods=min_periods).mean()
    std = spread.rolling(window, min_periods=min_periods).std(ddof=1)
    return (spread - mean) / std


def zscore_positions(zscore: pd.Series, entry: float = 2.0, exit: float = 0.5,
                     stop: float | None = None) -> pd.Series:
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
