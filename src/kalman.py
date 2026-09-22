from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["KalmanHedge", "kalman_hedge"]


@dataclass
class KalmanHedge:
    hedge_ratio: pd.Series      # filtered time-varying hedge ratio
    spread: pd.Series           # causal spread e_t = A_t - hedge_{t-1} B_t
    spread_std: pd.Series       # sqrt(one-step forecast variance S_t), a diagnostic

    @property
    def forecast_zscore(self) -> pd.Series:
        return self.spread / self.spread_std


def kalman_hedge(price_a: pd.Series, price_b: pd.Series, delta: float = 1e-9,
                 warmup: int = 250) -> KalmanHedge:
    a = price_a.values.astype(float)
    b = price_b.values.astype(float)
    n = len(a)
    warmup = min(warmup, max(20, n // 5))

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
