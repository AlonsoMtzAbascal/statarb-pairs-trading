from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import backtest_pair
from .cointegration import screen_pairs
from .kalman import kalman_hedge
from .signals import rolling_zscore, zscore_positions

__all__ = ["WalkForwardResult", "walk_forward"]


@dataclass
class WalkForwardResult:
    oos_returns: pd.Series
    is_returns: pd.Series
    selections: list = field(default_factory=list)   # (date, [pairs]) per roll

    @property
    def oos_equity(self) -> pd.Series:
        return (1.0 + self.oos_returns).cumprod()


def walk_forward(prices: pd.DataFrame, formation: int = 504, trading: int = 126,
                 top_k: int = 3, entry: float = 2.0, exit: float = 0.5,
                 cost_bps: float = 1.0, eg_pvalue: float = 0.05) -> WalkForwardResult:
    n = len(prices)
    oos_parts, is_parts, selections = [], [], []

    start = 0
    while start + formation + trading <= n:
        form = prices.iloc[start:start + formation]
        full = prices.iloc[start:start + formation + trading]

        screen = screen_pairs(form, eg_pvalue=eg_pvalue)
        chosen = screen[screen["flagged"]].head(top_k)
        pairs = list(zip(chosen["asset_a"], chosen["asset_b"]))
        selections.append((form.index[-1], pairs))
        if not pairs:
            start += trading
            continue

        oos_pair_rets, is_pair_rets = [], []
        for a, b in pairs:
            kf = kalman_hedge(full[a], full[b])
            z = rolling_zscore(kf.spread, window=40)
            pos = zscore_positions(z, entry=entry, exit=exit)
            bt = backtest_pair(full[a], full[b], pos, kf.hedge_ratio,
                               cost_bps=cost_bps)
            is_pair_rets.append(bt.returns.iloc[:formation])
            oos_pair_rets.append(bt.returns.iloc[formation:])

        oos_parts.append(pd.concat(oos_pair_rets, axis=1).mean(axis=1))
        is_parts.append(pd.concat(is_pair_rets, axis=1).mean(axis=1))
        start += trading

    oos = pd.concat(oos_parts) if oos_parts else pd.Series(dtype=float)
    is_ = pd.concat(is_parts) if is_parts else pd.Series(dtype=float)
    # de-duplicate any overlapping index labels defensively
    oos = oos[~oos.index.duplicated(keep="first")]
    is_ = is_[~is_.index.duplicated(keep="first")]
    return WalkForwardResult(oos_returns=oos, is_returns=is_, selections=selections)
