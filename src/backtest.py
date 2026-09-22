from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["BacktestResult", "backtest_pair"]


@dataclass
class BacktestResult:
    returns: pd.Series          # per-period returns on gross capital
    equity: pd.Series           # cumulative (1+r) product
    pnl: pd.Series              # per-period dollar PnL (net of costs)
    positions: pd.Series
    turnover: pd.Series         # dollar traded each period
    n_trades: int
    cost_bps: float

    @property
    def gross_leverage(self) -> float:
        return 1.0


def backtest_pair(price_a: pd.Series, price_b: pd.Series, positions: pd.Series,
                  hedge_ratio, cost_bps: float = 1.0) -> BacktestResult:
                      
    a = price_a.astype(float)
    b = price_b.astype(float)
    pos = positions.reindex(a.index).fillna(0.0)

    if np.isscalar(hedge_ratio):
        hedge = pd.Series(float(hedge_ratio), index=a.index)
    else:
        hedge = pd.Series(hedge_ratio).reindex(a.index).ffill().fillna(0.0)

    pos_lag = pos.shift(1).fillna(0.0)
    hedge_lag = hedge.shift(1).fillna(0.0)

    dA = a.diff().fillna(0.0)
    dB = b.diff().fillna(0.0)
    gross_pnl = pos_lag * (dA - hedge_lag * dB)

    a_leg = (pos * a) - (pos_lag * a)                      # change in A holding ($)
    b_leg = (pos * hedge * b) - (pos_lag * hedge_lag * b)  # change in B holding ($)
    traded = a_leg.abs() + b_leg.abs()
    cost = (cost_bps / 1e4) * traded

    net_pnl = gross_pnl - cost

    gross_capital = (pos_lag.abs() * a) + (pos_lag.abs() * hedge_lag.abs() * b)
    capital_base = gross_capital.replace(0.0, np.nan)
    returns = (net_pnl / capital_base).fillna(0.0)
    equity = (1.0 + returns).cumprod()

    n_trades = int((pos != pos_lag).sum())
    return BacktestResult(returns=returns, equity=equity, pnl=net_pnl,
                          positions=pos, turnover=traded, n_trades=n_trades,
                          cost_bps=cost_bps)
