"""
Backtest engine for a single pair.

Turns a position series and a (possibly time-varying) hedge ratio into a dollar
PnL, a return stream, and an equity curve, with transaction costs charged on
turnover. The accounting is deliberately explicit and causal -- every quantity
that decides day t's PnL is known at the end of day t-1 -- because a backtest's
credibility rests entirely on there being no lookahead.

Accounting
----------
The spread portfolio for one unit of position is: long 1 share of A, short
`hedge` shares of B. Its one-day dollar PnL is

    pnl_t = pos_{t-1} * [ (A_t - A_{t-1}) - hedge_{t-1} * (B_t - B_{t-1}) ]

Costs are charged whenever the position (or the hedge, which changes the B leg)
changes, proportional to the dollar value traded:

    traded_t = |pos_t*A_t - pos_{t-1}*A_{t-1}|        (A leg)
             + |pos_t*hedge_t*B_t - pos_{t-1}*hedge_{t-1}*B_t|   (B leg)
    cost_t   = cost_bps/1e4 * traded_t

Returns are expressed on the gross capital deployed (|A| + |hedge*B| per unit
position), so the Sharpe is that of a self-financing dollar-neutral book.
"""
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
    """Backtest one pair given a causal position series and a hedge ratio.

    hedge_ratio may be a scalar (static hedge) or a Series aligned to prices
    (dynamic hedge from the Kalman filter).
    """
    a = price_a.astype(float)
    b = price_b.astype(float)
    pos = positions.reindex(a.index).fillna(0.0)

    if np.isscalar(hedge_ratio):
        hedge = pd.Series(float(hedge_ratio), index=a.index)
    else:
        hedge = pd.Series(hedge_ratio).reindex(a.index).ffill().fillna(0.0)

    pos_lag = pos.shift(1).fillna(0.0)
    hedge_lag = hedge.shift(1).fillna(0.0)

    # --- gross PnL from holding the spread over the day (causal) -------------
    dA = a.diff().fillna(0.0)
    dB = b.diff().fillna(0.0)
    gross_pnl = pos_lag * (dA - hedge_lag * dB)

    # --- transaction costs on both legs when exposure changes ---------------
    a_leg = (pos * a) - (pos_lag * a)                      # change in A holding ($)
    b_leg = (pos * hedge * b) - (pos_lag * hedge_lag * b)  # change in B holding ($)
    traded = a_leg.abs() + b_leg.abs()
    cost = (cost_bps / 1e4) * traded

    net_pnl = gross_pnl - cost

    # --- returns on gross capital deployed ----------------------------------
    gross_capital = (pos_lag.abs() * a) + (pos_lag.abs() * hedge_lag.abs() * b)
    capital_base = gross_capital.replace(0.0, np.nan)
    returns = (net_pnl / capital_base).fillna(0.0)
    equity = (1.0 + returns).cumprod()

    n_trades = int((pos != pos_lag).sum())
    return BacktestResult(returns=returns, equity=equity, pnl=net_pnl,
                          positions=pos, turnover=traded, n_trades=n_trades,
                          cost_bps=cost_bps)
