"""Statistical-arbitrage pairs-trading research pipeline.

A cointegration-based pairs-trading strategy built around one question: is the
backtested edge real, or an artefact of overfitting and multiple testing? The
pipeline pairs a Kalman-filtered dynamic hedge ratio and OU spread modeling with
a rigorous evaluation stack -- out-of-sample selection, transaction costs, and a
Deflated Sharpe Ratio that corrects for the number of pairs searched.
"""
from . import (backtest, cointegration, data, kalman, metrics, ou, selection,
               signals)

__all__ = ["data", "cointegration", "kalman", "ou", "signals", "backtest",
           "metrics", "selection"]
__version__ = "0.1.0"
