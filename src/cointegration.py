from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

__all__ = ["engle_granger", "johansen", "screen_pairs"]


def engle_granger(price_a: pd.Series, price_b: pd.Series) -> dict:
    # statsmodels coint() runs the Engle-Granger test directly.
    t_stat, p_value, _ = coint(price_a, price_b)

    # Recover the static hedge ratio and residual diagnostics via OLS.
    b = np.vstack([price_b.values, np.ones(len(price_b))]).T
    hedge, intercept = np.linalg.lstsq(b, price_a.values, rcond=None)[0]
    spread = price_a.values - (hedge * price_b.values + intercept)
    adf_p = adfuller(spread, autolag="AIC", result_object=False)[1]

    return {"p_value": p_value, "t_stat": t_stat, "hedge_ratio": hedge,
            "intercept": intercept, "adf_p_spread": adf_p}


def johansen(price_a: pd.Series, price_b: pd.Series, det_order=0, k_ar_diff=1) -> dict:
    data = np.column_stack([price_a.values, price_b.values])
    res = coint_johansen(data, det_order, k_ar_diff)

    trace_stat = res.lr1[0]                 # trace statistic for r = 0
    crit_95 = res.cvt[0, 1]                 # 90% / 95% / 99% -> take 95%
    vec = res.evec[:, 0]                    # cointegrating vector
    hedge = -vec[1] / vec[0]                # normalise on asset A
    return {"trace_stat": trace_stat, "crit_95": crit_95,
            "cointegrated": bool(trace_stat > crit_95), "hedge_ratio": hedge}


def screen_pairs(prices: pd.DataFrame, eg_pvalue=0.05) -> pd.DataFrame:
    rows = []
    for a, b in combinations(prices.columns, 2):
        eg = engle_granger(prices[a], prices[b])
        joh = johansen(prices[a], prices[b])
        rows.append({
            "asset_a": a, "asset_b": b,
            "eg_pvalue": eg["p_value"],
            "johansen_trace": joh["trace_stat"],
            "johansen_crit95": joh["crit_95"],
            "johansen_cointegrated": joh["cointegrated"],
            "hedge_ratio": eg["hedge_ratio"],
            "flagged": eg["p_value"] < eg_pvalue,
        })
    return (pd.DataFrame(rows)
            .sort_values("eg_pvalue")
            .reset_index(drop=True))
