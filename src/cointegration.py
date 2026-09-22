"""
Cointegration testing.

Two assets are cointegrated if some linear combination of their (individually
non-stationary) prices is stationary -- i.e. a spread that keeps reverting to a
mean, which is the entire basis for a pairs trade. Correlation is not enough:
two series can be highly correlated yet drift apart forever. Cointegration is
the property that actually matters, so we test for it directly.

Two complementary tests:

* Engle-Granger -- regress one price on the other, then ADF-test the residual
  spread for stationarity. Simple and intuitive, but asymmetric (the answer can
  depend on which asset is the regressor) and limited to a single relationship.

* Johansen -- a systoms-based test that treats the assets symmetrically, can
  detect cointegration among more than two series, and returns the
  cointegrating vector via eigen-decomposition. More robust; used as the
  primary screen here.

`screen_pairs` runs both across every pair in a price panel and returns a tidy
table, which the multiple-testing analysis then consumes.
"""
from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# statsmodels 0.15 emits a FutureWarning about adfuller's return type that is
# irrelevant to how it is used here; silence it locally.
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

__all__ = ["engle_granger", "johansen", "screen_pairs"]


def engle_granger(price_a: pd.Series, price_b: pd.Series) -> dict:
    """Engle-Granger two-step test. Returns p-value and static hedge ratio.

    The hedge ratio is the OLS slope of A on B (with intercept); the spread is
    A - hedge*B. A low p-value rejects the null of no cointegration.
    """
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
    """Johansen trace test for a two-asset system.

    Compares the trace statistic for rank r = 0 against its 95% critical value.
    If it exceeds the critical value we reject "no cointegration". The
    cointegrating vector (first eigenvector) gives a symmetric hedge ratio.
    """
    data = np.column_stack([price_a.values, price_b.values])
    res = coint_johansen(data, det_order, k_ar_diff)

    trace_stat = res.lr1[0]                 # trace statistic for r = 0
    crit_95 = res.cvt[0, 1]                 # 90% / 95% / 99% -> take 95%
    vec = res.evec[:, 0]                    # cointegrating vector
    hedge = -vec[1] / vec[0]                # normalise on asset A
    return {"trace_stat": trace_stat, "crit_95": crit_95,
            "cointegrated": bool(trace_stat > crit_95), "hedge_ratio": hedge}


def screen_pairs(prices: pd.DataFrame, eg_pvalue=0.05) -> pd.DataFrame:
    """Run both tests on every pair; return a table sorted by Engle-Granger p.

    Columns: asset_a, asset_b, eg_pvalue, johansen_trace, johansen_crit95,
    johansen_cointegrated, hedge_ratio, flagged (EG p < threshold).
    """
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
