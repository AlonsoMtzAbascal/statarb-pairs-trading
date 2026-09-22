from __future__ import annotations

import os
import sys

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as C
from src import (backtest, cointegration, data, kalman, metrics, ou, signals)


def run_pair(prices, a, b, use_kalman=True):
    if use_kalman:
        kf = kalman.kalman_hedge(prices[a], prices[b], delta=C.STRAT["delta"])
        hedge, spread = kf.hedge_ratio, kf.spread
    else:
        eg = cointegration.engle_granger(prices[a], prices[b])
        hedge = pd.Series(eg["hedge_ratio"], index=prices.index)
        spread = prices[a] - (eg["hedge_ratio"] * prices[b] + eg["intercept"])
    z = signals.rolling_zscore(spread, window=C.STRAT["z_window"])
    pos = signals.zscore_positions(z, entry=C.STRAT["entry"], exit=C.STRAT["exit"])
    bt = backtest.backtest_pair(prices[a], prices[b], pos, hedge,
                                cost_bps=C.STRAT["cost_bps"])
    return bt, spread, z, pos


def main():
    prices, truth = data.generate_synthetic_universe(seed=C.DATA_SEED)
    screen = cointegration.screen_pairs(prices)
    a, b = screen.iloc[0]["asset_a"], screen.iloc[0]["asset_b"]

    bt_k, spread, z, pos = run_pair(prices, a, b, use_kalman=True)
    bt_s, *_ = run_pair(prices, a, b, use_kalman=False)

    sk = metrics.summarise_returns(bt_k.returns, bt_k.equity)
    ss = metrics.summarise_returns(bt_s.returns, bt_s.equity)
    eg = cointegration.engle_granger(prices[a], prices[b])
    ou_fit = ou.fit_ou((prices[a] - (eg["hedge_ratio"] * prices[b]
                                     + eg["intercept"])).values)

    tv = next(p for p in truth.cointegrated_pairs if truth.time_varying[p])
    ta, tb = tv
    bt_tvk, *_ = run_pair(prices, ta, tb, use_kalman=True)
    bt_tvs, *_ = run_pair(prices, ta, tb, use_kalman=False)
    tvk = metrics.summarise_returns(bt_tvk.returns, bt_tvk.equity)
    tvs = metrics.summarise_returns(bt_tvs.returns, bt_tvs.equity)

    def fmt(name, s, bt):
        return (f"{name:<26}{s['sharpe']:>8.2f}{s['sortino']:>9.2f}"
                f"{s['annual_return']:>10.1%}{s['annual_vol']:>9.1%}"
                f"{s['max_drawdown']:>9.1%}{bt.n_trades:>8}")

    lines = [f"Strategy performance (transaction costs {C.STRAT['cost_bps']} bps)",
             "=" * 78,
             f"{'':26}{'Sharpe':>8}{'Sortino':>9}{'AnnRet':>10}"
             f"{'AnnVol':>9}{'MaxDD':>9}{'Trades':>8}",
             f"Top pair {a}/{b} (constant hedge, OU half-life "
             f"{ou_fit.half_life:.0f}d):",
             fmt("  Dynamic (Kalman)", sk, bt_k),
             fmt("  Static (OLS hedge)", ss, bt_s),
             f"Time-varying pair {ta}/{tb} (drifting hedge):",
             fmt("  Dynamic (Kalman)", tvk, bt_tvk),
             fmt("  Static (OLS hedge)", tvs, bt_tvs),
             "-" * 78,
             f"Dynamic-hedge Sharpe uplift: {sk['sharpe']-ss['sharpe']:+.2f} on the "
             f"constant pair (a wash, as it should be),",
             f"{'':29}{tvk['sharpe']-tvs['sharpe']:+.2f} on the drifting pair "
             f"(the Kalman filter earning its keep)."]
    print("\n".join(lines))
    with open(f"{C.RESULTS_DIR}/strategy.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 9.5),
                                        gridspec_kw={"height_ratios": [1, 1, 1]})

    ax1.plot(bt_k.equity.index, bt_k.equity, color=C.TEAL, lw=1.6,
             label=f"Dynamic hedge (Sharpe {sk['sharpe']:.2f})")
    ax1.plot(bt_s.equity.index, bt_s.equity, color=C.GREY, lw=1.3, ls="--",
             label=f"Static hedge (Sharpe {ss['sharpe']:.2f})")
    ax1.axhline(1.0, color=C.NAVY, lw=0.8)
    ax1.set_ylabel("Growth of $1")
    ax1.set_title(f"Headline pair {a}/{b} (constant hedge): dynamic == static")
    ax1.legend(loc="upper left")

    ax2.plot(bt_tvk.equity.index, bt_tvk.equity, color=C.TEAL, lw=1.6,
             label=f"Dynamic hedge (Sharpe {tvk['sharpe']:.2f})")
    ax2.plot(bt_tvs.equity.index, bt_tvs.equity, color=C.CRIMSON, lw=1.3, ls="--",
             label=f"Static hedge (Sharpe {tvs['sharpe']:.2f})")
    ax2.axhline(1.0, color=C.NAVY, lw=0.8)
    ax2.set_ylabel("Growth of $1")
    ax2.set_title(f"Drifting pair {ta}/{tb}: the dynamic hedge earns its keep")
    ax2.legend(loc="upper left")

    entry, exit = C.STRAT["entry"], C.STRAT["exit"]
    ax3.plot(z.index, z, color=C.NAVY, lw=0.7, label="spread z-score")
    ax3.axhline(entry, color=C.CRIMSON, lw=1, ls="--", label=f"+/-{entry} entry")
    ax3.axhline(-entry, color=C.CRIMSON, lw=1, ls="--")
    ax3.axhline(exit, color=C.AMBER, lw=0.8, ls=":", label=f"+/-{exit} exit")
    ax3.axhline(-exit, color=C.AMBER, lw=0.8, ls=":")
    longs = pos[(pos == 1) & (pos.shift(1) != 1)]
    shorts = pos[(pos == -1) & (pos.shift(1) != -1)]
    ax3.scatter(longs.index, z.reindex(longs.index), marker="^", color=C.TEAL,
                s=28, zorder=3, label="enter long")
    ax3.scatter(shorts.index, z.reindex(shorts.index), marker="v",
                color=C.CRIMSON, s=28, zorder=3, label="enter short")
    ax3.set_ylabel("z-score")
    ax3.set_title(f"Signal mechanics on {a}/{b}: spread, bands, and entries")
    ax3.legend(loc="upper left", ncol=3, fontsize=8)

    fig.savefig(f"{C.FIG_DIR}/02_strategy.png", bbox_inches="tight")
    print("Saved figure -> figures/02_strategy.png")


if __name__ == "__main__":
    main()
