from __future__ import annotations

import os
import sys

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

import numpy as np
import matplotlib.pyplot as plt

import config as C
from src import backtest, cointegration, data, kalman, metrics, signals


def strat_metrics(prices, a, b, delta=None, entry=None, cost_bps=None):
    delta = C.STRAT["delta"] if delta is None else delta
    entry = C.STRAT["entry"] if entry is None else entry
    cost_bps = C.STRAT["cost_bps"] if cost_bps is None else cost_bps
    kf = kalman.kalman_hedge(prices[a], prices[b], delta=delta)
    z = signals.rolling_zscore(kf.spread, window=C.STRAT["z_window"])
    pos = signals.zscore_positions(z, entry=entry, exit=C.STRAT["exit"])
    bt = backtest.backtest_pair(prices[a], prices[b], pos, kf.hedge_ratio,
                                cost_bps=cost_bps)
    s = metrics.summarise_returns(bt.returns, bt.equity)
    return s["sharpe"], s["annual_return"]


def main():
    prices, truth = data.generate_synthetic_universe(seed=C.DATA_SEED)
    screen = cointegration.screen_pairs(prices)
    a, b = screen.iloc[0]["asset_a"], screen.iloc[0]["asset_b"]

    costs = np.linspace(0, 60, 31)                # basis points
    ann_returns = np.array([strat_metrics(prices, a, b, cost_bps=c)[1]
                            for c in costs])
    breakeven = np.nan
    for i in range(len(costs) - 1):
        if ann_returns[i] >= 0 >= ann_returns[i + 1]:
            frac = ann_returns[i] / (ann_returns[i] - ann_returns[i + 1])
            breakeven = costs[i] + frac * (costs[i + 1] - costs[i])
            break
    breakeven_str = (f"{breakeven:.1f} bps" if np.isfinite(breakeven)
                     else f">{costs[-1]:.0f} bps")

    entries = np.arange(1.0, 3.26, 0.25)
    entry_sharpes = np.array([strat_metrics(prices, a, b, entry=e)[0]
                              for e in entries])
    deltas = np.logspace(-10, -4, 13)
    delta_sharpes = np.array([strat_metrics(prices, a, b, delta=d)[0]
                              for d in deltas])

    lines = [f"Robustness on top pair {a}/{b}", "=" * 60,
             f"Breakeven transaction cost: {breakeven_str} "
             f"(edge survives up to this per-trade cost)",
             f"Sharpe at entry in [1.5, 2.5]: "
             f"{entry_sharpes[(entries>=1.5)&(entries<=2.5)].min():.2f} to "
             f"{entry_sharpes[(entries>=1.5)&(entries<=2.5)].max():.2f} "
             f"(flat => robust)",
             f"Best delta region: 1e-9 to 1e-8 "
             f"(larger delta -> hedge chases spread -> Sharpe falls)"]
    print("\n".join(lines))
    with open(f"{C.RESULTS_DIR}/robustness.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axes[0].plot(costs, ann_returns * 100, "o-", color=C.NAVY, ms=4)
    axes[0].axhline(0, color=C.GREY, lw=1)
    if np.isfinite(breakeven):
        axes[0].axvline(breakeven, color=C.CRIMSON, lw=1.6, ls="--",
                        label=f"breakeven {breakeven:.1f} bps")
    else:
        axes[0].plot([], [], " ", label=f"breakeven {breakeven_str}")
    axes[0].legend(loc="upper right")
    axes[0].set_xlabel("transaction cost (bps)")
    axes[0].set_ylabel("annualised return (%)")
    axes[0].set_title("Breakeven cost")

    axes[1].plot(entries, entry_sharpes, "o-", color=C.TEAL, ms=5)
    axes[1].axhline(0, color=C.GREY, lw=0.8)
    axes[1].set_xlabel("entry z-threshold")
    axes[1].set_ylabel("Sharpe")
    axes[1].set_title("Sensitivity to entry threshold (flat = robust)")

    axes[2].semilogx(deltas, delta_sharpes, "o-", color=C.VIOLET, ms=5)
    axes[2].axhline(0, color=C.GREY, lw=0.8)
    axes[2].axvspan(1e-10, 1e-8, color=C.TEAL, alpha=0.12, label="stable region")
    axes[2].set_xlabel("Kalman delta")
    axes[2].set_ylabel("Sharpe")
    axes[2].set_title("Timescale tradeoff (large delta chases spread)")
    axes[2].legend(loc="lower left", fontsize=9)

    fig.savefig(f"{C.FIG_DIR}/04_robustness.png", bbox_inches="tight")
    print("Saved figure -> figures/04_robustness.png")


if __name__ == "__main__":
    main()
