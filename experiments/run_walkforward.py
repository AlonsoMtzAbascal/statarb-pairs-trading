"""
Experiment 5 -- Out-of-sample selection (walk-forward).

The deadliest bias in a pairs backtest is selecting the pair on the same data
used to score it. Here selection is made OUT of sample: on each roll, pairs are
chosen using only a past formation window, then traded -- untouched -- over the
following window. Concatenating those untouched windows gives an honest track
record, and comparing it to the in-sample performance on the same pairs shows
how much of the in-sample edge was real versus fitted.

A large in-sample / out-of-sample gap is the classic signature of overfitting;
a modest gap means the selection generalises.
"""
from __future__ import annotations

import os
import sys

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

import numpy as np
import matplotlib.pyplot as plt

import config as C
from src import data, metrics, selection


def main():
    prices, truth = data.generate_synthetic_universe(seed=C.DATA_SEED)

    wf = selection.walk_forward(
        prices, formation=504, trading=126, top_k=3,
        entry=C.STRAT["entry"], exit=C.STRAT["exit"], cost_bps=C.STRAT["cost_bps"])

    is_sharpe = metrics.sharpe_ratio(wf.is_returns.values)
    oos_sharpe = metrics.sharpe_ratio(wf.oos_returns.values)
    oos_sum = metrics.summarise_returns(wf.oos_returns.values)

    lines = ["Walk-forward out-of-sample evaluation", "=" * 58,
             "Formation window: 504d  |  trading window: 126d  |  top-k: 3",
             "",
             f"In-sample Sharpe (selected pairs)   : {is_sharpe:.2f}",
             f"Out-of-sample Sharpe (never seen)   : {oos_sharpe:.2f}",
             f"Out-of-sample annual return         : {oos_sum['annual_return']:.1%}",
             f"Out-of-sample max drawdown          : {oos_sum['max_drawdown']:.1%}",
             f"IS -> OOS Sharpe decay              : "
             f"{is_sharpe - oos_sharpe:+.2f} "
             f"({'modest, generalises' if (is_sharpe - oos_sharpe) < is_sharpe*0.6 else 'large, overfit warning'})",
             "",
             "Pairs selected per roll (out-of-sample):"]
    for date, pairs in wf.selections:
        tag = ", ".join(f"{a}/{b}" for a, b in pairs) if pairs else "(none flagged)"
        lines.append(f"  as of {date.date()}: {tag}")
    print("\n".join(lines))
    with open(f"{C.RESULTS_DIR}/walkforward.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    # --- figure --------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    ax1.plot(wf.oos_equity.index, wf.oos_equity, color=C.TEAL, lw=1.6,
             label=f"out-of-sample (Sharpe {oos_sharpe:.2f})")
    ax1.axhline(1.0, color=C.NAVY, lw=0.8)
    ax1.set_ylabel("Growth of $1 (out-of-sample)")
    ax1.set_title("Concatenated out-of-sample equity curve")
    ax1.legend(loc="upper left")

    bars = ax2.bar(["In-sample", "Out-of-sample"], [is_sharpe, oos_sharpe],
                   color=[C.GREY, C.TEAL], edgecolor="white")
    for bar, v in zip(bars, [is_sharpe, oos_sharpe]):
        ax2.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                 ha="center", va="bottom", fontweight="bold")
    ax2.axhline(0, color=C.GREY, lw=0.8)
    ax2.set_ylabel("Sharpe ratio")
    ax2.set_title("In-sample vs out-of-sample")

    fig.savefig(f"{C.FIG_DIR}/05_walkforward.png", bbox_inches="tight")
    print("\nSaved figure -> figures/05_walkforward.png")


if __name__ == "__main__":
    main()
