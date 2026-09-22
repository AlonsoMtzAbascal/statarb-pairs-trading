from __future__ import annotations

import os
import sys

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)),
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

import config as C
from src import cointegration, data, kalman, ou


def main():
    prices, truth = data.generate_synthetic_universe(seed=C.DATA_SEED)
    true_pairs = set(truth.cointegrated_pairs)

    screen = cointegration.screen_pairs(prices)
    screen["is_true_pair"] = [
        (a, b) in true_pairs or (b, a) in true_pairs
        for a, b in zip(screen["asset_a"], screen["asset_b"])]
    top = screen.head(8)

    lines = ["Cointegration screen (45 pairs), ranked by Engle-Granger p-value",
             "-" * 66,
             f"{'pair':<20}{'EG p-value':>13}{'Johansen':>11}{'true?':>8}"]
    for _, r in top.iterrows():
        lines.append(f"{r.asset_a+'/'+r.asset_b:<20}{r.eg_pvalue:>13.2e}"
                     f"{str(r.johansen_cointegrated):>11}"
                     f"{'  <-- YES' if r.is_true_pair else '':>8}")
    n_true_top3 = int(screen.head(3)["is_true_pair"].sum())
    lines.append("-" * 66)
    lines.append(f"True pairs in top 3 by p-value: {n_true_top3}/3")
    print("\n".join(lines))

    tv_pair = next(p for p in truth.cointegrated_pairs if truth.time_varying[p])
    a, b = tv_pair
    kf = kalman.kalman_hedge(prices[a], prices[b], delta=C.STRAT["delta"])
    true_beta = pd.Series(truth.true_beta[tv_pair], index=prices.index)
    mask = kf.hedge_ratio.notna() & (np.arange(len(prices)) > 300)
    corr = np.corrcoef(kf.hedge_ratio[mask], true_beta[mask])[0, 1]
    print(f"\nKalman hedge recovery on time-varying pair {a}/{b}:")
    print(f"  true beta: {true_beta.iloc[0]:.3f} -> {true_beta.iloc[-1]:.3f};  "
          f"corr(estimated, true) = {corr:.3f}")

    print("\nOU half-life recovery (static Engle-Granger spread):")
    ou_lines = []
    for pair in truth.cointegrated_pairs:
        pa, pb = pair
        eg = cointegration.engle_granger(prices[pa], prices[pb])
        spread = prices[pa].values - (eg["hedge_ratio"] * prices[pb].values
                                      + eg["intercept"])
        fit = ou.fit_ou(spread)
        adf_p = adfuller(spread)[1]
        thl = truth.true_half_life[pair]
        tag = "time-varying" if truth.time_varying[pair] else "constant"
        hl = f"{fit.half_life:5.1f}" if np.isfinite(fit.half_life) else "  n/a"
        msg = (f"  {pa}/{pb:<9} ({tag:<12}): true={thl:>2}d  est={hl}d  "
               f"ADF p={adf_p:.3f}")
        print(msg)
        ou_lines.append(msg)

    with open(f"{C.RESULTS_DIR}/validation.txt", "w") as f:
        f.write("\n".join(lines) + "\n\n")
        f.write(f"Kalman time-varying hedge corr(estimated,true) = {corr:.3f}\n\n")
        f.write("OU half-life recovery (static-hedge spread):\n")
        f.write("\n".join(ou_lines) + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.plot(prices.index, true_beta, color=C.NAVY, lw=2.2, label="true beta_t")
    ax1.plot(prices.index, kf.hedge_ratio, color=C.CRIMSON, lw=1.1, alpha=0.9,
             label="Kalman estimate")
    ax1.set_title(f"Kalman tracks the drifting hedge (corr={corr:.2f})")
    ax1.set_ylabel("hedge ratio")
    ax1.legend(loc="upper right")

    # static vs dynamic spread for the time-varying pair
    eg = cointegration.engle_granger(prices[a], prices[b])
    static_spread = prices[a].values - (eg["hedge_ratio"] * prices[b].values
                                        + eg["intercept"])
    ax2.plot(prices.index, static_spread, color=C.GREY, lw=0.9,
             label="static-hedge spread (drifts)")
    ax2.plot(prices.index, kf.spread, color=C.TEAL, lw=0.9,
             label="Kalman-hedge spread (stationary)")
    ax2.axhline(0, color=C.NAVY, lw=0.8)
    ax2.set_title("Static hedge breaks when the true hedge drifts")
    ax2.set_ylabel("spread")
    ax2.legend(loc="upper left", fontsize=9)

    fig.savefig(f"{C.FIG_DIR}/01_validation.png", bbox_inches="tight")
    print("\nSaved figure -> figures/01_validation.png")


if __name__ == "__main__":
    main()
