from __future__ import annotations

import os
import warnings

import matplotlib as mpl

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_SEED = 7
STRAT = dict(
    delta=1e-9,        # Kalman process noise (slow-drift tracking)
    z_window=40,       # rolling window for the spread z-score
    entry=2.0,         # |z| to open
    exit=0.5,          # |z| to close
    cost_bps=1.0,      # per-trade transaction cost in basis points
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

NAVY = "#1f3b57"
TEAL = "#2a9d8f"
AMBER = "#e9a13b"
CRIMSON = "#c1443c"
GREY = "#8a8f98"
VIOLET = "#6d597a"

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.autolayout": True,
})
