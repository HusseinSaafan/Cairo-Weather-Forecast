"""
Initial EDA on database/train.csv
- Descriptive statistics
- Distribution plot (histogram + KDE + box-plot)
- Periodogram (power spectral density via FFT)

Figures saved to figures/eda/
Findings logged to logs/<date>/
"""

import os
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import get_logger
from src.utils.helpers import load_data

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH = os.path.join(PROJECT_ROOT, "database", "train.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "eda")
TARGET_COL = "temperature_2m_mean (°C)"
TIME_COL = "time"

os.makedirs(FIG_DIR, exist_ok=True)

# ── Logger ─────────────────────────────────────────────────────────────────────
log, log_file = get_logger(
    "src.eda.initial_eda",
    log_filename=f"initial_eda_{datetime.now():%Y%m%d_%H%M%S}.log",
)


def section(title: str) -> None:
    bar = "=" * 70
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load training data
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 1 — Load training data")

df = load_data(TRAIN_PATH, parse_dates=[TIME_COL], index_col=TIME_COL)
if df is None:
    raise RuntimeError(f"Could not load: {TRAIN_PATH}")

df.sort_index(inplace=True)

log.info(f"Source        : {TRAIN_PATH}")
log.info(f"Shape         : {df.shape[0]:,} rows × {df.shape[1]} columns")
log.info(f"Date range    : {df.index.min().date()} → {df.index.max().date()}")
log.info(f"Index freq    : {pd.infer_freq(df.index)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Descriptive statistics
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 2 — Descriptive statistics")

series: pd.Series = df[TARGET_COL]

stats = series.describe()
skewness = series.skew()
kurt = series.kurt()

log.info(f"Column: {TARGET_COL!r}")
log.info(f"\n{stats.to_string()}")
log.info(f"Skewness      : {skewness:.4f}")
log.info(f"Kurtosis      : {kurt:.4f}")
log.info(f"Missing values: {series.isna().sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Distribution plot — histogram + KDE + box-plot
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 3 — Distribution plot")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram + KDE
ax_hist = axes[0]
ax_hist.hist(series.dropna(), bins=50, color="steelblue",
             edgecolor="white", linewidth=0.4, density=True, label="Histogram")

# Overlay KDE using a manual Gaussian KDE for readability
from scipy.stats import gaussian_kde
kde = gaussian_kde(series.dropna())
x_kde = np.linspace(series.min(), series.max(), 300)
ax_hist.plot(x_kde, kde(x_kde), color="crimson", linewidth=2, label="KDE")
ax_hist.axvline(series.mean(), color="orange", linewidth=1.5,
                linestyle="--", label=f"Mean={series.mean():.1f}")
ax_hist.axvline(series.median(), color="green", linewidth=1.5,
                linestyle=":", label=f"Median={series.median():.1f}")
ax_hist.set_title("Distribution — Histogram + KDE")
ax_hist.set_xlabel(TARGET_COL)
ax_hist.set_ylabel("Density")
ax_hist.legend(fontsize=8)

# Box-plot
ax_box = axes[1]
ax_box.boxplot(series.dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.6),
               medianprops=dict(color="crimson", linewidth=2),
               whiskerprops=dict(linewidth=1.2),
               capprops=dict(linewidth=1.2),
               flierprops=dict(marker="o", markerfacecolor="orange",
                               markersize=4, alpha=0.6))
ax_box.set_title("Box-plot")
ax_box.set_ylabel(TARGET_COL)
ax_box.set_xticks([])

fig.suptitle(f"Distribution of {TARGET_COL}", fontsize=13, fontweight="bold")
plt.tight_layout()

dist_path = os.path.join(FIG_DIR, "train_distribution.png")
fig.savefig(dist_path, dpi=150)
plt.close(fig)
log.info(f"Saved: {dist_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Periodogram — dominant seasonal cycles
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 4 — Periodogram")

freq, per = periodogram(series.values)

# Skip zero frequency to avoid division by zero when plotting 1/frequency.
period_days = 1 / freq[1:]
power = per[1:]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(period_days, power, color="steelblue")
ax.set_title("Periodogram")
ax.set_xlabel("Period (days)")
ax.set_ylabel("Power")
ax.set_xlim(left=0)
plt.tight_layout()

pgram_path = os.path.join(FIG_DIR, "train_periodogram.png")
fig.savefig(pgram_path, dpi=150)
plt.close(fig)
log.info(f"Saved: {pgram_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Summary
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 5 — Summary")
log.info(f"Rows analyzed  : {len(series):,}")
log.info(f"Date range     : {df.index.min().date()} → {df.index.max().date()}")
log.info(f"Mean temp      : {series.mean():.2f} °C")
log.info(f"Std dev        : {series.std():.2f} °C")
log.info(f"Min / Max      : {series.min():.1f} / {series.max():.1f} °C")
log.info(f"Log file       : {log_file}")
log.info(f"Figures saved  : {FIG_DIR}")
log.info("Initial EDA complete.")
