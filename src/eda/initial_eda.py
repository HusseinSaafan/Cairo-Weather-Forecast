"""
Initial EDA on database/train.csv
- Descriptive statistics
- Raw time-series plot
- Rolling mean and rolling standard deviation
- Lag features and percentage-change features
- Distribution plot (histogram + KDE + box-plot)
- Periodogram (power spectral density via FFT)
- Outlier detection (IQR, Z-score, rolling Z-score)

Figures saved to figures/eda/
Findings logged to logs/<date>/
"""

import os
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
ROLLING_WINDOWS = (7, 30)
LAG_PERIODS = (1, 7, 30)
Z_THRESHOLD = 3.0
ROLLING_Z_WINDOW = 30

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
# 3. Raw series plot + rolling statistics
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 3 — Raw series and rolling statistics")

roll_7_mean = series.rolling(window=ROLLING_WINDOWS[0], min_periods=1).mean()
roll_7_std = series.rolling(window=ROLLING_WINDOWS[0], min_periods=1).std()
roll_30_mean = series.rolling(window=ROLLING_WINDOWS[1], min_periods=1).mean()
roll_30_std = series.rolling(window=ROLLING_WINDOWS[1], min_periods=1).std()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(series.index, series.values, color="steelblue", linewidth=0.9, label="Raw series")
ax.plot(roll_7_mean.index, roll_7_mean.values, color="orange", linewidth=1.2, label="Rolling mean (7d)")
ax.plot(roll_30_mean.index, roll_30_mean.values, color="crimson", linewidth=1.2, label="Rolling mean (30d)")

date_locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
ax.xaxis.set_major_locator(date_locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))

ax.set_title("Daily Temperature Series with Rolling Means", fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel(TARGET_COL)
ax.legend(loc="best", fontsize=8)
fig.tight_layout()

raw_roll_path = os.path.join(FIG_DIR, "train_series_with_rolling_mean.png")
fig.savefig(raw_roll_path, dpi=150)
plt.close(fig)
log.info(f"Saved: {raw_roll_path}")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(roll_7_std.index, roll_7_std.values, color="purple", linewidth=1.0, label="Rolling std (7d)")
ax.plot(roll_30_std.index, roll_30_std.values, color="darkgreen", linewidth=1.0, label="Rolling std (30d)")
ax.xaxis.set_major_locator(date_locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
ax.set_title("Rolling Standard Deviation", fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Standard deviation")
ax.legend(loc="best", fontsize=8)
fig.tight_layout()

roll_std_path = os.path.join(FIG_DIR, "train_rolling_std.png")
fig.savefig(roll_std_path, dpi=150)
plt.close(fig)
log.info(f"Saved: {roll_std_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Lag features and percentage changes
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 4 — Lag and percentage-change features")

feature_df = pd.DataFrame(index=series.index)
feature_df[TARGET_COL] = series
feature_df["rolling_mean_7"] = roll_7_mean
feature_df["rolling_std_7"] = roll_7_std
feature_df["rolling_mean_30"] = roll_30_mean
feature_df["rolling_std_30"] = roll_30_std

for lag in LAG_PERIODS:
    feature_df[f"lag_{lag}"] = series.shift(lag)
    feature_df[f"pct_change_{lag}"] = series.pct_change(periods=lag) * 100.0

nan_ratio = feature_df.isna().mean().sort_values(ascending=False)
log.info("Top NaN ratios introduced by feature engineering:")
log.info(f"\n{nan_ratio.head(8).to_string()}")
log.info("Feature sample (last 5 rows):")
log.info(f"\n{feature_df.tail(5).to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Distribution plot — histogram + KDE + box-plot
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 5 — Distribution plot")

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
# 6. Periodogram — dominant seasonal cycles
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 6 — Periodogram")

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
# 7. Outlier detection + visualization
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 7 — Outlier detection")

q1 = series.quantile(0.25)
q3 = series.quantile(0.75)
iqr = q3 - q1
iqr_lower = q1 - 1.5 * iqr
iqr_upper = q3 + 1.5 * iqr
iqr_mask = (series < iqr_lower) | (series > iqr_upper)

series_std = series.std()
if series_std == 0:
    z_mask = pd.Series(False, index=series.index)
else:
    z_scores = (series - series.mean()) / series_std
    z_mask = z_scores.abs() > Z_THRESHOLD

rolling_mean = series.rolling(window=ROLLING_Z_WINDOW, min_periods=ROLLING_Z_WINDOW).mean()
rolling_std = series.rolling(window=ROLLING_Z_WINDOW, min_periods=ROLLING_Z_WINDOW).std()
rolling_z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
rolling_z_mask = rolling_z.abs() > Z_THRESHOLD
rolling_z_mask = rolling_z_mask.fillna(False)

combined_mask = iqr_mask | z_mask | rolling_z_mask

log.info(f"IQR outliers            : {int(iqr_mask.sum())}")
log.info(f"Global Z-score outliers : {int(z_mask.sum())}")
log.info(f"Rolling Z-score outliers: {int(rolling_z_mask.sum())}")
log.info(f"Combined unique outliers: {int(combined_mask.sum())}")

fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(series.index, series.values, color="steelblue", linewidth=0.8, label="Raw series")
ax.scatter(series.index[iqr_mask], series[iqr_mask], color="crimson", s=14, label="IQR outliers")
ax.scatter(series.index[z_mask], series[z_mask], color="goldenrod", s=12, label="Z-score outliers")
ax.scatter(
    series.index[rolling_z_mask],
    series[rolling_z_mask],
    color="purple",
    s=10,
    label="Rolling Z-score outliers",
)
ax.set_title("Outlier Candidates in Time Series", fontweight="bold")
ax.set_ylabel(TARGET_COL)
ax.legend(loc="upper left", fontsize=8)

ax.xaxis.set_major_locator(date_locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))

fig.tight_layout()
outlier_path = os.path.join(FIG_DIR, "train_outlier_detection.png")
fig.savefig(outlier_path, dpi=150)
plt.close(fig)
log.info(f"Saved: {outlier_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Summary
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 8 — Summary")
log.info(f"Rows analyzed  : {len(series):,}")
log.info(f"Date range     : {df.index.min().date()} → {df.index.max().date()}")
log.info(f"Mean temp      : {series.mean():.2f} °C")
log.info(f"Std dev        : {series.std():.2f} °C")
log.info(f"Min / Max      : {series.min():.1f} / {series.max():.1f} °C")
log.info(f"Outliers (IQR) : {int(iqr_mask.sum())}")
log.info(f"Outliers (Z)   : {int(z_mask.sum())}")
log.info(f"Outliers (RZ)  : {int(rolling_z_mask.sum())}")
log.info(f"Log file       : {log_file}")
log.info(f"Figures saved  : {FIG_DIR}")
log.info("Initial EDA complete.")
