"""
Initial EDA for Cairo Weather Dataset
Findings and step logs are saved to the logs/ folder.
"""

import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT, "database", "raw", "Cairo-Weather.csv")
LOG_DIR   = os.path.join(ROOT, "logs")
FIG_DIR   = os.path.join(ROOT, "figures", "eda")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── Logging setup ──────────────────────────────────────────────────────────────
log_file = os.path.join(LOG_DIR, f"initial_eda_{datetime.now():%Y%m%d_%H%M%S}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Helper ─────────────────────────────────────────────────────────────────────
def section(title: str) -> None:
    bar = "=" * 70
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 1 — Load data")

df = pd.read_csv(DATA_PATH, parse_dates=["time"], dayfirst=False)
df.sort_values("time", inplace=True)
df.reset_index(drop=True, inplace=True)

log.info(f"Dataset path  : {DATA_PATH}")
log.info(f"Shape         : {df.shape[0]:,} rows × {df.shape[1]} columns")
log.info(f"Date range    : {df['time'].min().date()} → {df['time'].max().date()}")
log.info(f"Columns       :\n{chr(10).join(f'  {c}' for c in df.columns)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Data types & memory
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 2 — Data types & memory usage")

buf = []
for col in df.columns:
    buf.append(f"  {col:<50} {str(df[col].dtype):<12}")
log.info("Column dtypes:\n" + "\n".join(buf))
log.info(f"Memory usage  : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Missing values
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 3 — Missing values")

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"missing_count": missing, "missing_%": missing_pct})
missing_df = missing_df[missing_df["missing_count"] > 0].sort_values("missing_%", ascending=False)

if missing_df.empty:
    log.info("No missing values found.")
else:
    log.info(f"Columns with missing values:\n{missing_df.to_string()}")

# Save missing-value heatmap
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, ax=ax, cmap="viridis")
ax.set_title("Missing Value Heatmap")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "missing_heatmap.png"), dpi=150)
plt.close(fig)
log.info("Saved: figures/eda/missing_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Descriptive statistics
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 4 — Descriptive statistics (numeric columns)")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
desc = df[numeric_cols].describe().T
desc["skewness"] = df[numeric_cols].skew()
desc["kurtosis"] = df[numeric_cols].kurt()
log.info(f"\n{desc.to_string()}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Duplicate rows
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 5 — Duplicate rows")

n_dups = df.duplicated().sum()
log.info(f"Duplicate rows: {n_dups}")
if n_dups:
    log.info(df[df.duplicated()].to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 6. Outlier detection  (IQR method)
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 6 — Outlier detection (IQR method)")

outlier_summary = []
for col in numeric_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    if n_out:
        outlier_summary.append({"column": col, "outliers": n_out,
                                 "lower_fence": round(lo, 3), "upper_fence": round(hi, 3)})

if outlier_summary:
    out_df = pd.DataFrame(outlier_summary).sort_values("outliers", ascending=False)
    log.info(f"Columns with outliers (IQR):\n{out_df.to_string(index=False)}")
else:
    log.info("No outliers detected.")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Distribution plots  (numeric columns)
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 7 — Distribution plots")

n_cols = 3
n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col].dropna(), bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[i].set_title(col, fontsize=8)
    axes[i].tick_params(labelsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("Feature Distributions", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "distributions.png"), dpi=150)
plt.close(fig)
log.info("Saved: figures/eda/distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Correlation matrix
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 8 — Correlation matrix")

corr = df[numeric_cols].corr()

# Top correlations with temperature_2m_mean
target = "temperature_2m_mean (°C)"
if target in corr:
    top_corr = corr[target].drop(target).abs().sort_values(ascending=False).head(10)
    log.info(f"Top correlations with '{target}':\n{top_corr.to_string()}")

fig, ax = plt.subplots(figsize=(18, 15))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.3, annot_kws={"size": 6}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "correlation_matrix.png"), dpi=150)
plt.close(fig)
log.info("Saved: figures/eda/correlation_matrix.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Time-series plots for key weather variables
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 9 — Time-series plots")

key_vars = [
    "temperature_2m_mean (°C)",
    "temperature_2m_min (°C)",
    "temperature_2m_max (°C)",
    "rain_sum (mm)",
    "wind_speed_10m_max (km/h)",
    "cloud_cover_mean (%)",
]
key_vars = [v for v in key_vars if v in df.columns]

fig, axes = plt.subplots(len(key_vars), 1, figsize=(16, 3 * len(key_vars)), sharex=True)
for ax, col in zip(axes, key_vars):
    ax.plot(df["time"], df[col], linewidth=0.6, color="steelblue")
    ax.set_ylabel(col, fontsize=8)
    ax.tick_params(labelsize=7)

axes[-1].xaxis.set_major_locator(mdates.YearLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
fig.suptitle("Key Weather Variables Over Time", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "time_series.png"), dpi=150)
plt.close(fig)
log.info("Saved: figures/eda/time_series.png")


# ══════════════════════════════════════════════════════════════════════════════
# 10. Seasonality — monthly averages
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 10 — Seasonality (monthly averages)")

df["month"] = df["time"].dt.month
monthly = df.groupby("month")[key_vars].mean()
log.info(f"Monthly averages:\n{monthly.to_string()}")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

for i, col in enumerate(key_vars):
    axes[i].bar(monthly.index, monthly[col], color="coral", edgecolor="white")
    axes[i].set_xticks(range(1, 13))
    axes[i].set_xticklabels(month_labels, rotation=45, ha="right", fontsize=7)
    axes[i].set_title(col, fontsize=8)

fig.suptitle("Monthly Averages (Seasonality)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "monthly_seasonality.png"), dpi=150)
plt.close(fig)
log.info("Saved: figures/eda/monthly_seasonality.png")
df.drop(columns=["month"], inplace=True)


# ══════════════════════════════════════════════════════════════════════════════
# 11. Box-plots by month (temperature)
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 11 — Box-plots by month")

df["month"] = df["time"].dt.month
temp_col = "temperature_2m_mean (°C)"

if temp_col in df.columns:
    fig, ax = plt.subplots(figsize=(14, 5))
    df.boxplot(column=temp_col, by="month", ax=ax, grid=False,
               boxprops=dict(color="steelblue"),
               medianprops=dict(color="red", linewidth=2))
    ax.set_xticklabels(month_labels, rotation=45, ha="right")
    ax.set_title(f"Monthly Distribution — {temp_col}")
    ax.set_xlabel("Month")
    plt.suptitle("")
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "boxplot_temp_monthly.png"), dpi=150)
    plt.close(fig)
    log.info("Saved: figures/eda/boxplot_temp_monthly.png")

df.drop(columns=["month"], inplace=True)


# ══════════════════════════════════════════════════════════════════════════════
# 12. EDA summary
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 12 — EDA Summary")

log.info(f"Total rows             : {df.shape[0]:,}")
log.info(f"Total columns          : {df.shape[1]}")
log.info(f"Date range             : {df['time'].min().date()} → {df['time'].max().date()}")
log.info(f"Numeric columns        : {len(numeric_cols)}")
log.info(f"Columns with missing   : {len(missing_df)}")
log.info(f"Duplicate rows         : {n_dups}")
log.info(f"Columns with outliers  : {len(outlier_summary)}")
log.info(f"Log file saved to      : {log_file}")
log.info(f"Figures saved to       : {FIG_DIR}")
log.info("Initial EDA complete.")
