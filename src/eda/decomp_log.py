"""Seasonal decomposition on log-transformed train data.

Using log(y) converts multiplicative structure in original scale into
approximately additive structure in log scale.
"""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import get_logger
from src.utils.helpers import load_data


TRAIN_PATH = os.path.join(PROJECT_ROOT, "database", "train.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "eda")
TIME_COL = "time"
TARGET_COL = "temperature_2m_mean (°C)"
SEASONAL_PERIOD = 365


def main() -> None:
	logger, log_file = get_logger(
		"src.eda.decomp_log",
		log_filename=f"decomp_log_{datetime.now():%Y%m%d_%H%M%S}.log",
	)

	os.makedirs(FIG_DIR, exist_ok=True)

	logger.info("Loading training data for log decomposition")
	df = load_data(TRAIN_PATH, parse_dates=[TIME_COL], index_col=TIME_COL)
	if df is None:
		raise RuntimeError(f"Could not load: {TRAIN_PATH}")

	if TARGET_COL not in df.columns:
		raise ValueError(f"Missing required column: {TARGET_COL}")

	series = df[TARGET_COL].sort_index().interpolate(method="linear")
	if (series <= 0).any():
		raise ValueError("Log transform requires strictly positive values in the target series.")

	log_series = np.log(series)
	logger.info("Running additive decomposition on log(target)")
	decomp = seasonal_decompose(
		log_series,
		model="additive",
		period=SEASONAL_PERIOD,
	)

	fig, axes = plt.subplots(4, 1, figsize=(18, 12), sharex=True)
	components = ["observed", "trend", "seasonal", "resid"]
	titles = [
		"log(target) - observed",
		"log(target) - trend",
		"log(target) - seasonal",
		"log(target) - residual",
	]

	for ax, component, title in zip(axes, components, titles):
		ax.plot(getattr(decomp, component), linewidth=0.8)
		ax.set_title(title)

	plt.tight_layout()
	out_path = os.path.join(FIG_DIR, "decomposition_log_additive.png")
	fig.savefig(out_path, dpi=150)
	plt.close(fig)

	logger.info(f"Saved decomposition figure: {out_path}")
	logger.info(f"Log file: {log_file}")
	logger.info("Log decomposition complete")


if __name__ == "__main__":
	main()