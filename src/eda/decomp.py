"""Seasonal decomposition (additive vs multiplicative) for train data."""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.utils.helpers import load_data
from src.utils.config import get_logger


TRAIN_PATH = os.path.join(PROJECT_ROOT, "database", "train.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "eda")
TIME_COL = "time"
TARGET_COL = "temperature_2m_mean (°C)"
SEASONAL_PERIOD = 365


def main() -> None:
	logger, log_file = get_logger(
		"src.eda.decomp",
		log_filename=f"decomp_{datetime.now():%Y%m%d_%H%M%S}.log",
	)

	os.makedirs(FIG_DIR, exist_ok=True)

	logger.info("Loading training data for decomposition")
	df = load_data(TRAIN_PATH, parse_dates=[TIME_COL], index_col=TIME_COL)
	if df is None:
		raise RuntimeError(f"Could not load: {TRAIN_PATH}")

	if TARGET_COL not in df.columns:
		raise ValueError(f"Missing required column: {TARGET_COL}")

	series = df[TARGET_COL].sort_index().interpolate(method="linear")

	logger.info("Running additive decomposition")
	add_decomposition = seasonal_decompose(
		series,
		model="additive",
		period=SEASONAL_PERIOD,
	)

	logger.info("Running multiplicative decomposition")
	multi_decomposition = seasonal_decompose(
		series,
		model="multiplicative",
		period=SEASONAL_PERIOD,
	)

	# Numeric comparison (no plots): CV of seasonal component.
	add_seasonal = add_decomposition.seasonal.dropna()
	multi_seasonal = multi_decomposition.seasonal.dropna()

	cv_add = np.std(add_seasonal) / np.mean(np.abs(add_seasonal))
	cv_multi = np.std(multi_seasonal) / np.mean(multi_seasonal)

	print(f"cv_add = {cv_add:.6f}")
	print(f"cv_multi = {cv_multi:.6f}")
	logger.info(f"cv_add   = {cv_add:.6f}")
	logger.info(f"cv_multi = {cv_multi:.6f}")

	# Persist a compact CV summary file in the logs folder for quick lookup.
	cv_log_path = os.path.join(os.path.dirname(log_file), "cv_results.log")
	with open(cv_log_path, "a", encoding="utf-8") as f:
		f.write(
			f"{datetime.now():%Y-%m-%d %H:%M:%S} "
			f"cv_add={cv_add:.6f} cv_multi={cv_multi:.6f}\n"
		)
	logger.info(f"Saved CV summary: {cv_log_path}")

	fig, axes = plt.subplots(4, 2, figsize=(30, 12), sharex=True)
	components = ["observed", "trend", "seasonal", "resid"]

	for i, component in enumerate(components):
		axes[i, 0].plot(getattr(add_decomposition, component), color="steelblue", linewidth=0.8)
		axes[i, 0].set_title(f"Additive - {component}")

	for i, component in enumerate(components):
		axes[i, 1].plot(getattr(multi_decomposition, component), color="darkorange", linewidth=0.8)
		axes[i, 1].set_title(f"Multiplicative - {component}")

	plt.tight_layout()
	out_path = os.path.join(FIG_DIR, "decomposition_additive_multiplicative.png")
	fig.savefig(out_path, dpi=150)
	plt.close(fig)

	logger.info(f"Saved decomposition figure: {out_path}")
	logger.info(f"Log file: {log_file}")
	logger.info("Decomposition complete")


if __name__ == "__main__":
	main()
