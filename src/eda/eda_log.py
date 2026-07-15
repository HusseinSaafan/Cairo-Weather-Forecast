"""Plot ACF and PACF for log-transformed training temperature series."""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import get_logger
from src.utils.helpers import load_data


TRAIN_PATH = os.path.join(PROJECT_ROOT, "database", "train.csv")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "eda")
TIME_COL = "time"
TARGET_COL = "temperature_2m_mean (°C)"
MAX_LAGS = 400


def main() -> None:
	logger, log_file = get_logger(
		"src.eda.eda_log",
		log_filename=f"acf_pacf_log_{datetime.now():%Y%m%d_%H%M%S}.log",
	)

	os.makedirs(FIG_DIR, exist_ok=True)

	logger.info("Loading train data for log ACF/PACF")
	df = load_data(TRAIN_PATH, parse_dates=[TIME_COL], index_col=TIME_COL)
	if df is None:
		raise RuntimeError(f"Could not load: {TRAIN_PATH}")

	if TARGET_COL not in df.columns:
		raise ValueError(f"Missing required column: {TARGET_COL}")

	series = df[TARGET_COL].sort_index().ffill().bfill()
	if (series <= 0).any():
		raise ValueError("Log transform requires strictly positive values in the target series.")

	log_series = np.log(series)
	lags = min(MAX_LAGS, len(log_series) - 1)

	logger.info(f"Series length: {len(log_series)}")
	logger.info(f"Using lags  : {lags}")

	fig_acf, ax_acf = plt.subplots(figsize=(12, 10))
	plot_acf(log_series, lags=lags, ax=ax_acf, zero=False)
	ax_acf.set_title(f"ACF - log({TARGET_COL})")
	fig_acf.tight_layout()
	acf_path = os.path.join(FIG_DIR, "ACF_log.png")
	fig_acf.savefig(acf_path, dpi=300)
	plt.close(fig_acf)
	logger.info(f"Saved ACF plot : {acf_path}")

	fig_pacf, ax_pacf = plt.subplots(figsize=(12, 10))
	plot_pacf(log_series, lags=lags, method="ywm", ax=ax_pacf, zero=False)
	ax_pacf.set_title(f"PACF - log({TARGET_COL})")
	fig_pacf.tight_layout()
	pacf_path = os.path.join(FIG_DIR, "PACF_log.png")
	fig_pacf.savefig(pacf_path, dpi=300)
	plt.close(fig_pacf)
	logger.info(f"Saved PACF plot: {pacf_path}")

	logger.info(f"Log file: {log_file}")
	logger.info("Log ACF/PACF plotting complete")


if __name__ == "__main__":
	main()