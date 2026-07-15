"""Statistical tests on log-transformed train temperature series.

1) Ljung-Box test at lag 365.
2) ADF checks on log series, ordinary differenced log series,
   and seasonal differenced log series.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import get_logger
from src.utils.helpers import load_data


TRAIN_PATH = os.path.join(PROJECT_ROOT, "database", "train.csv")
TIME_COL = "time"
TARGET_COL = "temperature_2m_mean (°C)"
SEASONAL_LAG = 365
ALPHA = 0.05


def run_adf(series: pd.Series, label: str, logger) -> None:
	result = adfuller(series.ffill().bfill())
	stat, pvalue, used_lag, nobs, crit_vals, _ = result

	logger.info(f"ADF - {label}")
	logger.info(f"  test_statistic : {stat:.6f}")
	logger.info(f"  p_value        : {pvalue:.6f}")
	logger.info(f"  used_lag       : {used_lag}")
	logger.info(f"  n_obs          : {nobs}")
	logger.info(
		"  critical_values: "
		f"1%={crit_vals['1%']:.6f}, 5%={crit_vals['5%']:.6f}, 10%={crit_vals['10%']:.6f}"
	)

	if pvalue < ALPHA:
		logger.info(f"  conclusion     : Stationary (reject H0 at alpha={ALPHA})")
	else:
		logger.info(f"  conclusion     : Non-stationary (fail to reject H0 at alpha={ALPHA})")


def main() -> None:
	logger, log_file = get_logger(
		"src.eda.test_log",
		log_filename=f"stationarity_tests_log_{datetime.now():%Y%m%d_%H%M%S}.log",
	)

	logger.info("Loading train data")
	df = load_data(TRAIN_PATH, parse_dates=[TIME_COL], index_col=TIME_COL)
	if df is None:
		raise RuntimeError(f"Could not load train data: {TRAIN_PATH}")

	if TARGET_COL not in df.columns:
		raise ValueError(f"Missing required column: {TARGET_COL}")

	series = df[TARGET_COL].sort_index().ffill().bfill()
	if (series <= 0).any():
		raise ValueError("Log transform requires strictly positive values in the target series.")

	log_series = np.log(series)

	logger.info("=" * 70)
	logger.info("STEP 1 - Ljung-Box on log series at lag 365")
	logger.info("=" * 70)

	lb = acorr_ljungbox(log_series, lags=[SEASONAL_LAG], return_df=True)
	lb_stat = float(lb["lb_stat"].iloc[0])
	lb_pvalue = float(lb["lb_pvalue"].iloc[0])

	logger.info(f"Ljung-Box lag     : {SEASONAL_LAG}")
	logger.info(f"lb_stat           : {lb_stat:.6f}")
	logger.info(f"lb_p_value        : {lb_pvalue:.6f}")
	if lb_pvalue < ALPHA:
		logger.info(f"conclusion        : Autocorrelation exists (reject H0 at alpha={ALPHA})")
	else:
		logger.info(f"conclusion        : No evidence of autocorrelation (fail to reject H0 at alpha={ALPHA})")

	logger.info("=" * 70)
	logger.info("STEP 2 - ADF stationarity tests on log scale")
	logger.info("=" * 70)

	run_adf(log_series, "log(original series)", logger)
	run_adf(log_series.diff(periods=1).dropna(), "log series ordinary difference (d=1)", logger)
	run_adf(
		log_series.diff(periods=SEASONAL_LAG).dropna(),
		f"log series seasonal difference (D=1, m={SEASONAL_LAG})",
		logger,
	)

	logger.info("=" * 70)
	logger.info("Tests complete")
	logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
	main()