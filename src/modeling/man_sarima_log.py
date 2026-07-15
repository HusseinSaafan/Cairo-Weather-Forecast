"""Train ARIMA on seasonally-differenced log(target) and evaluate on original scale.

Manual seasonality approach in log scale:
- Seasonal difference log(train) with period=365.
- Fit plain ARIMA via SARIMAX(order=(1,0,0), trend='c') on differenced log series.
- Forecast on differenced log scale, invert differencing in log scale,
  then exponentiate back to the original scale.
"""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import get_logger
from src.utils.helpers import load_data


TRAIN_PATH = os.path.join(PROJECT_ROOT, "database", "train.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "database", "test2.csv")
TIME_COL = "time"
TARGET_COL = "temperature_2m_mean (°C)"
SEASONAL_PERIOD = 365
FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "modeling")
EVAL_FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "model_eval")


def main() -> None:
	logger, log_file = get_logger(
		"src.modeling.man_sarima_log",
		log_filename=f"man_sarima_log_{datetime.now():%Y%m%d_%H%M%S}.log",
	)

	os.makedirs(FIG_DIR, exist_ok=True)
	os.makedirs(EVAL_FIG_DIR, exist_ok=True)

	logger.info("Loading train/test data")
	train_df = load_data(TRAIN_PATH, parse_dates=[TIME_COL], index_col=TIME_COL)
	test_df = load_data(TEST_PATH, parse_dates=[TIME_COL], index_col=TIME_COL)
	if train_df is None or test_df is None:
		raise RuntimeError(f"Could not load train/test data from: {TRAIN_PATH}, {TEST_PATH}")

	if TARGET_COL not in train_df.columns:
		raise ValueError(f"Missing required column in train data: {TARGET_COL}")
	if TARGET_COL not in test_df.columns:
		raise ValueError(f"Missing required column in test data: {TARGET_COL}")

	train_series = train_df[TARGET_COL].sort_index().ffill().bfill()
	test_series = test_df[TARGET_COL].sort_index().ffill().bfill()
	logger.info(f"Training rows: {len(train_series)}")
	logger.info(f"Test rows: {len(test_series)}")

	if (train_series <= 0).any() or (test_series <= 0).any():
		raise ValueError("Log transform requires strictly positive target values in both train and test sets.")

	if len(train_series) <= SEASONAL_PERIOD:
		raise ValueError(
			f"Training data has {len(train_series)} rows; needs more than {SEASONAL_PERIOD} for seasonal differencing."
		)

	train_log = np.log(train_series)

	logger.info(f"Applying manual seasonal differencing on log(target) with period={SEASONAL_PERIOD}")
	diff_train_log = train_log.diff(periods=SEASONAL_PERIOD).dropna()

	logger.info("Building ARIMA model on seasonally-differenced log series with order=(1,0,0), trend='c'")
	arima_model = SARIMAX(
		diff_train_log,
		order=(1, 0, 0),
		trend="c",
		enforce_stationarity=False,
		enforce_invertibility=False,
	)

	arima_fit = arima_model.fit(disp=False)
	logger.info("Model fitting complete")

	summary_text = arima_fit.summary().as_text()
	summary_file = os.path.join(
		PROJECT_ROOT,
		"logs",
		datetime.now().strftime("%Y-%m-%d"),
		f"man_sarima_log_summary_{datetime.now():%Y%m%d_%H%M%S}.txt",
	)
	with open(summary_file, "w", encoding="utf-8") as f:
		f.write(summary_text)
	logger.info(f"Saved model summary: {summary_file}")

	fig_diag = arima_fit.plot_diagnostics(figsize=(20, 12))
	diag_path = os.path.join(FIG_DIR, "man_sarima_log_diagnostics.png")
	fig_diag.savefig(diag_path, dpi=300)
	plt.close(fig_diag)

	forecast_steps = len(test_series)
	logger.info(f"Forecasting {forecast_steps} step(s) on differenced log scale")
	diff_forecast_log = arima_fit.forecast(steps=forecast_steps)

	# Invert seasonal differencing in log scale, then exponentiate back.
	last_season_log_values = train_log.iloc[-SEASONAL_PERIOD:].values
	n_repeats = int(np.ceil(forecast_steps / SEASONAL_PERIOD))
	base_log_values = np.tile(last_season_log_values, n_repeats)[:forecast_steps]
	forecast_log_values = diff_forecast_log.values + base_log_values
	forecast_values = np.exp(forecast_log_values)

	forecast = pd.Series(forecast_values, index=test_series.index, name="ManualSeasonalDiffARIMALog")

	eval_df = pd.DataFrame(
		{
			"Actual": test_series,
			"ManualSeasonalDiffARIMALog": forecast,
		}
	)

	mse = mean_squared_error(eval_df["Actual"], eval_df["ManualSeasonalDiffARIMALog"])
	mae = mean_absolute_error(eval_df["Actual"], eval_df["ManualSeasonalDiffARIMALog"])
	rmse = np.sqrt(mse)

	logger.info("Evaluation metrics")
	logger.info(f"MSE  : {mse:.6f}")
	logger.info(f"MAE  : {mae:.6f}")
	logger.info(f"RMSE : {rmse:.6f}")

	print(f"MSE  = {mse:.6f}")
	print(f"MAE  = {mae:.6f}")
	print(f"RMSE = {rmse:.6f}")

	fig_eval, ax = plt.subplots(figsize=(10, 6))
	ax.plot(eval_df.index, eval_df["Actual"], label="Actual", color="blue")
	ax.plot(
		eval_df.index,
		eval_df["ManualSeasonalDiffARIMALog"],
		label="ManualSeasonalDiffARIMALog",
		color="red",
		linestyle="--",
	)
	ax.legend()
	ax.set_title("Actual vs Predicted (Manual Seasonal Diff + ARIMA on log target)")
	ax.set_xlabel("Date")
	ax.set_ylabel(TARGET_COL)
	fig_eval.tight_layout()

	eval_plot_path = os.path.join(EVAL_FIG_DIR, "man_sarima_log_actual_vs_predicted.png")
	fig_eval.savefig(eval_plot_path, dpi=300)
	plt.close(fig_eval)

	logger.info(f"Saved diagnostics plot: {diag_path}")
	logger.info(f"Saved evaluation plot: {eval_plot_path}")
	logger.info(f"Log file: {log_file}")
	logger.info("Manual seasonal differencing ARIMA-log script complete")


if __name__ == "__main__":
	main()
