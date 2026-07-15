"""Train SARIMA on log(target), invert forecasts, and evaluate on test2 data."""

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
FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "modeling")
EVAL_FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "model_eval")


def main() -> None:
	logger, log_file = get_logger(
		"src.modeling.sarima_log",
		log_filename=f"sarima_log_{datetime.now():%Y%m%d_%H%M%S}.log",
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

	train_log = np.log(train_series)

	logger.info("Building SARIMAX model on log(target) with order=(1,0,0), seasonal_order=(1,0,1,91)")
	sarima_model = SARIMAX(
		train_log,
		order=(1, 0, 0),
		seasonal_order=(1, 0, 1, 91),
		enforce_stationarity=False,
		enforce_invertibility=False,
	)

	sarima_fit = sarima_model.fit(disp=False)
	logger.info("Model fitting complete")

	summary_text = sarima_fit.summary().as_text()
	summary_file = os.path.join(
		PROJECT_ROOT,
		"logs",
		datetime.now().strftime("%Y-%m-%d"),
		f"sarima_log_summary_{datetime.now():%Y%m%d_%H%M%S}.txt",
	)
	with open(summary_file, "w", encoding="utf-8") as f:
		f.write(summary_text)

	logger.info(f"Saved model summary: {summary_file}")

	fig = sarima_fit.plot_diagnostics(figsize=(20, 12))
	diag_path = os.path.join(FIG_DIR, "sarima_log_diagnostics.png")
	fig.savefig(diag_path, dpi=300)
	plt.close(fig)

	forecast_log = sarima_fit.forecast(steps=len(test_series))
	forecast_log.index = test_series.index
	forecast = np.exp(forecast_log)

	eval_df = pd.DataFrame(
		{
			"Actual": test_series,
			"SARIMALogForecasting": forecast,
		}
	)

	mse = mean_squared_error(eval_df["Actual"], eval_df["SARIMALogForecasting"])
	mae = mean_absolute_error(eval_df["Actual"], eval_df["SARIMALogForecasting"])
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
		eval_df["SARIMALogForecasting"],
		label="SARIMALogForecasting",
		color="red",
		linestyle="--",
	)
	ax.legend()
	ax.set_title("Actual vs Predicted (SARIMA on log target)")
	ax.set_xlabel("Date")
	ax.set_ylabel(TARGET_COL)
	fig_eval.tight_layout()

	eval_plot_path = os.path.join(EVAL_FIG_DIR, "sarima_log_actual_vs_predicted.png")
	fig_eval.savefig(eval_plot_path, dpi=300)
	plt.close(fig_eval)

	logger.info(f"Saved diagnostics plot: {diag_path}")
	logger.info(f"Saved evaluation plot: {eval_plot_path}")
	logger.info(f"Log file: {log_file}")
	logger.info("SARIMA-log modeling + evaluation script complete")


if __name__ == "__main__":
	main()
