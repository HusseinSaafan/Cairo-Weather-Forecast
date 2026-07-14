"""Train Auto-ARIMA, save diagnostics/summary, and evaluate on test data."""

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import get_logger
from src.utils.helpers import load_data


TRAIN_PATH = os.path.join(PROJECT_ROOT, "database", "train.csv")
TEST2_PATH = os.path.join(PROJECT_ROOT, "database", "test2.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "database", "test.csv")
TIME_COL = "time"
TARGET_COL = "temperature_2m_mean (°C)"
FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "modeling")
EVAL_FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "model_eval")
SEASONAL_PERIOD = 30


def _resolve_test_path() -> str:
	if os.path.exists(TEST2_PATH):
		return TEST2_PATH
	if os.path.exists(TEST_PATH):
		return TEST_PATH
	raise FileNotFoundError(f"No test file found. Expected one of: {TEST2_PATH}, {TEST_PATH}")


def main() -> None:
	logger, log_file = get_logger(
		"src.modeling.auto_arima",
		log_filename=f"auto_arima_{datetime.now():%Y%m%d_%H%M%S}.log",
	)

	os.makedirs(FIG_DIR, exist_ok=True)
	os.makedirs(EVAL_FIG_DIR, exist_ok=True)

	resolved_test_path = _resolve_test_path()

	logger.info("Loading train/test data")
	train_df = load_data(TRAIN_PATH, parse_dates=[TIME_COL], index_col=TIME_COL)
	test_df = load_data(resolved_test_path, parse_dates=[TIME_COL], index_col=TIME_COL)
	if train_df is None or test_df is None:
		raise RuntimeError(f"Could not load train/test data from: {TRAIN_PATH}, {resolved_test_path}")

	if TARGET_COL not in train_df.columns:
		raise ValueError(f"Missing required column in train data: {TARGET_COL}")
	if TARGET_COL not in test_df.columns:
		raise ValueError(f"Missing required column in test data: {TARGET_COL}")

	train_series = train_df[TARGET_COL].sort_index().ffill().bfill()
	test_series = test_df[TARGET_COL].sort_index().ffill().bfill()
	logger.info(f"Training rows: {len(train_series)}")
	logger.info(f"Test rows: {len(test_series)}")

	logger.info(
		"Running auto_arima with seasonal=True, m=30, max_p=2, max_q=1, max_P=1, max_Q=1"
	)
	model = auto_arima(
		train_series,
		seasonal=True,
		m=SEASONAL_PERIOD,
		d=None,
		D=None,
		start_p=0,
		start_q=0,
		start_P=0,
		start_Q=0,
		max_p=2,
		max_q=1,
		max_P=1,
		max_Q=1,
		trace=True,
		error_action="raise",
		suppress_warnings=True,
		stepwise=False,
	)

	logger.info(f"Selected model order={model.order}, seasonal_order={model.seasonal_order}")

	summary_file = os.path.join(
		PROJECT_ROOT,
		"logs",
		datetime.now().strftime("%Y-%m-%d"),
		f"auto_arima_summary_{datetime.now():%Y%m%d_%H%M%S}.txt",
	)
	with open(summary_file, "w", encoding="utf-8") as f:
		f.write(str(model.summary()))
	logger.info(f"Saved model summary: {summary_file}")

	fig_diag = model.arima_res_.plot_diagnostics(figsize=(20, 12))
	diag_path = os.path.join(FIG_DIR, "auto_arima_diagnostics.png")
	fig_diag.savefig(diag_path, dpi=300)
	plt.close(fig_diag)

	forecast_values = model.predict(n_periods=len(test_series))
	forecast = pd.Series(forecast_values, index=test_series.index, name="AutoARIMAForecasting")

	eval_df = pd.DataFrame(
		{
			"Actual": test_series,
			"AutoARIMAForecasting": forecast,
		}
	)

	mse = mean_squared_error(eval_df["Actual"], eval_df["AutoARIMAForecasting"])
	mae = mean_absolute_error(eval_df["Actual"], eval_df["AutoARIMAForecasting"])
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
		eval_df["AutoARIMAForecasting"],
		label="AutoARIMAForecasting",
		color="red",
		linestyle="--",
	)
	ax.legend()
	ax.set_title("Actual vs Predicted (Auto-ARIMA)")
	ax.set_xlabel("Date")
	ax.set_ylabel(TARGET_COL)
	fig_eval.tight_layout()

	eval_plot_path = os.path.join(EVAL_FIG_DIR, "auto_arima_actual_vs_predicted.png")
	fig_eval.savefig(eval_plot_path, dpi=300)
	plt.close(fig_eval)

	logger.info(f"Saved diagnostics plot: {diag_path}")
	logger.info(f"Saved evaluation plot: {eval_plot_path}")
	logger.info(f"Log file: {log_file}")
	logger.info("Auto-ARIMA modeling + evaluation script complete")


if __name__ == "__main__":
	main()
