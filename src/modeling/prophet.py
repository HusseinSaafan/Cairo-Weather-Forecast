"""Train Prophet model, forecast on test data, and save summary/plots/metrics."""

import os
import json
import pickle
import sys
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR in sys.path:
	sys.path.remove(CURRENT_DIR)

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import get_logger
from src.utils.helpers import load_data


TRAIN_PATH = os.path.join(PROJECT_ROOT, "database", "train.csv")
TEST_PATH = os.path.join(PROJECT_ROOT, "database", "test.csv")
TIME_COL = "time"
TARGET_COL = "temperature_2m_mean (°C)"
FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "modeling")
EVAL_FIG_DIR = os.path.join(PROJECT_ROOT, "figures", "model_eval")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_ARTIFACT_PATH = os.path.join(ARTIFACTS_DIR, "prophet.pkl")
METRICS_ARTIFACT_PATH = os.path.join(ARTIFACTS_DIR, "prophet_metrics.json")

# Cross-validation setup for Prophet diagnostics.
CV_INITIAL = "730 days"
CV_PERIOD = "180 days"
CV_HORIZON = "365 days"

# Keep this grid intentionally small to avoid excessive runtime.
PROPHET_PARAM_GRID = {
	"changepoint_prior_scale": [0.01, 0.05, 0.1],
	"seasonality_prior_scale": [1.0, 10.0],
	"seasonality_mode": ["additive", "multiplicative"],
	"weekly_seasonality": [True, False],
	"quarterly_fourier_order": [6],
}


def _build_model(params: Optional[Dict[str, Any]] = None) -> Prophet:
	"""Create a Prophet model tuned for yearly weather seasonality."""
	params = params or {}
	changepoint_prior_scale = float(params.get("changepoint_prior_scale", 0.05))
	seasonality_prior_scale = float(params.get("seasonality_prior_scale", 10.0))
	seasonality_mode = str(params.get("seasonality_mode", "additive"))
	weekly_seasonality = bool(params.get("weekly_seasonality", True))
	quarterly_fourier_order = int(params.get("quarterly_fourier_order", 8))

	model = Prophet(
		growth="linear",
		yearly_seasonality=True,
		weekly_seasonality=weekly_seasonality,
		daily_seasonality=False,
		seasonality_mode=seasonality_mode,
		changepoint_prior_scale=changepoint_prior_scale,
		seasonality_prior_scale=seasonality_prior_scale,
	)
	# Extra quarterly-like smooth seasonal component for climate transition effects.
	model.add_seasonality(name="quarterly", period=91.25, fourier_order=quarterly_fourier_order)
	return model


def _grid_search_with_cv(prophet_train: pd.DataFrame, logger) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
	"""Run a small grid search scored by Prophet cross-validation RMSE."""
	grid_keys = list(PROPHET_PARAM_GRID.keys())
	param_sets = [dict(zip(grid_keys, values)) for values in product(*(PROPHET_PARAM_GRID[k] for k in grid_keys))]
	results: List[Dict[str, Any]] = []

	logger.info(f"Starting Prophet CV grid search with {len(param_sets)} combinations")
	for idx, params in enumerate(param_sets, start=1):
		logger.info(f"[{idx}/{len(param_sets)}] Evaluating params: {params}")
		try:
			candidate = _build_model(params)
			candidate.fit(prophet_train)
			cv_df = cross_validation(
				candidate,
				initial=CV_INITIAL,
				period=CV_PERIOD,
				horizon=CV_HORIZON,
				parallel=None,
			)
			metrics_df = performance_metrics(cv_df, rolling_window=1)

			rmse = float(metrics_df["rmse"].iloc[-1])
			mae = float(metrics_df["mae"].iloc[-1])
			mape = float(metrics_df["mape"].iloc[-1]) if "mape" in metrics_df.columns else np.nan

			result = {
				"rmse": rmse,
				"mae": mae,
				"mape": mape,
				**params,
			}
			results.append(result)
			logger.info(f"CV result -> RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.6f}")
		except Exception as exc:
			logger.warning(f"Skipping params due to CV failure: {params}. Error: {exc}")

	if not results:
		raise RuntimeError("All Prophet grid-search combinations failed during cross-validation")

	best_result = min(results, key=lambda x: x["rmse"])
	logger.info(f"Best CV params: {best_result}")
	return best_result, results


def main() -> Dict[str, Any]:
	logger, log_file = get_logger(
		"src.modeling.prophet",
		log_filename=f"prophet_{datetime.now():%Y%m%d_%H%M%S}.log",
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

	prophet_train = train_series.reset_index().rename(columns={TIME_COL: "ds", TARGET_COL: "y"})

	log_day_dir = os.path.join(PROJECT_ROOT, "logs", datetime.now().strftime("%Y-%m-%d"))
	os.makedirs(log_day_dir, exist_ok=True)

	best_result, grid_results = _grid_search_with_cv(prophet_train, logger)
	best_params = {
		"changepoint_prior_scale": best_result["changepoint_prior_scale"],
		"seasonality_prior_scale": best_result["seasonality_prior_scale"],
		"seasonality_mode": best_result["seasonality_mode"],
		"weekly_seasonality": best_result["weekly_seasonality"],
		"quarterly_fourier_order": best_result["quarterly_fourier_order"],
	}

	logger.info("Building final Prophet model with best CV hyperparameters")
	model = _build_model(best_params)
	model.fit(prophet_train)
	logger.info("Model fitting complete")
	os.makedirs(ARTIFACTS_DIR, exist_ok=True)
	with open(MODEL_ARTIFACT_PATH, "wb") as f:
		pickle.dump(model, f)
	logger.info(f"Saved model artifact: {MODEL_ARTIFACT_PATH}")

	grid_results_df = pd.DataFrame(grid_results).sort_values("rmse", ascending=True)
	grid_results_file = os.path.join(
		log_day_dir,
		f"prophet_grid_search_{datetime.now():%Y%m%d_%H%M%S}.csv",
	)
	grid_results_df.to_csv(grid_results_file, index=False)
	logger.info(f"Saved grid-search table: {grid_results_file}")

	summary_file = os.path.join(
		log_day_dir,
		f"prophet_summary_{datetime.now():%Y%m%d_%H%M%S}.txt",
	)
	with open(summary_file, "w", encoding="utf-8") as f:
		f.write("Prophet model summary\n")
		f.write(f"Train path: {TRAIN_PATH}\n")
		f.write(f"Test path : {TEST_PATH}\n")
		f.write(f"Rows (train/test): {len(train_series)}/{len(test_series)}\n")
		f.write("\nModel params:\n")
		f.write(f"growth={model.growth}\n")
		f.write(f"seasonality_mode={model.seasonality_mode}\n")
		f.write(f"yearly_seasonality={model.yearly_seasonality}\n")
		f.write(f"weekly_seasonality={model.weekly_seasonality}\n")
		f.write(f"daily_seasonality={model.daily_seasonality}\n")
		f.write(f"n_changepoints={model.n_changepoints}\n")
		f.write(f"changepoint_prior_scale={model.changepoint_prior_scale}\n")
		f.write(f"seasonality_prior_scale={model.seasonality_prior_scale}\n")
		f.write("\nCross-validation setup:\n")
		f.write(f"initial={CV_INITIAL}\n")
		f.write(f"period={CV_PERIOD}\n")
		f.write(f"horizon={CV_HORIZON}\n")
		f.write("\nBest CV result:\n")
		f.write(f"best_rmse={best_result['rmse']:.6f}\n")
		f.write(f"best_mae={best_result['mae']:.6f}\n")
		f.write(f"best_mape={best_result['mape']:.6f}\n")
		f.write(f"best_params={best_params}\n")
		f.write(f"grid_results_file={grid_results_file}\n")
		f.write("\nExtra seasonality:\n")
		for seasonality_name, seasonality_cfg in model.seasonalities.items():
			f.write(f"- {seasonality_name}: {seasonality_cfg}\n")
	logger.info(f"Saved model summary: {summary_file}")

	future = pd.DataFrame({"ds": test_series.index})
	forecast_df = model.predict(future)
	forecast = pd.Series(forecast_df["yhat"].values, index=test_series.index, name="ProphetForecasting")

	eval_df = pd.DataFrame(
		{
			"Actual": test_series,
			"ProphetForecasting": forecast,
		}
	)

	mse = mean_squared_error(eval_df["Actual"], eval_df["ProphetForecasting"])
	mae = mean_absolute_error(eval_df["Actual"], eval_df["ProphetForecasting"])
	rmse = np.sqrt(mse)

	logger.info("Evaluation metrics")
	logger.info(f"MSE  : {mse:.6f}")
	logger.info(f"MAE  : {mae:.6f}")
	logger.info(f"RMSE : {rmse:.6f}")

	print(f"MSE  = {mse:.6f}")
	print(f"MAE  = {mae:.6f}")
	print(f"RMSE = {rmse:.6f}")

	fig_forecast = model.plot(forecast_df)
	ax_forecast = fig_forecast.axes[0]
	ax_forecast.set_title("Prophet Forecast on Test2 Horizon")
	ax_forecast.set_xlabel("Date")
	ax_forecast.set_ylabel(TARGET_COL)
	fig_forecast.tight_layout()
	forecast_plot_path = os.path.join(FIG_DIR, "prophet_forecast.png")
	fig_forecast.savefig(forecast_plot_path, dpi=300)
	plt.close(fig_forecast)

	fig_components = model.plot_components(forecast_df)
	components_plot_path = os.path.join(FIG_DIR, "prophet_components.png")
	fig_components.savefig(components_plot_path, dpi=300)
	plt.close(fig_components)

	fig_eval, ax = plt.subplots(figsize=(10, 6))
	ax.plot(eval_df.index, eval_df["Actual"], label="Actual", color="blue")
	ax.plot(
		eval_df.index,
		eval_df["ProphetForecasting"],
		label="ProphetForecasting",
		color="red",
		linestyle="--",
	)
	ax.legend()
	ax.set_title("Actual vs Predicted (Prophet)")
	ax.set_xlabel("Date")
	ax.set_ylabel(TARGET_COL)
	fig_eval.tight_layout()

	eval_plot_path = os.path.join(EVAL_FIG_DIR, "prophet_actual_vs_predicted.png")
	fig_eval.savefig(eval_plot_path, dpi=300)
	plt.close(fig_eval)

	logger.info(f"Saved forecast plot: {forecast_plot_path}")
	logger.info(f"Saved components plot: {components_plot_path}")
	logger.info(f"Saved evaluation plot: {eval_plot_path}")
	logger.info(f"Log file: {log_file}")
	logger.info("Prophet modeling + evaluation script complete")

	metrics: Dict[str, Any] = {
		"model": "prophet",
		"mse": float(mse),
		"mae": float(mae),
		"rmse": float(rmse),
		"artifact_path": MODEL_ARTIFACT_PATH,
		"metrics_path": METRICS_ARTIFACT_PATH,
	}
	with open(METRICS_ARTIFACT_PATH, "w", encoding="utf-8") as f:
		json.dump(metrics, f, indent=2)
	logger.info(f"Saved metrics artifact: {METRICS_ARTIFACT_PATH}")

	return metrics


if __name__ == "__main__":
	main()
