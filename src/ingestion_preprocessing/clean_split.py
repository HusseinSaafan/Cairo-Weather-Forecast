"""Clean and split Cairo weather data into train/validation/test sets.

Rules:
- Keep only `time` and `temperature_2m_mean (°C)`.
- Set `time` as the index.
- Test set is the last 7 days in the source data.
- Validation set is the 365 days immediately before the test window.
- Train set is all earlier rows.
"""

import os
import sys
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.utils.helpers import load_data
from src.utils.config import get_logger


TARGET_COL = "temperature_2m_mean (°C)"
TIME_COL = "time"
TEST_DAYS = 7
VALIDATION_DAYS = 365

DATA_PATH = os.path.join(PROJECT_ROOT, "database", "raw", "Cairo-Weather.csv")
DB_DIR = os.path.join(PROJECT_ROOT, "database")
TRAIN_PATH = os.path.join(DB_DIR, "train.csv")
VALIDATION_PATH = os.path.join(DB_DIR, "validation.csv")
TEST_PATH = os.path.join(DB_DIR, "test.csv")


def main() -> None:
	logger, log_file = get_logger(
		"src.ingestion_preprocessing.clean_split",
		log_filename=f"clean_split_{datetime.now():%Y%m%d_%H%M%S}.log",
	)

	os.makedirs(DB_DIR, exist_ok=True)

	logger.info("Starting clean/split pipeline")
	logger.info(f"Input file: {DATA_PATH}")

	df = load_data(DATA_PATH, parse_dates=[TIME_COL], dayfirst=False)
	if df is None:
		raise RuntimeError(f"Failed to load source data from {DATA_PATH}")

	required_cols = [TIME_COL, TARGET_COL]
	missing_cols = [col for col in required_cols if col not in df.columns]
	if missing_cols:
		raise ValueError(f"Missing required columns: {missing_cols}")

	# Keep only the two required columns and ensure deterministic ordering.
	clean_df = df[required_cols].copy()
	clean_df.sort_values(TIME_COL, inplace=True)

	# Set datetime index and remove rows that are not usable for modeling.
	clean_df.set_index(TIME_COL, inplace=True)
	clean_df.dropna(subset=[TARGET_COL], inplace=True)

	if clean_df.empty:
		raise ValueError("No usable rows after cleaning.")

	max_date = clean_df.index.max().normalize()
	test_start_date = max_date - pd.Timedelta(days=TEST_DAYS - 1)
	validation_start_date = test_start_date - pd.Timedelta(days=VALIDATION_DAYS)

	date_index = clean_df.index.normalize()
	test_df = clean_df.loc[date_index >= test_start_date].copy()
	validation_df = clean_df.loc[
		(date_index >= validation_start_date) & (date_index < test_start_date)
	].copy()
	train_df = clean_df.loc[date_index < validation_start_date].copy()

	if test_df.empty:
		raise ValueError("Test split is empty. Source data does not include the last 7-day window.")
	if validation_df.empty:
		raise ValueError("Validation split is empty. Source data does not include the required 365-day window before test.")
	if train_df.empty:
		raise ValueError("Train split is empty. No rows exist before the validation window.")

	train_df.to_csv(TRAIN_PATH, index=True, index_label=TIME_COL)
	validation_df.to_csv(VALIDATION_PATH, index=True, index_label=TIME_COL)
	test_df.to_csv(TEST_PATH, index=True, index_label=TIME_COL)

	logger.info(f"Clean shape      : {clean_df.shape}")
	logger.info(f"Train shape      : {train_df.shape}")
	logger.info(f"Validation shape : {validation_df.shape}")
	logger.info(f"Test shape       : {test_df.shape}")
	logger.info(f"Train date range : {train_df.index.min()} -> {train_df.index.max()}")
	logger.info(f"Validation range : {validation_df.index.min()} -> {validation_df.index.max()}")
	logger.info(f"Test date range  : {test_df.index.min()} -> {test_df.index.max()}")
	logger.info(f"Saved train data : {TRAIN_PATH}")
	logger.info(f"Saved valid data : {VALIDATION_PATH}")
	logger.info(f"Saved test data  : {TEST_PATH}")
	logger.info(f"Log file         : {log_file}")
	logger.info("Clean/split pipeline complete")


if __name__ == "__main__":
	main()
