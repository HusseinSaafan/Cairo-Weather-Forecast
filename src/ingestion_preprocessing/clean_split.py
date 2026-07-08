"""Clean and split Cairo weather data into train/test sets.

Rules:
- Keep only `time` and `temperature_2m_mean (°C)`.
- Set `time` as the index.
- Test set is the last 30 records; train is all earlier records.
"""

import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from src.utils.helpers import load_data
from src.utils.config import get_logger


TARGET_COL = "temperature_2m_mean (°C)"
TIME_COL = "time"
TEST_SIZE = 30

DATA_PATH = os.path.join(PROJECT_ROOT, "database", "raw", "Cairo-Weather.csv")
DB_DIR = os.path.join(PROJECT_ROOT, "database")
TRAIN_PATH = os.path.join(DB_DIR, "train.csv")
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

	if len(clean_df) <= TEST_SIZE:
		raise ValueError(
			f"Not enough records to split: {len(clean_df)} total, needs more than {TEST_SIZE}."
		)

	train_df = clean_df.iloc[:-TEST_SIZE].copy()
	test_df = clean_df.iloc[-TEST_SIZE:].copy()

	train_df.to_csv(TRAIN_PATH, index=True, index_label=TIME_COL)
	test_df.to_csv(TEST_PATH, index=True, index_label=TIME_COL)

	logger.info(f"Clean shape      : {clean_df.shape}")
	logger.info(f"Train shape      : {train_df.shape}")
	logger.info(f"Test shape       : {test_df.shape}")
	logger.info(f"Train date range : {train_df.index.min()} -> {train_df.index.max()}")
	logger.info(f"Test date range  : {test_df.index.min()} -> {test_df.index.max()}")
	logger.info(f"Saved train data : {TRAIN_PATH}")
	logger.info(f"Saved test data  : {TEST_PATH}")
	logger.info(f"Log file         : {log_file}")
	logger.info("Clean/split pipeline complete")


if __name__ == "__main__":
	main()
