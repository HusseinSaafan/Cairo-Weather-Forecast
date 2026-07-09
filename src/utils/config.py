import logging
import os
from datetime import datetime
from typing import Optional, Tuple

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TIMESTAMP_YMD = datetime.now().strftime("%Y-%m-%d")

os.makedirs(f"logs/{TIMESTAMP_YMD}/", exist_ok=True)


def get_logger(name: str = __name__, log_filename: Optional[str] = None) -> Tuple[logging.Logger, str]:
	"""Create a stream + file logger and return both logger and log file path."""
	if log_filename is None:
		log_filename = f"{TIMESTAMP}.log"

	log_file_path = f"logs/{TIMESTAMP_YMD}/{log_filename}"
	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)

	# Avoid duplicate handlers if get_logger is called multiple times.
	if not logger.handlers:
		formatter = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")

		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(formatter)

		file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
		file_handler.setFormatter(formatter)

		logger.addHandler(stream_handler)
		logger.addHandler(file_handler)

	return logger, log_file_path


# Backward-compatible module logger for existing imports.
logger, _ = get_logger(__name__)
