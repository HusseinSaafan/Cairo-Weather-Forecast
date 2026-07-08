import os
import numpy as np
import pandas as pd
from src.utils.config import get_logger


logger, _ = get_logger(__name__)


def load_data(file_path, **read_csv_kwargs):
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path, encoding="utf-8", **read_csv_kwargs)
        logger.info("Data loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
