import os
import numpy as np
import pandas as pd
from src.utils.config import logger


def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    try:
        # Do not force an index column here; different datasets may or may not
        # contain Loan_ID as a regular column.
        df = pd.read_csv(file_path, encoding='utf-8')
        logger.info("Data loaded successfully.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
