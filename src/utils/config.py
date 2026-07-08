import logging
import os
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TIMESTAMP_YMD = datetime.now().strftime("%Y-%m-%d")

os.makedirs(f'logs/{TIMESTAMP_YMD}/', exist_ok=True)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.addHandler(logging.FileHandler(f'logs/{TIMESTAMP_YMD}/{TIMESTAMP}.log'))
