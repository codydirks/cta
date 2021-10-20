import os
from dotenv import load_dotenv
import pandas as pd

from pathlib import Path

import cta


PROJECT_DIR = Path(cta.__file__).parent.parent
load_dotenv(os.path.join(PROJECT_DIR, '.env'))

API_ENDPOINT = 'data.cityofchicago.org'
SODAPY_APP_TOKEN = os.environ['SODAPY_APP_TOKEN']
SODAPY_SECRET_TOKEN = os.environ['SODAPY_SECRET_TOKEN']
SODAPY_API_KEY = os.environ['SODAPY_API_KEY']
SODAPY_API_SECRET = os.environ['SODAPY_API_SECRET']

DATA_DIR = os.path.join(PROJECT_DIR, 'data')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
ARTIFACTS_DIR = os.path.join(DATA_DIR, 'artifacts')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')

SPLIT_DT = pd.Timestamp('2019-01-01')

TRAIN_LINES = ['red', 'blue', 'green', 'brown', 'purple', 'purple_express', 'yellow', 'pink', 'orange']
