import os
from dotenv import load_dotenv
import pandas as pd

from pathlib import Path

import cta


PROJECT_DIR = Path(cta.__file__).parent.parent
load_dotenv(os.path.join(PROJECT_DIR, '.env'))

DATA_DIR = os.path.join(PROJECT_DIR, 'data')

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
ARTIFACTS_DIR = os.path.join(DATA_DIR, 'artifacts')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')

SPLIT_DT = pd.Timestamp('2019-01-01')

TRAIN_LINES = ['red', 'blue', 'green', 'brown', 'purple', 'purple_express', 'yellow', 'pink', 'orange']
