import os
import pandas as pd
from cta.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    SODAPY_APP_TOKEN, SODAPY_API_KEY, SODAPY_API_SECRET
)

from sodapy import Socrata


def fetch_dataframe_from_resource_id(resource_id, limit: int = None):
    client_kwargs = {
        'app_token': SODAPY_APP_TOKEN,
        'username': SODAPY_API_KEY,
        'password': SODAPY_API_SECRET
    }
    with Socrata("data.cityofchicago.org", **client_kwargs) as client:
        kwargs = {'content_type': 'csv'}
        if limit is not None:
            limit = int(limit)
            kwargs['limit'] = limit
            results = client.get(resource_id, **kwargs)
        else:
            kwargs['limit'] = 1_000_000
            results = list(client.get_all(resource_id, **kwargs))
    headers = results[0]
    data = [i for i in results[1:] if i != headers]
    df = pd.DataFrame.from_records(data, columns=headers)
    return df


def fetch_daily_ridership_data(limit: int = None):
    resource_id = "5neh-572f"
    return fetch_dataframe_from_resource_id(resource_id, limit)


def fetch_station_info(limit: int = None):
    resource_id = '8pix-ypme'
    return fetch_dataframe_from_resource_id(resource_id, limit)


def load_raw_station_info():
    path = os.path.join(RAW_DATA_DIR, 'raw_station_info_df.csv')
    return pd.read_csv(path, index_col=0)


def load_raw_riders_df():
    path = os.path.join(RAW_DATA_DIR, 'raw_riders_df.csv')
    df = pd.read_csv(path, index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_processed_df():
    path = os.path.join(PROCESSED_DATA_DIR, 'daily_ridership_df.csv')
    df = pd.read_csv(path, index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    return df
