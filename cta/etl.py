
import os
import pandas as pd
from cta.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

import urllib.parse as urlparse

API_ENDPOINT = 'https://data.cityofchicago.org/api/views'
DEFAULT_QUERY_PARAMS = {'accessType': 'DOWNLOAD', 'api_foundry': 'true'}


def generate_url(resource_id: str, query_params: dict = DEFAULT_QUERY_PARAMS):
    base_url = f'{API_ENDPOINT}/{resource_id}/rows.csv'
    parsed = list(urlparse.urlparse(base_url))
    parsed[4] = urlparse.urlencode(query_params)
    url = urlparse.urlunparse(parsed)
    return url


def fetch_dataframe_from_resource_id(resource_id):
    url = generate_url(resource_id)
    df = pd.read_csv(url)
    return df


def fetch_daily_ridership_data():
    resource_id = '5neh-572f'
    return fetch_dataframe_from_resource_id(resource_id)


def fetch_station_info():
    resource_id = '8pix-ypme'
    return fetch_dataframe_from_resource_id(resource_id)


def load_raw_station_info():
    path = os.path.join(RAW_DATA_DIR, 'raw_station_info_df.csv')
    return pd.read_csv(path, index_col=0)


def load_raw_riders_df():
    path = os.path.join(RAW_DATA_DIR, 'raw_riders_df.csv')
    df = pd.read_csv(path_index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    return pd.read_csv(path, index_col=0)


def load_processed_df():
    path = os.path.join(PROCESSED_DATA_DIR, 'daily_ridership.csv')
    df = pd.read_csv(path_index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    return pd.read_csv(path, index_col=0)
