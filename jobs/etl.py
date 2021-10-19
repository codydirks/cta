import os
import pandas as pd
from cta.config import RAW_DATA_DIR
from cta.etl import fetch_daily_ridership_data, fetch_station_info


def main():
    raw_riders_df = fetch_daily_ridership_data()

    raw_station_info_df = fetch_station_info()

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    pd.to_csv(raw_riders_df, os.path.join(RAW_DATA_DIR, 'raw_riders_df.csv'))
    pd.to_csv(raw_station_info_df, os.path.join(RAW_DATA_DIR, 'raw_station_info_df.csv'))


if __name__ == '__main__':
    main()
