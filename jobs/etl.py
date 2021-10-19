import os
import pandas as pd
from cta.config import RAW_DATA_DIR
from cta.etl import fetch_daily_ridership_data, fetch_station_info


def main():
    raw_riders_df = fetch_daily_ridership_data()

    raw_station_info_df = fetch_station_info()

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    raw_riders_df.to_csv(os.path.join(RAW_DATA_DIR, 'raw_riders_df.csv'))
    raw_station_info_df.to_csv(os.path.join(RAW_DATA_DIR, 'raw_station_info_df.csv'))


if __name__ == '__main__':
    main()
