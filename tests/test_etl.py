import unittest

from cta.etl import fetch_daily_ridership_data, fetch_station_info


class TestEtl(unittest.TestCase):
    def test_fetch_daily_ridership_data(self):
        limits = [1, 10, 100]
        for limit in limits:
            df = fetch_daily_ridership_data(limit=limit)
            assert len(df) == limit
            expected_cols = ['station_id', 'stationname', 'date', 'daytype', 'rides']
            extra_cols = set(df.columns) - set(expected_cols)
            missing_cols = set(expected_cols) - set(df.columns)
            assert extra_cols == set(), f"Limit: {limit}, Extra columns: {extra_cols}"
            assert missing_cols == set(), f"Limit: {limit}, Extra columns: {missing_cols}"

    def test_fetch_station_info(self):
        limits = [1, 10, 100]
        for limit in limits:
            df = fetch_station_info(limit=limit)
            assert len(df) == limit

            expected_cols = [
                'stop_id', 'direction_id', 'stop_name', 'station_name', 'station_descriptive_name', 'map_id',
                'ada', 'red', 'blue', 'g', 'brn', 'p', 'pexp', 'y', 'pnk', 'o', 'location'
            ]
            extra_cols = set(df.columns) - set(expected_cols)
            missing_cols = set(expected_cols) - set(df.columns)
            assert extra_cols == set(), f"Limit: {limit}, Extra columns: {extra_cols}"
            assert missing_cols == set(), f"Limit: {limit}, Extra columns: {missing_cols}"
