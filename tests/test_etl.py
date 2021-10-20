import unittest

from cta.etl import fetch_daily_ridership_data, fetch_station_info


class TestEtl(unittest.TestCase):
    def test_fetch_daily_ridership_data(self):
        df = fetch_daily_ridership_data(limit=1)
        assert len(df) == 1

        expected_cols = ['station_id', 'stationname', 'date', 'daytype', 'rides']
        extra_cols = set(df.columns) - set(expected_cols)
        missing_cols = set(expected_cols) - set(df.columns)
        assert extra_cols == set(), f"Extra columns: {extra_cols}"
        assert missing_cols == set(), f"Extra columns: {missing_cols}"

    def test_fetch_station_info(self):
        df = fetch_station_info(limit=1)
        assert len(df) == 1

        expected_cols = [
            'stop_id', 'direction_id', 'stop_name', 'station_name', 'station_descriptive_name', 'map_id',
            'ada', 'red', 'blue', 'g', 'brn', 'p', 'pexp', 'y', 'pnk', 'o', 'location'
        ]
        extra_cols = set(df.columns) - set(expected_cols)
        missing_cols = set(expected_cols) - set(df.columns)
        assert extra_cols == set(), f"Extra columns: {extra_cols}"
        assert missing_cols == set(), f"Extra columns: {missing_cols}"
