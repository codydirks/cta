import unittest
from unittest.mock import patch

from cta.etl import (
    fetch_daily_ridership_data, fetch_station_info,
    load_raw_station_info, load_raw_riders_df, load_processed_df
)

expected_ridership_cols = ['station_id', 'stationname', 'date', 'daytype', 'rides']

expected_station_info_cols = [
    'stop_id',
    'direction_id',
    'stop_name',
    'station_name',
    'station_descriptive_name',
    'map_id',
    'ada',
    'red', 'blue', 'g', 'brn', 'p', 'pexp', 'y', 'pnk', 'o',
    'location'
]

expected_processed_dataset_cols = ['dataset', 'date', 'rides', 'line', 'daytype']


class TestEtl(unittest.TestCase):
    def test_fetch_daily_ridership_data(self):
        limits = [1, 10, 100]
        for limit in limits:
            df = fetch_daily_ridership_data(limit=limit)
            assert len(df) == limit
            extra_cols = set(df.columns) - set(expected_ridership_cols)
            missing_cols = set(expected_ridership_cols) - set(df.columns)
            assert extra_cols == set(), f"Limit: {limit}, Extra columns: {extra_cols}"
            assert missing_cols == set(), f"Limit: {limit}, Missing columns: {missing_cols}"

    def test_fetch_station_info(self):
        limits = [1, 10, 100]
        for limit in limits:
            df = fetch_station_info(limit=limit)
            assert len(df) == limit

            extra_cols = set(df.columns) - set(expected_station_info_cols)
            missing_cols = set(expected_station_info_cols) - set(df.columns)
            assert extra_cols == set(), f"Limit: {limit}, Extra columns: {extra_cols}"
            assert missing_cols == set(), f"Limit: {limit}, Missing columns: {missing_cols}"

    def path_patcher(*join_args):
        this_dir = __file__
        fl = join_args[-1]
        return this_dir.replace("test_etl.py", f"sample_{fl}")

    @patch("cta.etl.os.path.join")
    def test_load_raw_station_info(self, mock_path_join):
        mock_path_join.side_effect = self.path_patcher
        df = load_raw_station_info()

        cols = df.columns.str.lower()
        extra_cols = set(cols) - set(expected_station_info_cols)
        missing_cols = set(expected_station_info_cols) - set(cols)

        assert extra_cols == set(), f"Extra columns: {extra_cols}"
        assert missing_cols == set(), f"Missing columns: {missing_cols}"

    @patch("cta.etl.os.path.join")
    def test_load_raw_riders_df(self, mock_path_join):
        mock_path_join.side_effect = self.path_patcher
        df = load_raw_riders_df()

        cols = df.columns.str.lower()
        extra_cols = set(cols) - set(expected_ridership_cols)
        missing_cols = set(expected_ridership_cols) - set(cols)

        assert extra_cols == set(), f"Extra columns: {extra_cols}"
        assert missing_cols == set(), f"Missing columns: {missing_cols}"

    @patch("cta.etl.os.path.join")
    def test_load_processed_df(self, mock_path_join):
        mock_path_join.side_effect = self.path_patcher
        df = load_processed_df()

        cols = df.columns.str.lower()
        extra_cols = set(cols) - set(expected_processed_dataset_cols)
        missing_cols = set(expected_processed_dataset_cols) - set(cols)

        assert extra_cols == set(), f"Extra columns: {extra_cols}"
        assert missing_cols == set(), f"Missing columns: {missing_cols}"
