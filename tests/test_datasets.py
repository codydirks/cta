import os
import mock
import unittest
import pandas as pd

from cta.config import TRAIN_LINES
from cta.datasets import clean_station_info, process_station_line_info, clean_rider_info, get_ridership_dataset


class TestDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        this_dir = os.path.dirname(__file__)
        cls.sample_raw_station_info_df = pd.read_csv(os.path.join(this_dir, 'sample_raw_station_info_df.csv'), index_col=0)
        cls.sample_raw_riders_df = pd.read_csv(os.path.join(this_dir, 'sample_raw_riders_df.csv'), index_col=0)

    def test_clean_station_info(self):
        """
        Asserts that we correctly clean the input raw df column names.
        Does not test the one-off corrections made to typos in the dataset.
        """
        cleaned_df = clean_station_info(self.sample_raw_station_info_df)
        assert all([i in cleaned_df.columns for i in TRAIN_LINES])

    def test_process_station_line_info(self):
        """
        Asserts that we correctly pivoted the raw station info data to a long dataframe
        with 1 row per station per train-line.
        """
        raw_df_patch = mock.patch("cta.etl.load_raw_station_info")
        with raw_df_patch as raw_df_mock:
            raw_df_mock.return_value = self.sample_raw_station_info_df

        processed_df = process_station_line_info()
        assert set(processed_df.columns) == set(['station_name', 'station_descriptive_name', 'map_id', 'line', 'n_lines'])

    def test_clean_rider_info(self):
        cleaned_df = clean_rider_info(self.sample_raw_riders_df)
        truth_cols = ['station_id', 'stationname', 'date', 'daytype', 'rides', 'station_name', 'station_descriptive_name', 'map_id', 'line', 'n_lines']
        assert set(cleaned_df.columns) == set(truth_cols), f"{set(cleaned_df.columns) - set(truth_cols)}"

    def get_ridership_dataset(self):
        df = get_ridership_dataset()

        truth_cols = ['date', 'daytype', 'line', 'rides']
        assert set(df.columns) == set(truth_cols)

        extra_lines = set(df['lines']) - set(TRAIN_LINES)
        assert extra_lines == set(), f"Unexpected train lines found in dataset: {extra_lines}"
        assert df['rides'].min >= 0
