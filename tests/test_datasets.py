import os
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from cta.config import TRAIN_LINES
from cta.datasets import (
    clean_station_info, process_station_line_info,
    clean_rider_info, get_ridership_dataset,
    CTADataset
)

this_dir = os.path.dirname(__file__)
sample_raw_station_info_df = pd.read_csv(os.path.join(this_dir, 'sample_raw_station_info_df.csv'), index_col=0)
sample_raw_riders_df = pd.read_csv(os.path.join(this_dir, 'sample_raw_riders_df.csv'), index_col=0)


@patch("cta.datasets.load_raw_riders_df", MagicMock(return_value=sample_raw_riders_df))
@patch("cta.datasets.load_raw_station_info", MagicMock(return_value=sample_raw_station_info_df))
class TestDatasets(unittest.TestCase):
    def test_clean_station_info(self):
        """
        Asserts that we correctly clean the input raw df column names.
        Does not test the one-off corrections made to typos in the dataset.
        """
        cleaned_df = clean_station_info(sample_raw_station_info_df)
        assert all([i in cleaned_df.columns for i in TRAIN_LINES])

    def test_process_station_line_info(self):
        """
        Asserts that we correctly pivoted the raw station info data to a long dataframe
        with 1 row per station per train-line.
        """
        processed_df = process_station_line_info()

        assert set(processed_df.columns) == set(['station_name', 'station_descriptive_name', 'map_id', 'line', 'n_lines'])

    def test_clean_rider_info(self):
        cleaned_df = clean_rider_info(sample_raw_riders_df)
        truth_cols = ['station_id', 'stationname', 'date', 'daytype', 'rides', 'station_name', 'station_descriptive_name', 'map_id', 'line', 'n_lines']
        assert set(cleaned_df.columns) == set(truth_cols), f"{set(cleaned_df.columns) - set(truth_cols)}"

    def test_get_ridership_dataset(self):
        df = get_ridership_dataset()

        truth_cols = ['date', 'daytype', 'line', 'rides']
        assert set(df.columns) == set(truth_cols)

        extra_lines = set(df['line']) - set(TRAIN_LINES)
        assert extra_lines == set(), f"Unexpected train lines found in dataset: {extra_lines}"
        assert df['rides'].min() >= 0

    def test_dataset(self):
        df = get_ridership_dataset()

        ds = CTADataset.from_dataframe(df)

        assert ds.tensors[0].shape == (len(TRAIN_LINES), 365, 1)
