import pandas as pd

from cta.etl import load_raw_station_info, load_raw_riders_df
from cta.config import TRAIN_LINES

from torchcast.utils.data import TimeSeriesDataset


def clean_station_info(df):
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Clean up column names
    old_names = ['red', 'blue', 'g', 'brn', 'p', 'pexp', 'y', 'pnk', 'o']
    mapper = dict(zip(old_names, TRAIN_LINES))
    df = df.rename(columns=mapper)

    # Fix a couple of one-off typos in the data
    df['station_descriptive_name'] = df['station_descriptive_name'].str.replace('  ', ' ')

    df['station_name'] = df['station_name'].str.replace('Ran Ryan', 'Dan Ryan')
    df['station_descriptive_name'] = df['station_descriptive_name'].str.replace('Ran Ryan', 'Dan Ryan')

    return df


def process_station_line_info():
    raw_station_info_df = load_raw_station_info()
    station_info_df = clean_station_info(raw_station_info_df)

    station_info_cols = ['station_name', 'station_descriptive_name', 'map_id']

    # All normal Purple Line stations are also marked express, let's instead make them distinct
    station_info_df['purple_express'] = (station_info_df['purple_express']) & (~station_info_df['purple'])

    # Reduce df to relevant columns
    station_info_df = station_info_df[station_info_cols + TRAIN_LINES]
    station_info_df = station_info_df.drop_duplicates()  # Each station has a separate entry for inbound/outbound, we only care about the actual station

    # For stations in the loop, different directions have different trains available, but we don't really
    # care about that for raw ridership data, so do a quick grouping and aggregation so we end up with
    # 1 row per station, with all available train lines for that station marked.
    aggs = dict(zip(TRAIN_LINES, [pd.NamedAgg(column=t, aggfunc=lambda x: any(x)) for t in TRAIN_LINES]))
    station_lines = station_info_df.sort_values('map_id').groupby('map_id').agg(**aggs).reset_index()

    # Melt the above result into a long dataframe, and join back to station info
    # data to get 1 row per station per train line

    stations = station_info_df[station_info_cols].drop_duplicates()
    stations = stations.merge(station_lines, on='map_id')

    stations = stations.melt(id_vars=station_info_cols, value_vars=TRAIN_LINES, var_name='line')
    stations = stations[stations['value']].drop('value', axis=1)

    # Also add in # of train lines that touch this station - will use this later to split
    # ridership at junction stations and attribute rides to individual train lines
    n_lines = stations.groupby(station_info_cols).agg(n_lines=pd.NamedAgg(column='map_id', aggfunc='count')).reset_index()
    stations = stations.merge(n_lines, on=station_info_cols, how='left')

    return stations


def clean_rider_info(df):
    assert [i in df.columns for i in ('date', 'daytype', 'station_id')]
    df = df.copy()

    stations = process_station_line_info()

    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

    df = df.merge(stations, left_on='station_id', right_on='map_id')

    # Purple Express only runs on weekdays, so drop non-weekday entries for that line, and
    # adjust the # of possible lines for the relevant station on those non-weekdays to be n-1
    is_p_exp = df['line'] == 'purple_express'
    is_non_weekday = df['daytype'] != 'W'
    p_exp_stations = df[is_p_exp]['station_id'].unique()

    needs_correction = df['station_id'].isin(p_exp_stations) & is_non_weekday
    df.loc[needs_correction, 'n_lines'] = df.loc[needs_correction, 'n_lines'] - 1

    df = df[~((is_non_weekday) & (is_p_exp))]

    return df


def get_ridership_dataset():
    df = load_raw_riders_df()

    df = clean_rider_info(df)

    df['line_corrected_ridership'] = df['rides'] / df['n_lines']

    df = df.groupby(['date', 'daytype', 'line'])['line_corrected_ridership'].sum().reset_index()
    df = df.rename({'line_corrected_ridership': 'rides'}, axis=1)

    return df


class CTADataset(TimeSeriesDataset):
    group_colname = 'line'
    time_colname = 'date'
    measure_colnames = ['rides']

    @classmethod
    def from_dataframe(cls, df):
        return TimeSeriesDataset.from_dataframe(
            df,
            group_colname=cls.group_colname,
            time_colname=cls.time_colname,
            measure_colnames=cls.measure_colnames,
            dt_unit='D'
        )
