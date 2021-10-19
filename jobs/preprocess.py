import os

import pandas as pd

from cta.config import PROCESSED_DATA_DIR, SPLIT_DT
from cta.datasets import get_ridership_dataset


def main():
    df = get_ridership_dataset()

    # Drop pre-covid data
    df = df[df['date'] < pd.Timestamp('2020-01-01')].copy()

    df['dataset'] = (df['date'] >= SPLIT_DT).map({False: 'train', True: 'validation'})

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    output_path = os.path.join(PROCESSED_DATA_DIR, 'daily_ridership.csv')
    pd.to_csv(df, output_path)


if __name__ == '__main__':
    main()
