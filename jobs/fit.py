import os
import torch
import pandas as pd

from cta.config import ARTIFACTS_DIR, SPLIT_DT
from cta.datasets import CTADataset
from cta.etl import load_processed_df
from cta.model import CTAModel


def main():
    df = load_processed_df()
    df['date'] = pd.to_datetime(df['date'])

    # For temporary speed-up purposes, only consider 1 year of training data.
    # This will yield bad predictions if left in.

    df = df[df['date'] >= SPLIT_DT.replace(year=SPLIT_DT.year-1)]

    train_ds = CTADataset.from_dataframe(df.query("dataset == 'train'"))

    model = CTAModel()

    model.fit(train_ds)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    output_path = os.path.join(ARTIFACTS_DIR, "cta_model.pt")
    torch.save(model.state_dict(), output_path)


if __name__ == '__main__':
    main()
