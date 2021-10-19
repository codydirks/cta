import os
from cta.config import ARTIFACTS_DIR

import torch

from cta.datasets import CTADataset
from cta.etl import load_processed_df
from cta.model import CTAModel


def main():
    df = load_processed_df()

    train_ds = CTADataset(df.query("dataset == 'train"))

    model = CTAModel()

    model.fit(train_ds)

    output_path = os.path.join(ARTIFACTS_DIR, "cta_model.pt")
    torch.save(model.state_dict(), output_path)


if __name__ == '__main__':
    main()
