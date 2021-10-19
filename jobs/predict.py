import os
import torch
import pandas as pd


from cta.config import ARTIFACTS_DIR, PREDICTIONS_DIR
from cta.datasets import CTADataset
from cta.etl import load_processed_df
from cta.model import CTAModel


def main():
    df = load_processed_df()

    eval_ds = CTADataset(df)

    model = CTAModel()
    model.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "cta_model.pt")))

    with torch.no_grad():
        pred = model(
            eval_ds.tensors[0],
            start_offsets=eval_ds.start_datetimes
        )
        df_pred = pred.to_dataframe(eval_ds)
    df_pred = df_pred.loc[~df_pred['actual'].isnull(), :].reset_index(drop=True)

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    output_path = os.path.join(PREDICTIONS_DIR, 'ridership_predictions.csv')
    df_pred.to_csv(output_path)


if __name__ == '__main__':
    main()
