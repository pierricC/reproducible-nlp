"""Script that train a classifier on the movie["review"] dataset."""

import nltk
import pandas as pd

from src.config import config as cfg
from src.features.labelencoder import label_encoder
from src.features.sampler import sample_df
from src.modeling.trainer import run_model_cv

if __name__ == "__main__":

    # Read dataset
    df = pd.read_csv(cfg.DATA_PATH)

    # Label encode the target
    df = label_encoder(
        df, column_to_encode=cfg.TARGET, positive_value=cfg.POSITIVE_VALUE
    )

    # We only take a fraction of the data to speed up training time.
    df = sample_df(df, fraction=cfg.FRACTION_SAMPLE, random_state=cfg.SEED)

    X = df[cfg.TEXT_FEATURE]
    # Target
    y = df[cfg.TARGET]

    # Install the nlkt punkt package on the machine if not there already.
    nltk.download("punkt")

    train_kpis, valid_kpis = run_model_cv(
        X, y, params=cfg.PARAMS, n_splits=cfg.NB_SPLIT, random_state=cfg.SEED
    )

    print(train_kpis)

    print(valid_kpis)
