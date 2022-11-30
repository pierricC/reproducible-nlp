"""Script that train a classifier on the movie["review"] dataset."""

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import config as cfg
from src.features.labelencoder import label_encoder
from src.features.preprocessing import text_to_vector
from src.features.sampler import sample_df
from src.modeling.trainer import run_model_holdout

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=cfg.SEED
    )

    X_train, X_test = text_to_vector(X_train, X_test)

    clf, test_kpis = run_model_holdout(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        params=cfg.PARAMS,
    )

    print(test_kpis)
