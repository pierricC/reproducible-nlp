"""Script that train a classifier on the movie["review"] dataset."""

import json

import hydra
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

from src.conf.config import ImbdConfig
from src.features.labelencoder import label_encoder
from src.features.preprocessing import denoise_text, text_to_vector
from src.features.sampler import sample_df
from src.modeling.trainer import run_model_holdout
from src.utils.io import ensure_directory


@hydra.main(version_base=None, config_path="src/conf", config_name="config")
def main(cfg: ImbdConfig):
    """Pipeline to train and save a model with holdout data."""
    # Read dataset
    df = pd.read_csv(cfg.paths.dataset)

    # We only take a fraction of the data to speed up training time.
    df = sample_df(
        df,
        fraction=cfg.preprocess.fraction_sample,
        random_state=cfg.preprocess.seed,
    )

    # Label encode the target
    df = label_encoder(
        df,
        column_to_encode=cfg.data.target,
        positive_value=cfg.data.positive_value,
    )

    df = denoise_text(
        df,
        column_to_denoise=cfg.data.text_feature,
        patterns_to_apply=list(cfg.preprocess.regex_pattern_to_apply.values()),
    )

    X = df[cfg.data.text_feature]
    # Target
    y = df[cfg.data.target]

    # Install the nlkt punkt package on the machine if not there already.
    nltk.download("punkt")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.preprocess.test_size,
        random_state=cfg.preprocess.seed,
    )

    X_train, X_test = text_to_vector(X_train, X_test)

    clf, test_kpis = run_model_holdout(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        params=cfg.params.logistic_reg,
    )

    filepath = "results/test_kpis.json"

    print(test_kpis)

    ensure_directory(filepath)
    with open(filepath, "w") as f:
        json.dump(test_kpis, f)


if __name__ == "__main__":
    main()
