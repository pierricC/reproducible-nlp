"""Script that train a classifier on the movie["review"] dataset."""

import hydra
import nltk
import pandas as pd

from src.conf.config import ImbdConfig
from src.features.labelencoder import label_encoder
from src.features.preprocessing import denoise_text
from src.features.sampler import sample_df
from src.modeling.trainer import run_model_cv


@hydra.main(version_base=None, config_path="src/conf", config_name="config")
def main(cfg: ImbdConfig):
    """Pipeline to evaluate model with cross-validation."""
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

    valid_kpis = run_model_cv(
        X,
        y,
        params=cfg.params.logistic_reg,
        n_splits=cfg.preprocess.nb_split,
        random_state=cfg.preprocess.seed,
    )

    print(valid_kpis)


if __name__ == "__main__":
    main()
