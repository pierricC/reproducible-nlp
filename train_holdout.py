"""Script that train a classifier on the movie["review"] dataset."""

import hydra
import mlflow
import mlflow.sklearn
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split

from src.conf.config import ImbdConfig
from src.features.labelencoder import label_encoder
from src.features.preprocessing import denoise_text, text_to_vector
from src.features.sampler import sample_df
from src.modeling.inference import evaluate
from src.modeling.trainer import train_model_holdout
from src.utils import git
from src.utils.io import ensure_file_directory, save_to_json


@hydra.main(version_base=None, config_path="src/conf", config_name="config")
def main(cfg: ImbdConfig):
    """Pipeline to train and save a model with holdout data."""
    # Read dataset
    df = pd.read_csv(cfg.paths.raw_dataset)

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

    ensure_file_directory(cfg.paths.cleaned_train_dataset)
    X_train.to_csv(cfg.paths.cleaned_train_dataset)
    X_test.to_csv(cfg.paths.cleaned_test_dataset)
    y_train.to_csv(cfg.paths.cleaned_train_labels)
    y_test.to_csv(cfg.paths.cleaned_test_labels)

    X_train, X_test = text_to_vector(X_train, X_test)

    mlflow.set_tracking_uri(cfg.ml_registry.tracking_uri)

    with mlflow.start_run():

        clf = train_model_holdout(
            X_train=X_train,
            y_train=y_train,
            params=cfg.params.logistic_reg,
        )

        test_kpis = evaluate(clf, X_test, y_test)

        # Log config parameters and metrics
        mlflow.log_params(cfg.params.logistic_reg)
        mlflow.log_params(cfg.preprocess)
        mlflow.log_metrics(test_kpis)

        mlflow.set_tag("branch", git.get_current_git_branch())
        mlflow.set_tag("commit", git.get_current_git_commit())

        # Log the sklearn model
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="clf_sentiment",
            registered_model_name="sentiment-sklearn-clf",
        )
        filepath = "results/test_kpis.json"
        save_to_json(filepath=filepath, object_to_save=test_kpis)


if __name__ == "__main__":
    main()
