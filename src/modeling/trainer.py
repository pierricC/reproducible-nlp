"""Functions for training and evaluating classifiers."""

from typing import Any, Dict

import pandas as pd
import sklearn
from sklearn import linear_model, model_selection
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from tqdm import tqdm

from src.features.preprocessing import text_to_vector
from src.modeling.inference import get_prediction_metrics_fold

# Cross-validation using a simple logistic regression clf.


def run_model_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    params: Dict[Any, Any],
    n_splits: int,
    random_state: int,
) -> Dict[str, float]:
    """
    Stratified Cross-validation for simple classifier.

    Parameters
    ----------
    X
        Full dataset without labels
    y
        Full labels
    params
        Classifier parameters in a dict
    n_splits
        Number of splits for the cross-validation
    """
    valid_kpis = {
        "valid_auc_mean": 0.0,
        "valid_mcc_mean": 0.0,
        "valid_acc_mean": 0.0,
    }
    i = 0

    folds = model_selection.StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    for id_train, id_valid in tqdm(folds.split(X=X, y=y)):
        X_train, y_train = X.iloc[id_train], y.iloc[id_train]
        X_valid, y_valid = X.iloc[id_valid], y.iloc[id_valid]

        X_train, X_valid = text_to_vector(X_train, X_valid)

        # Model training
        clf = train_model_holdout(X_train, y_train, params=params)

        # Evaluate on fold
        predictions_valid = clf.predict(X_valid)
        valid_kpis = get_prediction_metrics_fold(
            predictions_valid, y_valid, current_kpis=valid_kpis, iteration=i
        )

        valid_kpis["valid_auc_mean"] += roc_auc_score(
            y_valid, predictions_valid
        )
        valid_kpis["valid_mcc_mean"] += matthews_corrcoef(
            y_valid, predictions_valid
        )
        valid_kpis["valid_acc_mean"] += accuracy_score(
            y_valid, predictions_valid
        )

        i += 1

    valid_kpis["valid_auc_mean"] /= n_splits
    valid_kpis["valid_mcc_mean"] /= n_splits
    valid_kpis["valid_acc_mean"] /= n_splits

    return valid_kpis


def train_model_holdout(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: Dict[Any, Any],
) -> sklearn.base.ClassifierMixin:
    """
    Holdout validation for simple classifier.

    It should be called after cross-validation has been used
    to evaluate the model.

    Parameters
    ----------
    X_train
        Train dataset without labels
    y_train
        Train labels
    params
        Classifier parameters in a dict
    Returns
    -------
    clf
        Trained classifier
    """
    # Model training
    clf = linear_model.LogisticRegression(**params)
    clf.fit(X_train, y_train)

    return clf
