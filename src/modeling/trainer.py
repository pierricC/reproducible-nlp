"""Functions for training and evaluating classifiers."""

from typing import Any, Dict, Tuple

import pandas as pd
from sklearn import linear_model, model_selection
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from tqdm import tqdm

from src.features.preprocessing import text_to_vector

# Cross-validation using a simple logistic regression clf.


def run_model_cv(
    X: pd.DataFrame,
    y: pd.DataFrame,
    params: Dict[Any, Any],
    n_splits: int,
    random_state: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
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
    train_kpis = {}
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
        clf = linear_model.LogisticRegression(**params)
        clf.fit(X_train, y_train)

        # Evaluate on fold
        preds_train = clf.predict(X_train)
        preds_valid = clf.predict(X_valid)

        train_kpis[f"train_auc_fold_{i}"] = roc_auc_score(y_train, preds_train)
        train_kpis[f"train_mcc_fold_{i}"] = matthews_corrcoef(
            y_train, preds_train
        )
        train_kpis[f"train_acc_fold_{i}"] = accuracy_score(
            y_valid, preds_valid
        )

        valid_kpis[f"valid_auc_fold_{i}"] = roc_auc_score(y_valid, preds_valid)
        valid_kpis[f"valid_mcc_fold_{i}"] = matthews_corrcoef(
            y_valid, preds_valid
        )
        valid_kpis[f"valid_acc_fold_{i}"] = accuracy_score(
            y_valid, preds_valid
        )

        valid_kpis["valid_auc_mean"] += roc_auc_score(y_valid, preds_valid)
        valid_kpis["valid_mcc_mean"] += matthews_corrcoef(y_valid, preds_valid)
        valid_kpis["valid_acc_mean"] += accuracy_score(y_valid, preds_valid)

        i += 1

    valid_kpis["valid_auc_mean"] /= n_splits
    valid_kpis["valid_mcc_mean"] /= n_splits
    valid_kpis["valid_acc_mean"] /= n_splits

    return train_kpis, valid_kpis


def run_model_holdout(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    params: Dict[Any, Any],
) -> Tuple[Dict[str, float], Dict[str, float]]:
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
    X_test
        Test dataset without labels
    y_test
        Test labels
    params
        Classifier parameters in a dict
    Returns
    -------
    clf
        Trained classifier
    test_kpis
        kpis that contains evaluation metrics of on the test dataset
    """
    # Model training
    clf = linear_model.LogisticRegression(**params)
    clf.fit(X_train, y_train)

    # Evaluate on test
    preds_test = clf.predict(X_test)

    test_kpis = {}
    test_kpis["test_auc"] = roc_auc_score(y_test, preds_test)
    test_kpis["test_mcc"] = matthews_corrcoef(y_test, preds_test)
    test_kpis["test_acc"] = accuracy_score(y_test, preds_test)

    return clf, test_kpis
