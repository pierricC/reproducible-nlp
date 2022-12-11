"""Functions to evaluate trained model."""
from typing import Dict, Union

import numpy as np
import pandas as pd
import sklearn
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score


def evaluate(
    trained_model: sklearn.base.ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Dict[str, float]:
    """
    Evaluate a trained model against a test dataset.

    trained_model
        Previously trained classifier.
    X_test
        Test dataset without labels.
    y_test
        Test labels.
    """
    # Evaluate on test
    predictions_test = trained_model.predict(X_test)

    test_kpis = get_prediction_metrics(predictions_test, y_test)

    return test_kpis


def get_prediction_metrics(
    predictions_test: ArrayLike,
    y_test: ArrayLike,
) -> Dict[str, float]:
    """Get the metrics from a set of predictions and ground truth."""
    test_kpis = {}
    test_kpis["test_auc"] = roc_auc_score(y_test, predictions_test)
    test_kpis["test_mcc"] = matthews_corrcoef(y_test, predictions_test)
    test_kpis["test_acc"] = accuracy_score(y_test, predictions_test)

    return test_kpis


def get_prediction_metrics_fold(
    predictions_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.DataFrame],
    current_kpis: Dict[str, float],
    iteration: int,
) -> Dict[str, float]:
    """
    Get the metrics from a set of predictions and ground truth.

    This function purpose is to be used during cross-validation,
    to get the metrics of the current fold.
    """
    kpis = current_kpis.copy()
    kpis[f"auc_fold_{iteration}"] = roc_auc_score(y_test, predictions_test)
    kpis[f"mcc_fold_{iteration}"] = matthews_corrcoef(y_test, predictions_test)
    kpis[f"acc_fold_{iteration}"] = accuracy_score(y_test, predictions_test)

    return kpis
