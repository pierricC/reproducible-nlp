"""Test for the inference functions."""


import numpy as np
import pandas as pd

from src.modeling.inference import get_prediction_metrics


def test_get_prediction_metrics():
    """Test for the `get_prediction_metrics` function."""
    # create sample predictions and ground truth
    predictions_test = [0, 1, 0, 1]
    y_test = [1, 1, 0, 0]

    # test calculating the AUC, MCC, and accuracy
    test_kpis = get_prediction_metrics(predictions_test, y_test)
    assert (
        test_kpis["test_auc"] == 0.5
    ), "Expected the AUC to be calculated correctly"
    assert (
        test_kpis["test_mcc"] == 0.0
    ), "Expected the MCC to be calculated correctly"
    assert (
        test_kpis["test_acc"] == 0.5
    ), "Expected the accuracy to be calculated correctly"

    # test with numpy array inputs
    np_predictions: np.ndarray = np.array([0, 1, 0, 1])
    np_y: np.ndarray = np.array([1, 1, 0, 0])
    np_kpis = get_prediction_metrics(np_predictions, np_y)
    assert (
        np_kpis["test_auc"] == 0.5
    ), "Expected the AUC to be calculated correctly for ndarray inputs"

    # test with dataframe inputs
    df_predictions = pd.DataFrame([0, 1, 0, 1])
    df_y = pd.DataFrame([1, 1, 0, 0])
    df_kpis = get_prediction_metrics(df_predictions, df_y)
    assert (
        df_kpis["test_auc"] == 0.5
    ), "Expected the AUC to be calculated correctly for DataFrame inputs"
