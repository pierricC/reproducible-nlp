"""Test labelencoder functions."""

import pandas as pd

from src.features.labelencoder import label_encoder

DICT_TEST = {"Response": ["yes", "no", "no", "yes"], "Age": [23, 22, 21, 24]}


def test_label_encoder() -> None:
    """Test the label_encoder function."""
    df_test = pd.DataFrame(DICT_TEST)
    df_sample = label_encoder(
        df=df_test, column_to_encode="Response", positive_value="yes"
    )
    assert df_sample.to_dict() == {
        "Response": {0: 1, 1: 0, 2: 0, 3: 1},
        "Age": {0: 23, 1: 22, 2: 21, 3: 24},
    }
