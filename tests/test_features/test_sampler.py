"""Sampler function testing."""

import pandas as pd

from src.features.sampler import sample_df

LIST_TEST = [1.8, 5.5, 4.0, 96.9, -5]


def test_sample_df():
    """Test the sample_df function."""
    df = pd.DataFrame(LIST_TEST)
    df_test = sample_df(df, fraction=0.3, random_state=9)
    assert df_test.to_dict() == {0: {0: 96.9, 1: 1.8}}
