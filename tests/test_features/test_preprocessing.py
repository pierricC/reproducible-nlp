"""Test preprocessing folder."""

import pandas as pd

from src.features.preprocessing import denoise_text, sub_regex


def test_sub_regex() -> None:
    """Test the `sub_regex` function."""
    test = "Its got a pretty good plot, some good action, good characters."
    regex_punctuation = r"[^\w\s]"

    text = sub_regex(test, regex_punctuation)
    assert (
        text == "Its got a pretty good plot some good action good characters"
    )


def test_denoise_text():
    """Tests for the `denoise_text` function."""
    # create a sample dataframe
    df = pd.DataFrame(
        {
            "col1": ["abc", "def", "ghi"],
            "col2": ["jkl", "mno", "pqr"],
        }
    )

    df_empty = pd.DataFrame(
        {
            "col1": ["", None, "ghi"],
            "col2": ["jkl", "mno", ""],
        }
    )

    # test raising a ValueError if column_to_denoise is not in the dataframe
    try:
        denoise_text(df, "col3")
    except ValueError as e:
        assert str(e) == "col3 is not a column in the dataframe"
    else:
        assert False, "Expected a ValueError to be raised"

    # test applying the regex patterns to the specified column
    patterns = ["bc", "ef"]
    assert denoise_text(df, "col1", patterns).equals(
        pd.DataFrame(
            {"col1": ["a", "d", "ghi"], "col2": ["jkl", "mno", "pqr"]}
        )
    )

    # test handling empty or missing values in the input dataframe
    df_denoised = denoise_text(df_empty, "col1")
    assert df_denoised["col1"].tolist() == [
        "",
        None,
        "ghi",
    ], "Expected empty or missing values to be handled correctly"

    # test handling when no regex patterns are provided
    df_denoised = denoise_text(df, "col1")
    assert df_denoised["col1"].tolist() == [
        "abc",
        "def",
        "ghi",
    ], "Expected the original dataframe to be returned if no patterns are provided"

    # test handling an empty list of patterns
    df_denoised = denoise_text(df, "col1", [])
    assert df_denoised["col1"].tolist() == [
        "abc",
        "def",
        "ghi",
    ], "Expected the original dataframe to be returned if no patterns are provided"
