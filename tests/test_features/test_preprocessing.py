"""Test preprocessing folder."""

from src.features.preprocessing import sub_regex

test = "Its got a pretty good plot, some good action, good characters."
regex_punctuation = r"[^\w\s]"


def test_sub_regex() -> None:
    """Test the `sub_regex` function."""
    text = sub_regex(test, regex_punctuation)
    assert (
        text == "Its got a pretty good plot some good action good characters"
    )
