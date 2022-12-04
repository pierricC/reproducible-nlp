"""Test preprocessing folder."""

from src.config.regex import RE_PUNCTUATION
from src.features.preprocessing import sub_regex

test = "Its got a pretty good plot, some good action, good characters."


def test_sub_regex() -> None:
    """Test the `sub_regex` function."""
    text = sub_regex(test, RE_PUNCTUATION)
    assert (
        text == "Its got a pretty good plot some good action good characters"
    )
