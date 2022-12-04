"""Configuration file to keep default values."""

import src.config.regex as rgx

DATA_PATH = "data/imdb-dataset.csv"
NB_SPLIT = 5
FRACTION_SAMPLE = 0.1
SEED = 0
TEST_SIZE = 0.3
# Default params
PARAMS = {}  # type: ignore

TARGET = "sentiment"
TEXT_FEATURE = "review"
POSITIVE_VALUE = "Positive"

PATTERNS_TO_APPLY = [
    rgx.RE_BRACKET,
    rgx.RE_HTML,
    rgx.RE_PUNCTUATION,
    rgx.RE_URL,
    rgx.RE_SPE_CHARACTERS,
]
