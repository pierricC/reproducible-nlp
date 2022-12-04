"""Functions to apply miscelleanous preprocessing on the data."""
import re
from typing import AnyStr, Callable, List, Optional, Tuple

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


def text_to_vector(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    tokenizer: Callable = word_tokenize,
    token_pattern: str = r"(?u)\\b\\w\\w+\\b",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform a given text into a vector.

    This is done on the basis of the frequency (count) of each word
    that occurs in the entire text.
    The reference text is the one coming from the training set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Full training text to fit the CountVectorizer.
    X_test : pd.DataFrame
        Full testing text to apply the transformation.
    tokenizer : Callable, optional
        A custom tokenizer, if needed.
    token_pattern : str, optional
        Regular expression denoting what constitutes a "token".

    Returns
    -------
    Dict[pd.DataFrame, pd.DataFrame]
        Training and testing set after vectorization.
    """
    count_vec = CountVectorizer(
        tokenizer=tokenizer, token_pattern=token_pattern
    )

    count_vec.fit(X_train)

    X_train = count_vec.transform(X_train)
    X_test = count_vec.transform(X_test)

    return X_train, X_test


def sub_regex(text: str, pattern) -> str:
    """Remove characters from text that matches the regex pattern provided."""
    tag = re.compile(pattern)
    text = tag.sub(r"", text)
    return text


def denoise_text(
    df: pd.DataFrame,
    column_to_denoise: str,
    patterns_to_apply: Optional[List[AnyStr]] = None,
) -> pd.DataFrame:
    """
    Denoise a column in a dataset, by applying regex matching.

    Parameters
    ----------
    df : pd.DataFrame
        Input Dataset
    column_to_denoise : str
        Which column to denoise
    patterns_to_apply : List[AnyStr]
        A list of regex pattern to apply

    Returns
    -------
    pd.DataFrame
        Denoised dataset
    """
    if not patterns_to_apply:
        patterns_to_apply = []
    if column_to_denoise not in list(df.columns):
        raise ValueError(
            f"{column_to_denoise} is not a column in the dataframe"
        )
    df_denoised = df.copy()

    for pattern in patterns_to_apply:
        df_denoised[column_to_denoise] = df_denoised[column_to_denoise].apply(
            sub_regex, pattern=pattern
        )
    return df_denoised
