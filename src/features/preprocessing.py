"""Functions to apply miscelleanous preprocessing on the data."""
from typing import Callable, Tuple

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
