"""Functions to apply label encoding on the data."""
import pandas as pd


def label_encoder(
    df: pd.DataFrame, column_to_encode: str, positive_value: str
) -> pd.DataFrame:
    """
    Apply simple encoder to one column, most of the time the target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to labelencode.
    column_to_encode : str
        Which column to apply in the dataframe with binary classes.
    positive_value : str
        The name of the category that is "positive".
    """
    # Label encode the target
    if column_to_encode not in list(df.columns):
        raise ValueError(
            f"{column_to_encode} is not a column in the dataframe"
        )
    if positive_value not in list(df[column_to_encode]):
        raise ValueError(f"{positive_value} : this category doesn't exist.")

    df_encode = df.copy()
    df_encode[column_to_encode] = df[column_to_encode].apply(
        lambda x: 1 if x == positive_value else 0
    )
    return df_encode
