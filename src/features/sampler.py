"""Generate samples to reduce computation needs."""
import pandas as pd


def sample_df(
    df: pd.DataFrame, fraction: float, random_state: int
) -> pd.DataFrame:
    """
    Sample a dataframe to a fraction of it.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to sample.
    fraction : float
        Size of the sample to get [Between 0 and 1].
    random_state : int
        Seed for reproducibility.
    """
    df_sample = df.sample(
        frac=fraction,
        random_state=random_state,
    ).reset_index(drop=True)

    return df_sample
