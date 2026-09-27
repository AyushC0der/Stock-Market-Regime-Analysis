"""
Chronological train/validation/test split.

NEVER use sklearn's train_test_split() on this data - that shuffles rows
randomly, which means the model would "learn from the future" during
training. That's the single most common way student stock-prediction
projects quietly cheat without realizing it.
"""

import pandas as pd


def time_series_split(
    df: pd.DataFrame,
    date_col: str = "Date",
    train_end: str = "2021-12-31",
    val_end: str = "2023-12-31",
):
    """
    Split rows strictly by date:
        train : date <= train_end
        val   : train_end < date <= val_end
        test  : date > val_end
    Adjust the cutoff dates once you've looked at how much data you actually got back.
    """
    df = df.sort_values(date_col)
    train = df[df[date_col] <= train_end]
    val = df[(df[date_col] > train_end) & (df[date_col] <= val_end)]
    test = df[df[date_col] > val_end]
    return train, val, test
