"""
Unit 2 - shared data loading, feature prep, and chronological splitting.

Every regression/classification module in Unit 2 imports from here so that
one thing is enforced everywhere: the train/val/test split is chronological
(never shuffled), and any scaling/normalization is FIT ON THE TRAINING SET
ONLY, then applied unchanged to validation/test. Fitting a scaler on the
full dataset before splitting is a classic, subtle form of data leakage -
the scaler would "see" the mean/std of future data during training - so
we make that mistake structurally impossible by fitting it strictly after
the split, inside get_model_ready_splits().

We also derive a few PRICE-RELATIVE features (Price_to_MA5, etc.) instead
of using the raw moving-average columns from engineer.py directly. AAPL's
share price and GOOGL's share price live on totally different scales, so a
model pooling all 5 tickers together needs scale-invariant features - a
raw `MA_50` column would mostly just encode "which ticker is this,"
not any real momentum signal.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.split import time_series_split

PROCESSED_DIR = os.path.join("data", "processed")
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

RANDOM_SEED = 42  # every stochastic model in Unit 2 is seeded with this

# The feature set every Unit 2 model is trained on. All of these are
# strictly backward-looking (see engineer.py's Rule #1), and all are
# either already scale-free (returns, RSI, ratios) or made scale-free
# below (the Price_to_MA* / Volume_to_MA ratios).
FEATURE_COLUMNS = [
    "Return_1D", "Return_5D", "Return_20D",
    "Price_to_MA5", "Price_to_MA20", "Price_to_MA50",
    "Volatility_5D", "Volatility_20D",
    "Volume_Change", "Volume_to_MA",
    "RSI", "MACD",
]

REGRESSION_TARGET = "Target_Return"
CLASSIFICATION_TARGET = "Target_Direction"


def load_pooled_dataset(tickers=None) -> pd.DataFrame:
    """Concatenate every ticker's *_features.csv into one long table, tagged by Ticker."""
    tickers = tickers or TICKERS
    frames = []
    for ticker in tickers:
        path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run `python main.py` first.")
        df = pd.read_csv(path, parse_dates=["Date"])
        df["Ticker"] = ticker
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_derived_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn absolute-price moving averages into scale-free ratios around 0."""
    df = df.copy()
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    df["Price_to_MA5"] = df[price_col] / df["MA_5"] - 1
    df["Price_to_MA20"] = df[price_col] / df["MA_20"] - 1
    df["Price_to_MA50"] = df[price_col] / df["MA_50"] - 1
    df["Volume_to_MA"] = df["Volume"] / df["Volume_MA"] - 1
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=FEATURE_COLUMNS + [REGRESSION_TARGET, CLASSIFICATION_TARGET])


def load_model_ready_dataset(tickers=None) -> pd.DataFrame:
    """One-call convenience: load + derive features, ready to split."""
    return add_derived_ratio_features(load_pooled_dataset(tickers))


def chronological_split(df: pd.DataFrame, train_end="2021-12-31", val_end="2023-12-31"):
    """
    Thin wrapper around src.data.split.time_series_split. Splitting a
    POOLED multi-ticker table by Date this way is still leakage-free:
    every row in train has a date <= train_end regardless of which
    ticker it belongs to, so no row anywhere in val/test describes
    information from before its own date.
    """
    return time_series_split(df, date_col="Date", train_end=train_end, val_end=val_end)


def get_model_ready_splits(
    df: pd.DataFrame,
    target_col: str,
    feature_cols=None,
    train_end="2021-12-31",
    val_end="2023-12-31",
    scale: bool = True,
):
    """
    Returns a dict with X_train/X_val/X_test (as numpy arrays), the
    matching y arrays, the fitted scaler (or None if scale=False), and
    the feature name list - everything a model needs, with the scaler
    fit strictly on the training split.
    """
    feature_cols = feature_cols or FEATURE_COLUMNS
    train_df, val_df, test_df = chronological_split(df, train_end, val_end)

    X_train = train_df[feature_cols].to_numpy()
    X_val = val_df[feature_cols].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()

    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)          # fit ONLY on train
        X_val = scaler.transform(X_val)                   # re-use train's mean/std
        X_test = scaler.transform(X_test)                  # re-use train's mean/std

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
    }


def print_split_sizes(splits: dict, label: str):
    print(f"\n{label} - chronological split sizes:")
    print(f"  train: {len(splits['y_train'])} rows")
    print(f"  val:   {len(splits['y_val'])} rows")
    print(f"  test:  {len(splits['y_test'])} rows")
