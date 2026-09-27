"""
Milestone 1 - Step 2: Clean raw OHLCV data.

Handles: duplicate dates, missing values, invalid OHLC rows, sorting.
Reads from data/raw/, writes to data/processed/ as <TICKER>_clean.csv.

Run: python src/data/clean.py   (or via main.py)
"""

import os
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close"]


def load_raw(ticker: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, f"{ticker}.csv")
    return pd.read_csv(path, parse_dates=["Date"])


def clean_ticker(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="Date", keep="first")
    df = df.sort_values("Date").reset_index(drop=True)

    # Forward-fill small, isolated gaps (rare data glitches, not real market holidays -
    # holidays simply don't appear as rows at all, so there's nothing to fill for those).
    existing_price_cols = [c for c in PRICE_COLS if c in df.columns]
    df[existing_price_cols] = df[existing_price_cols].ffill()

    # If Close is still missing after ffill, the row is unusable - drop it.
    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    # Sanity checks: High >= Low, Low >= 0, Volume >= 0.
    invalid = (df["High"] < df["Low"]) | (df["Low"] < 0) | (df["Volume"] < 0)
    if invalid.any():
        print(f"  dropping {int(invalid.sum())} invalid rows (bad OHLC/volume)")
        df = df[~invalid].reset_index(drop=True)

    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    tickers = sorted(f[:-4] for f in os.listdir(RAW_DIR) if f.endswith(".csv"))
    if not tickers:
        print("No raw files found in data/raw/. Run download.py first.")
        return

    for ticker in tickers:
        df = load_raw(ticker)
        clean_df = clean_ticker(df)
        out_path = os.path.join(PROCESSED_DIR, f"{ticker}_clean.csv")
        clean_df.to_csv(out_path, index=False)
        print(f"{ticker}: {len(df)} -> {len(clean_df)} rows after cleaning -> {out_path}")


if __name__ == "__main__":
    main()
