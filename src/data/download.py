"""
Milestone 1 - Step 1: Download historical OHLCV data.

Downloads daily data for a small, fixed set of tickers and saves ONE raw
CSV per ticker into data/raw/. Raw files are never modified again -
if something looks wrong later, you re-download, you don't hand-edit raw data.

Run: python src/data/download.py   (or via main.py)
"""

import os
import time
import yfinance as yf
import pandas as pd

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"
RAW_DIR = os.path.join("data", "raw")


def download_ticker(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Download one ticker with a couple of retries - Yahoo Finance can be flaky."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}.")
            # yfinance sometimes returns MultiIndex columns even for a single ticker
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            df["Ticker"] = ticker
            return df
        except Exception as e:  # noqa: BLE001 - we want to retry on anything and report it
            last_err = e
            print(f"  attempt {attempt}/{retries} failed for {ticker}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Could not download {ticker} after {retries} attempts") from last_err


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    for ticker in TICKERS:
        print(f"Downloading {ticker} ...")
        df = download_ticker(ticker, START_DATE, END_DATE)
        out_path = os.path.join(RAW_DIR, f"{ticker}.csv")
        df.to_csv(out_path, index=False)
        print(f"  saved {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
