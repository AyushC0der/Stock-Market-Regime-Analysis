"""One entry point for the complete Unit 1 + Unit 2 teaching project.

Run `python main.py` to use the saved dataset and show all four units.
Run `python main.py --unit 1`, `--unit 2`, `--unit 3`, or `--unit 4` for one unit.
Run `python main.py --refresh` only when you want to download the data again.
"""

import argparse
import os

import numpy as np
import pandas as pd

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"


def download_data():
    """Download once when needed; the saved CSVs can be reused offline."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install requirements with: pip install -r requirements.txt") from exc

    os.makedirs(RAW_DIR, exist_ok=True)
    for ticker in TICKERS:
        print(f"Downloading {ticker}...")
        frame = yf.download(ticker, start=START_DATE, end=END_DATE,
                            auto_adjust=False, progress=False)
        if frame.empty:
            raise RuntimeError(f"No data returned for {ticker}. Check your internet and retry.")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.reset_index()
        frame["Ticker"] = ticker
        path = os.path.join(RAW_DIR, f"{ticker}.csv")
        temp = path + ".tmp"
        frame.to_csv(temp, index=False)
        os.replace(temp, path)
        print(f"  saved {len(frame)} rows")


def add_features(frame):
    """Create backward-looking indicators and tomorrow's two targets."""
    frame = frame.copy()
    close = frame["Adj Close"] if "Adj Close" in frame else frame["Close"]
    daily = close.pct_change()
    frame["Return_1D"] = daily
    frame["Return_5D"] = close.pct_change(5)
    frame["Return_20D"] = close.pct_change(20)
    frame["MA_5"] = close.rolling(5).mean()
    frame["MA_20"] = close.rolling(20).mean()
    frame["MA_50"] = close.rolling(50).mean()
    frame["Volatility_5D"] = daily.rolling(5).std()
    frame["Volatility_20D"] = daily.rolling(20).std()
    frame["Volume_Change"] = frame["Volume"].pct_change()
    frame["Volume_MA"] = frame["Volume"].rolling(20).mean()

    change = close.diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    relative_strength = gain / loss.replace(0, np.nan)
    frame["RSI"] = 100 - 100 / (1 + relative_strength)

    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    macd = fast - slow
    frame["MACD"] = macd - macd.ewm(span=9, adjust=False).mean()

    next_return = daily.shift(-1)
    frame["Target_Return"] = next_return
    frame["Target_Direction"] = (next_return > 0).astype(int)
    return frame.replace([np.inf, -np.inf], np.nan)


def prepare_data(refresh=False):
    """Reuse features when available; otherwise build them from raw CSVs."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    feature_paths = [os.path.join(PROCESSED_DIR, f"{t}_features.csv") for t in TICKERS]
    if not refresh and all(os.path.exists(path) for path in feature_paths):
        print("Using the saved processed data (no download needed).")
        return pd.concat([pd.read_csv(p, parse_dates=["Date"]) for p in feature_paths],
                         ignore_index=True)

    if refresh or not all(os.path.exists(os.path.join(RAW_DIR, f"{t}.csv")) for t in TICKERS):
        download_data()

    frames = []
    for ticker in TICKERS:
        path = os.path.join(RAW_DIR, f"{ticker}.csv")
        frame = pd.read_csv(path, parse_dates=["Date"])
        frame = frame.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
        if "Ticker" not in frame:
            frame["Ticker"] = ticker
        if "Adj Close" not in frame:
            frame["Adj Close"] = frame["Close"]
        frame = frame.dropna(subset=["Close"])
        features = add_features(frame).dropna(subset=["MA_50", "Target_Return", "RSI"])
        out = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
        features.to_csv(out, index=False)
        print(f"Prepared {ticker}: {len(features)} usable rows")
        frames.append(features)
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Easy-to-follow Unit 1 and Unit 2 ML lessons")
    parser.add_argument("--unit", choices=["1", "2", "3", "4", "all"], default="all",
                        help="choose one unit, or run all four (default: all)")
    parser.add_argument("--refresh", action="store_true",
                        help="download the historical stock data again")
    args = parser.parse_args()

    data = prepare_data(refresh=args.refresh)
    print(f"\nDataset ready: {len(data)} rows across {data['Ticker'].nunique()} stocks.\n")

    if args.unit in ("1", "all"):
        import unit1
        unit1.run(data)
    if args.unit in ("2", "all"):
        import unit2
        unit2.run(data)
    if args.unit in ("3", "all"):
        import unit3
        unit3.run(data)
    if args.unit in ("4", "all"):
        import unit4
        unit4.run(data)


if __name__ == "__main__":
    main()
