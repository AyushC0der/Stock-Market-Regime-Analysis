"""
Milestone 1 - Step 3: Feature engineering + target generation.

RULE #1 (do not break this): every feature for row t must only use
information available AT OR BEFORE day t. Rolling windows and .shift(1)-style
lags are safe. Only the TARGET columns are allowed to look at day t+1,
because that's literally what we're trying to predict.

Reads data/processed/<TICKER>_clean.csv, writes data/processed/<TICKER>_features.csv.

Run: python src/features/engineer.py   (or via main.py)
"""

import os
import numpy as np
import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line - signal_line  # MACD histogram, a common simplification


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Use Adjusted Close, not raw Close, for every return/indicator/target
    # calculation below. Raw Close is NOT split-adjusted, so a stock split
    # (e.g. AAPL 4:1 in Aug 2020, NVDA 4:1 in 2021 + 10:1 in 2024, GOOGL/AMZN
    # 20:1 in 2022) would otherwise show up as a fake ~-75%/-95% single-day
    # return right in the middle of the series - and that fake spike would
    # land directly in Target_Return/Target_Direction, silently corrupting
    # the label. Adj Close is already split- (and dividend-) adjusted.
    close = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

    # --- Returns ---
    df["Return_1D"] = close.pct_change(1)
    df["Return_5D"] = close.pct_change(5)
    df["Return_20D"] = close.pct_change(20)

    # --- Moving averages ---
    df["MA_5"] = close.rolling(5).mean()
    df["MA_20"] = close.rolling(20).mean()
    df["MA_50"] = close.rolling(50).mean()

    # --- Volatility (std dev of daily returns) ---
    df["Volatility_5D"] = df["Return_1D"].rolling(5).std()
    df["Volatility_20D"] = df["Return_1D"].rolling(20).std()

    # --- Volume ---
    df["Volume_Change"] = df["Volume"].pct_change(1)
    df["Volume_MA"] = df["Volume"].rolling(20).mean()

    # --- Technical indicators ---
    df["RSI"] = compute_rsi(close)
    df["MACD"] = compute_macd(close)

    # --- Targets (the ONLY place we look forward) ---
    next_return = close.pct_change(1).shift(-1)
    df["Target_Return"] = next_return
    df["Target_Direction"] = np.where(next_return > 0, 1, 0)

    return df


def main():
    tickers = sorted(
        f[: -len("_clean.csv")] for f in os.listdir(PROCESSED_DIR) if f.endswith("_clean.csv")
    )
    if not tickers:
        print("No cleaned files found in data/processed/. Run clean.py first.")
        return

    for ticker in tickers:
        path = os.path.join(PROCESSED_DIR, f"{ticker}_clean.csv")
        df = pd.read_csv(path, parse_dates=["Date"])
        feat_df = add_features(df)

        # Drop the warm-up period (no MA_50 yet) and the very last row
        # (no Target_Return yet, since there's no "tomorrow" for it).
        feat_df = feat_df.dropna(subset=["MA_50", "Target_Return"]).reset_index(drop=True)

        out_path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
        feat_df.to_csv(out_path, index=False)
        print(f"{ticker}: {len(feat_df)} rows with features -> {out_path}")


if __name__ == "__main__":
    main()
