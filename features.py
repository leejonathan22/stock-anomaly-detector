"""
features.py — Engineers trading features from raw OHLCV data for anomaly detection.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add five technical/statistical features to an OHLCV DataFrame.

    Features added:
        daily_return    — percentage change in Close price day-over-day
        volume_zscore   — rolling 20-day z-score of Volume (how unusual today's vol is)
        intraday_range  — (High - Low) / Open * 100  (daily price swing as % of open)
        volatility      — rolling 20-day standard deviation of daily_return
        rel_volume      — today's Volume divided by its rolling 20-day mean

    Args:
        df: DataFrame with columns Open, High, Low, Close, Volume and a DatetimeIndex

    Returns:
        Enriched DataFrame with 5 new feature columns; NaN rows dropped.
    """

    # Work on a copy so we never mutate the caller's DataFrame
    df = df.copy()

    # --- Feature 1: Daily Return ---
    # Percentage change in closing price relative to the previous day
    df["daily_return"] = df["Close"].pct_change() * 100

    # --- Feature 2: Volume Z-Score ---
    # Measures how many standard deviations today's volume is from its 20-day mean.
    # A large positive z-score signals unusually high trading activity.
    rolling_vol = df["Volume"].rolling(window=20)
    df["volume_zscore"] = (df["Volume"] - rolling_vol.mean()) / rolling_vol.std()

    # --- Feature 3: Intraday Range ---
    # Captures the magnitude of intraday price movement as a percentage of the open.
    # Large values indicate volatile or event-driven sessions.
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Open"] * 100

    # --- Feature 4: Volatility ---
    # Rolling 20-day standard deviation of daily returns.
    # Spikes indicate regime changes or periods of heightened uncertainty.
    df["volatility"] = df["daily_return"].rolling(window=20).std()

    # --- Feature 5: Relative Volume ---
    # Today's volume relative to its 20-day average.
    # Values > 2 typically mean 2× normal volume — often driven by news/events.
    df["rel_volume"] = df["Volume"] / df["Volume"].rolling(window=20).mean()

    # Drop rows that contain NaN values introduced by rolling windows (first ~20 rows)
    df.dropna(inplace=True)

    return df
