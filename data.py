"""
data.py — Fetches and validates historical OHLCV stock data using yfinance.
"""

import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Download OHLCV (Open, High, Low, Close, Volume) data for a given ticker.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL"
        period: Time period to fetch, e.g. "2y" for 2 years (yfinance format)

    Returns:
        DataFrame with DatetimeIndex and columns: Open, High, Low, Close, Volume

    Raises:
        ValueError: If the ticker is invalid or no data is returned
    """

    # Normalize ticker to uppercase to avoid case-sensitivity issues
    ticker = ticker.strip().upper()

    # Download data via yfinance; auto_adjust=True adjusts for splits/dividends
    try:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        error_msg = str(e)
        if "Too Many Requests" in error_msg or "Rate" in error_msg:
            raise ConnectionError(
                "yfinance rate limit hit. Please wait 30–60 seconds and try again."
            )
        raise ValueError(f"Failed to download data for '{ticker}': {error_msg}")

    # yfinance returns an empty DataFrame for unknown / delisted tickers
    if raw.empty:
        raise ValueError(
            f"No data found for ticker '{ticker}'. "
            "Check that the symbol is correct and listed on a supported exchange."
        )

    # Keep only the core OHLCV columns; drop any extras (e.g. Dividends, Stock Splits)
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = raw[ohlcv_cols].copy()

    # Flatten MultiIndex columns that yfinance sometimes returns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure the index is named "Date" for downstream consistency
    df.index.name = "Date"

    # Drop any rows where all values are NaN (can occur on market holidays)
    df.dropna(how="all", inplace=True)

    return df
