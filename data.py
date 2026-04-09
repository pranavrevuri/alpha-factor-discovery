"""data.py - Download and clean OHLCV data via yfinance."""

import yfinance as yf
import pandas as pd


def download_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Download 1 year of OHLCV data for the given ticker.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL').
    period : str
        yfinance period string (default '1y' = 1 year).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume,
        and a DatetimeIndex named 'Date'.
    """
    data = yf.download(ticker, period=period, progress=False)

    # Handle MultiIndex columns from newer yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [c.lower() for c in data.columns]
    data.reset_index(inplace=True)
    data.rename(columns={"date": "Date"}, inplace=True)
    data.set_index("Date", inplace=True)
    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN in critical columns and forward-fill gaps."""
    df = df.copy()
    df.dropna(subset=["close", "volume"], inplace=True)
    df.fillna(df.ffill(), inplace=True)
    return df
