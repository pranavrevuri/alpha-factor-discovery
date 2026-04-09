"""data.py - Download and clean OHLCV data via yfinance."""
import yfinance as yf
import pandas as pd

def download_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    data = yf.download(ticker, period=period, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [c.lower() for c in data.columns]
    data.reset_index(inplace=True)
    data.rename(columns={"date": "Date"}, inplace=True)
    data.set_index("Date", inplace=True)
    return data

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.dropna(subset=["close", "volume"], inplace=True)
    df.fillna(df.ffill(), inplace=True)
    return df

def get_data(ticker: str) -> pd.DataFrame:
    """Wrapper that downloads and cleans in one call."""
    df = download_data(ticker)
    return clean_data(df)
