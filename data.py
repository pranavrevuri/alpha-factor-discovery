import yfinance as yf
import pandas as pd

def get_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
    
    # Flatten columns if multi-level (yfinance sometimes returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    
    # Drop rows with missing values
    df.dropna(inplace=True)
    
    # Make sure index is datetime
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    return df