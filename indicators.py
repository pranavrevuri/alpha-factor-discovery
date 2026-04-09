"""indicators.py - Compute the 5 technical indicators manually."""

import pandas as pd
import numpy as np


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    RS  = avg gain / avg loss  (over window days)
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    # Use exponential moving average for subsequent values (Wilder smoothing)
    avg_gain_smooth = avg_gain.copy()
    avg_loss_smooth = avg_loss.copy()

    for i in range(window, len(close)):
        avg_gain_smooth.iloc[i] = (
            (avg_gain_smooth.iloc[i - 1] * (window - 1) + gain.iloc[i]) / window
        )
        avg_loss_smooth.iloc[i] = (
            (avg_loss_smooth.iloc[i - 1] * (window - 1) + loss.iloc[i]) / window
        )

    rs = avg_gain_smooth / avg_loss_smooth
    rsi = 100 - (100 / (1 + rs))
    rsi.name = "RSI"
    return rsi


def compute_macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Moving Average Convergence/Divergence.

    MACD line  = fast EMA - slow EMA
    Signal line = EMA of MACD line
    Histogram  = MACD line - Signal line
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    result = pd.DataFrame(
        {"MACD": macd_line, "Signal": signal_line, "Histogram": histogram},
        index=close.index,
    )
    return result


def compute_sma(close: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average."""
    sma = close.rolling(window=window, min_periods=window).mean()
    sma.name = "SMA"
    return sma


def compute_ema(close: pd.Series, window: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    ema = close.ewm(span=window, adjust=False).mean()
    ema.name = "EMA"
    return ema


def compute_bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands.

    Middle = SMA(close, window)
    Upper  = Middle + num_std * std(close, window)
    Lower  = Middle - num_std * std(close, window)
    """
    sma = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std()

    upper = sma + num_std * std
    lower = sma - num_std * std

    result = pd.DataFrame(
        {"BB_Upper": upper, "BB_Middle": sma, "BB_Lower": lower},
        index=close.index,
    )
    return result


def compute_volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Volume Ratio = Today's volume / 20-day average volume.
    """
    avg_vol = volume.rolling(window=window, min_periods=window).mean()
    vr = volume / avg_vol
    vr.name = "VolumeRatio"
    return vr


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 5 indicators for a dataframe that has
    'close' and 'volume' columns.

    Returns a DataFrame with columns:
    RSI, MACD, Signal, Histogram, SMA, EMA,
    BB_Upper, BB_Middle, BB_Lower, VolumeRatio
    """
    close = df["close"]
    volume = df["volume"]

    rsi = compute_rsi(close)
    macd = compute_macd(close)
    sma = compute_sma(close)
    ema = compute_ema(close)
    bb = compute_bollinger_bands(close)
    vr = compute_volume_ratio(volume)

    indicators = pd.DataFrame(index=close.index)
    indicators["RSI"] = rsi
    indicators["MACD"] = macd["MACD"]
    indicators["Signal"] = macd["Signal"]
    indicators["Histogram"] = macd["Histogram"]
    indicators["SMA"] = sma
    indicators["EMA"] = ema
    indicators["BB_Upper"] = bb["BB_Upper"]
    indicators["BB_Middle"] = bb["BB_Middle"]
    indicators["BB_Lower"] = bb["BB_Lower"]
    indicators["VolumeRatio"] = vr

    return indicators
