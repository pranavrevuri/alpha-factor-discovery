"""compare.py - Side-by-side comparison of two tickers."""

import pandas as pd


def compare_indicators(ticker_a: str, indicators_a: pd.DataFrame,
                         ticker_b: str, indicators_b: pd.DataFrame) -> None:
    """
    Print a side-by-side table of the latest indicator values
    for two tickers, and a per-indicator winner summary.
    """
    cols = ["RSI", "MACD", "Signal", "Histogram", "SMA", "EMA",
            "BB_Upper", "BB_Middle", "BB_Lower", "VolumeRatio"]

    latest_a = indicators_a[cols].iloc[-1]
    latest_b = indicators_b[cols].iloc[-1]

    print(f"\n{'='*60}")
    print(f"  INDICATOR COMPARISON: {ticker_a} vs {ticker_b}")
    print(f"{'='*60}")
    print(f"{'Indicator':<15} {ticker_a:>12} {ticker_b:>12} {'Winner':>12}")
    print("-" * 55)

    for col in cols:
        val_a = latest_a[col]
        val_b = latest_b[col]
        if col == "RSI":
            # Winner = further from overbought/oversold extremes
            dist_a = abs(val_a - 50)
            dist_b = abs(val_b - 50)
            winner = ticker_a if dist_a > dist_b else ticker_b
        elif col in ("BB_Upper", "BB_Lower", "BB_Middle"):
            winner = ticker_a if val_a > val_b else ticker_b
        elif col in ("MACD", "Signal", "Histogram"):
            winner = ticker_a if val_a > val_b else ticker_b
        else:
            winner = ticker_a if val_a > val_b else ticker_b
        print(f"{col:<15} {val_a:>12.4f} {val_b:>12.4f} {winner:>12}")

    print("-" * 55)


def compare_models(ticker_a: str, acc_a: float,
                    ticker_b: str, acc_b: float,
                    imp_a: pd.Series, imp_b: pd.Series) -> None:
    """Print model accuracy and top feature for each ticker."""
    print(f"\n{'='*60}")
    print(f"  MODEL COMPARISON: {ticker_a} vs {ticker_b}")
    print(f"{'='*60}")
    print(f"  {ticker_a} Accuracy: {acc_a:.4f}  |  Top Feature: {imp_a.idxmax()}")
    print(f"  {ticker_b} Accuracy: {acc_b:.4f}  |  Top Feature: {imp_b.idxmax()}")
    winner = ticker_a if acc_a > acc_b else ticker_b
    print(f"\n  Higher accuracy model: {winner}")
