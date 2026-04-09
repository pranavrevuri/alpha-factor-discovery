#!/usr/bin/env python3
"""
analyze.py — Alpha Factor Discovery

Usage:
    Single ticker : python analyze.py AAPL
    Two tickers   : python analyze.py AAPL MSFT
"""

import sys
from data import download_data, clean_data
from indicators import compute_all
from model import add_target, prepare_features, train_random_forest, predict_direction
from charts import plot_single_stock, plot_comparison
from compare import compare_indicators, compare_models


def analyze_ticker(ticker: str) -> dict:
    print(f"\n{'#'*60}")
    print(f"  {ticker} — Alpha Factor Analysis")
    print(f"{'#'*60}")

    df = download_data(ticker)
    df = clean_data(df)
    print(f"Downloaded {len(df)} rows of OHLCV data for {ticker}")

    indicators = compute_all(df)
    print(f"\nLatest 10 rows of indicators:\n{indicators.tail(10)}")

    merged = df[["close", "volume"]].join(indicators)
    df_with_target = add_target(merged)
    X, y = prepare_features(df_with_target)
    print(f"\nFeature matrix shape: {X.shape}")

    model, accuracy, importances = train_random_forest(X, y)

    pred = predict_direction(model, X)
    direction = "UP" if pred == 1 else "DOWN"
    print(f"\n5-day direction prediction (today): {direction}")

    return {"df": df, "indicators": indicators, "accuracy": accuracy, "importances": importances}


def main() -> None:
    tickers = [t.upper().strip() for t in sys.argv[1:]]

    if not tickers:
        print("Usage: python analyze.py AAPL        # single ticker")
        print("       python analyze.py AAPL MSFT   # two tickers")
        sys.exit(1)

    if len(tickers) == 1:
        ticker = tickers[0]
        result = analyze_ticker(ticker)
        plot_single_stock(df=result["df"], indicators=result["indicators"],
                          ticker=ticker, importance=result["importances"],
                          save_path=f"{ticker}_analysis.png")

    elif len(tickers) == 2:
        ticker_a, ticker_b = tickers
        result_a = analyze_ticker(ticker_a)
        result_b = analyze_ticker(ticker_b)
        compare_indicators(ticker_a, result_a["indicators"], ticker_b, result_b["indicators"])
        compare_models(ticker_a, result_a["accuracy"], ticker_b, result_b["accuracy"],
                       result_a["importances"], result_b["importances"])
        plot_comparison(ticker_a, result_a["df"], result_a["indicators"],
                        ticker_b, result_b["df"], result_b["indicators"],
                        result_a["importances"], result_b["importances"], save_dir=".")
    else:
        print("Only 1 or 2 tickers supported.")
        sys.exit(1)


if __name__ == "__main__":
    main()
