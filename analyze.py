import sys
from data import get_data

def main():
    args = sys.argv[1:]

    if len(args) == 0:
        print("Usage:")
        print("  Single ticker:     python analyze.py AAPL")
        print("  Compare two:       python analyze.py AAPL MSFT")
        sys.exit(1)

    elif len(args) == 1:
        ticker = args[0].upper()
        df = get_data(ticker)
        print(f"Downloaded {len(df)} rows for {ticker}")

    elif len(args) == 2:
        ticker1, ticker2 = args[0].upper(), args[1].upper()
        df1, df2 = get_data(ticker1), get_data(ticker2)
        print(f"{ticker1}: {len(df1)} rows | {ticker2}: {len(df2)} rows")

    else:
        print("Error: 1 or 2 tickers only.")
        sys.exit(1)

if __name__ == "__main__":
    main()