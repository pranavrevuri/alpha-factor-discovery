# Alpha Factor Discovery

**Team:** Pranav Revuri & Krish Shah
**Track:** FPD FinTech – Junior Analyst Project

---

## What It Does

`analyze.py` runs a complete alpha-factor discovery pipeline:

1. **Downloads 1 year** of OHLCV price data via `yfinance`
2. **Computes 5 technical indicators** manually (no `ta-lib`):
   - RSI (14-day)
   - MACD (12/26/9)
   - SMA (20-day)
   - EMA (20-day)
   - Bollinger Bands (20-day, ±2σ)
   - Volume Ratio (today's volume / 20-day avg)
3. **Trains a RandomForest** to predict whether the price goes up or down 5 days later
4. **Prints** a full indicator summary table, model accuracy, and ranked feature importances
5. **Generates charts** saved as PNG files

---

## Install

```bash
pip install -r requirements.txt
```

Required packages: `yfinance`, `pandas`, `matplotlib`, `scikit-learn`, `numpy`

---

## Run

### Single Ticker
```bash
python analyze.py AAPL
```

Output:
- Indicator summary table (last 10 rows)
- Model test accuracy and ranked feature importances
- `AAPL_analysis.png` chart file

### Two Tickers (Comparison Mode)
```bash
python analyze.py AAPL MSFT
```

Output:
- Full analysis for both tickers
- Side-by-side indicator comparison table
- Model comparison (accuracy + top feature)
- `comparison_rsi.png` and `comparison_importances.png`

---

## File Structure

```
alpha-factor/
├── analyze.py      # CLI entry point
├── data.py         # yfinance download & cleaning
├── indicators.py   # RSI, MACD, SMA, EMA, BB, VolumeRatio
├── model.py        # RandomForest training & prediction
├── charts.py       # Matplotlib visualisation
├── compare.py      # Side-by-side comparison logic
├── requirements.txt
└── README.md
```

---

## Indicators

| Indicator | Type | Description |
|-----------|------|-------------|
| RSI | Momentum | 14-day avg gain/loss ratio — overbought >70, oversold <30 |
| MACD | Trend | 12-day EMA − 26-day EMA; signal line = 9-day EMA of MACD |
| SMA | Trend | Simple 20-day moving average |
| EMA | Trend | Exponential 20-day moving average (recent days weighted more) |
| Bollinger Bands | Volatility | 20-day SMA ± 2 standard deviations |
| Volume Ratio | Volume | Today's volume ÷ 20-day average volume |

---

## Model

- **Algorithm:** RandomForestClassifier (100 trees, random_state=42)
- **Target:** 1 if price 5 days in the future > today's close; 0 otherwise
- **Split:** Chronological — first 80% train, last 20% test (no random shuffle)
- **Features:** All 5 indicators (RSI, MACD, Signal, Histogram, SMA, EMA, BB_Upper, BB_Middle, BB_Lower, VolumeRatio)
