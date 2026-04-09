# Alpha Factor Discovery — Results Guide

## Running the Analysis

```bash
# Single ticker
python analyze.py AAPL

# Two tickers (comparison mode)
python analyze.py AAPL MSFT
```

---

## Understanding Your Results (AAPL Example)

### 1. Data Downloaded

```
Downloaded 251 rows of OHLCV data for AAPL
```

**What this means:** 251 days of daily Open, High, Low, Close, Volume data was fetched from Yahoo Finance covering approximately the last year. This serves as the raw input for all indicator calculations.

---

### 2. Technical Indicators

After running `compute_all()`, you get 10 indicators per day. The last 10 rows printed look like this:

```
                  RSI      MACD    Signal  Histogram   SMA    EMA   BB_Upper  BB_Middle  BB_Lower  VolumeRatio
2026-03-26  43.076593 -3.538753 -3.398417  -0.140336  ...   ...   ...       ...       ...       1.000570
...
2026-04-09  52.892597 -0.777591 -2.033399  1.255808   ...   ...   ...       ...       ...       0.232553
```

| Indicator | What it measures | Latest AAPL value |
|---|---|---|
| **RSI (14)** | Momentum — how overbought/oversold the stock is | 52.89 (neutral zone) |
| **MACD** | Trend — difference between 12-day and 26-day EMA | -0.78 (slightly bearish) |
| **Signal** | Signal line — 9-day EMA of MACD | -2.03 |
| **Histogram** | MACD minus Signal | +1.26 (positive = bullish momentum) |
| **SMA (20)** | Simple average of last 20 close prices | ~252.94 |
| **EMA (20)** | Exponentially weighted average (more weight to recent) | ~252.94 |
| **BB_Upper** | Upper Bollinger Band (+2 std dev from SMA) | ~253.06 |
| **BB_Middle** | Middle Bollinger Band (SMA 20) | ~252.94 |
| **BB_Lower** | Lower Bollinger Band (-2 std dev from SMA) | ~245.78 |
| **VolumeRatio** | Today's volume / 20-day average volume | 0.23 (very low volume day) |

**Key interpretation points:**
- **RSI < 30** = oversold (potential buy signal), **RSI > 70** = overbought (potential sell signal). AAPL at 52.89 is neutral.
- **MACD above Signal** = bullish; below = bearish. AAPL's Histogram is positive (+1.26), suggesting short-term bullish momentum despite MACD still being negative.
- **Bollinger Bands** show the price is trading within the bands. The narrow gap between BB_Upper and BB_Lower suggests low volatility.
- **VolumeRatio < 1** means today had below-average volume (0.23 = 23% of the 20-day average).

---

### 3. Feature Matrix

```
Feature matrix shape: (232, 10)
```

**What this means:** After dropping rows with NaN values (due to rolling calculations needing history), you have 232 usable data points with 10 features each. These 232 rows are split 80/20 for training/testing.

---

### 4. Model Accuracy

```
Model Test Accuracy: 0.5957
```

**What this means:** The Random Forest correctly predicted whether the price would go up or down 5 days later **59.57% of the time** on the held-out test set (last 20% of data). This is slightly better than random guessing (50%), suggesting the model captures some signal, but is not highly reliable on its own — real trading systems would need more refinement.

> **Note:** 59.6% accuracy does NOT mean it wins 59.6% of the time in live trading. The test is on historical data, and the actual 5-day forward prediction is based on the most recent data point.

---

### 5. Feature Importances

```
Feature Importances (ranked):
  BB_Upper: 0.1387    ← Most important feature
  EMA: 0.1316
  Signal: 0.1125
  Histogram: 0.1049
  BB_Lower: 0.1025
  BB_Middle: 0.0969
  SMA: 0.0940
  RSI: 0.0869
  MACD: 0.0746
  VolumeRatio: 0.0574  ← Least important feature
```

**What this means:** The Random Forest ranked features by how much each one contributes to making accurate predictions.

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | BB_Upper | 13.87% | Volatility band upper boundary is most predictive |
| 2 | EMA | 13.16% | Short-term moving average carries significant signal |
| 3 | Signal | 11.25% | MACD signal line contributes meaningfully |
| 4 | Histogram | 10.49% | Momentum change rate matters |
| 5 | BB_Lower | 10.25% | Lower band also important (symmetric) |
| ... | ... | ... | ... |
| 10 | VolumeRatio | 5.74% | Volume alone is least predictive |

> **Insight:** Bollinger Bands (Upper, Lower, Middle combined = ~33.8%) and EMA are the dominant signals. Volume ratio matters least for predicting 5-day direction.

---

### 6. 5-Day Direction Prediction

```
5-day direction prediction (today): UP
```

**What this means:** Based on the most recent row of data (today 2026-04-09), the model predicts the price will be **higher 5 trading days from now** compared to today. This is a single binary output (UP=1, DOWN=0).

---

### 7. Chart Saved

```
Chart saved to AAPL_analysis.png
```

The chart has **3 panels** (or 4 if feature importance is included):

```
┌─────────────────────────────────────────────────────┐
│  Panel 1: Price + SMA(20) + EMA(20) + BBands       │
│  - Black line: close price                          │
│  - Blue: SMA  (slower, simple average)             │
│  - Orange: EMA (faster, recent-weighted)            │
│  - Red dashed: BB Upper/Lower (volatility bands)    │
│  - Shaded red area: BB band width                   │
├─────────────────────────────────────────────────────┤
│  Panel 2: RSI (14-day)                             │
│  - Purple line: RSI value                           │
│  - Red dashed at 70: overbought threshold           │
│  - Green dashed at 30: oversold threshold           │
│  - Shaded zones: danger areas                       │
├─────────────────────────────────────────────────────┤
│  Panel 3: Volume Ratio                             │
│  - Blue bars: daily volume / 20-day avg volume     │
│  - Black horizontal line at 1.0: average volume     │
│  - Bars above 1 = above-average volume              │
│  - Bars below 1 = below-average volume             │
└─────────────────────────────────────────────────────┘
```

*(If running single ticker with importance data, a 4th panel shows a horizontal bar chart of feature importances.)*

---

## Comparison Mode (Two Tickers)

When you run `python analyze.py AAPL MSFT`:

1. Each ticker goes through the full pipeline above
2. `compare_indicators()` prints a side-by-side table of latest indicator values
3. `compare_models()` compares accuracy and top features side-by-side
4. `plot_comparison()` saves two charts:
   - `comparison_rsi.png` — overlaid price and RSI for both tickers
   - `comparison_importances.png` — side-by-side feature importance bars

---

## Key Takeaways

| Question | Answer |
|---|---|
| Is the model reliable? | Moderately — 59.6% accuracy is better than chance but far from a trading system guarantee |
| What's the most important indicator? | Bollinger Bands (upper/lower) and EMA |
| What's the least important? | Volume ratio |
| What does RSI at 52.89 mean? | Neutral — not overbought, not oversold |
| What's the current signal? | Model predicts UP for next 5 days |