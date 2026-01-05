# Google Stock Prediction (GOOG)

Predicts **next-day Google (GOOG) adjusted close price** (regression) and **next-day direction (up/down)** (classification) using historical market data + technical indicators.

> Note: The project objective statement in the notebook mentions “predict the next 30 days of open and close.”
> The current implementation predicts **next-day adjusted close** and **next-day direction**.


---

## Project Overview

This repo loads historical GOOG prices from `Google_Stock_Price.csv`, engineers common technical indicators, then trains:

- **Regression model:** `LinearRegression` to predict **tomorrow’s adjusted close**
- **Classification model:** `RandomForestClassifier` to predict whether **tomorrow’s price goes up (1) or down (0)**

It also includes:
- plots for price trends and indicators
- rolling volatility plot
- a simple “toy” equity curve simulation based on predicted direction

---

## Data

**Input file:** `Google_Stock_Price.csv`  
Columns (from your dataset):
- `date`, `open`, `high`, `low`, `close`, `volume`
- adjusted versions: `adjOpen`, `adjHigh`, `adjLow`, `adjClose`, `adjVolume`
- `divCash`, `splitFactor`, `symbol`

The script converts `date` to datetime, sorts ascending, and checks missing values.

---

## Features Engineered

From `adjClose`, you compute:

### Moving Averages
- `MA_7`, `MA_30`, `MA_90`

### MACD
- `MACD = EMA(12) - EMA(26)`

### RSI (14-day)
- `RSI` using average gains/losses over a rolling 14-day window

### Bollinger Band Midline (20-day)
- `BB_MA20` (you also compute `BB_Upper` / `BB_Lower`)

### Lag Features
- `lag_1`, `lag_3`, `lag_7` (previous adjusted closes)

### Targets
- **Regression target:** `target_price = adjClose.shift(-1)` (tomorrow’s adjClose)
- **Classification target:** `target_direction = (target_price > adjClose).astype(int)`

Final feature set used:
```python
features = [
  'adjClose','MA_7','MA_30','MA_90','MACD','RSI','BB_MA20',
  'lag_1','lag_3','lag_7'
]
