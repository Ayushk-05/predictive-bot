# Predictive Bot — SMC + VSA + LightGBM Signal Engine

Real-time crypto trading signal generator for DOGEUSDT (Binance) combining Smart Money Concepts (SMC), Volume Spread Analysis (VSA), and a LightGBM ML filter.

## Architecture

```
signals.py          → detects SMC + VSA setups, applies ML filter, sends Telegram alert
features.py         → computes 21 market features (RSI, ATR, ADX, EMA, volume ratio etc.)
model.py            → trains LightGBM on resolved signals, generates SHAP plot
outcome_tracker.py  → fills future_high/future_low in feedback log after 2h
feedback_logger.py  → manual feedback logging utility
```

## Signal Logic

1. **SMC detection** (1h candles) — BOS, CHoCH, Liquidity Grab, Order Block, Strong Candle. Requires 4/5 signals.
2. **VSA detection** (5m candles) — Bullish/Bearish reversal, Absorption.
3. **HTF filter** — 4h EMA20 > EMA50 required for longs; reversed for shorts.
4. **ML filter** — LightGBM trained on historical signal outcomes. Only fires if probability ≥ 0.60.
5. **ATR-based SL/TP** — SL = 1.5×ATR, TP = 2.5×ATR → RR ≈ 1:1.67

## Features (21 total)

| Feature | Timeframe | Why |
|---|---|---|
| RSI | 5m, 1h, 4h | Momentum & overbought/oversold |
| ATR | 5m, 1h | Volatility for SL/TP sizing |
| ADX | 5m, 1h | Trend strength |
| Volume Ratio | 5m, 1h | Volume vs 20-period average |
| BB Width | 5m | Volatility squeeze detection |
| EMA Diff | 5m | Trend direction (EMA20 vs EMA50) |
| Body Ratio | 5m | Candle conviction |
| Close Position | 5m | Where close is in the candle range |
| HTF Trend | 4h | EMA crossover bias |
| Session | — | London / NY / Asia |
| Hour, Day of Week | — | Time-of-day patterns |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add TELEGRAM_TOKEN and TELEGRAM_CHAT_ID
```

Run the signal engine:
```bash
python signals.py
```

Run the outcome tracker (separate terminal):
```bash
python outcome_tracker.py
```

After collecting 30+ resolved signals, retrain the model:
```bash
python model.py
```

## Requirements

```
lightgbm
scikit-learn
pandas
numpy
requests
python-dotenv
joblib
shap
```

## Disclaimer

This is an educational project. Not financial advice. Do not risk money you cannot afford to lose.
