"""
outcome_tracker.py — fills in future_high / future_low for each logged signal

Run this as a separate process or cron job.
After OUTCOME_CANDLES * 5 minutes, it fetches the OHLC window after each
signal and records the highest high and lowest low, which model.py uses to
generate labels (TP hit or SL hit first).

Usage:
    python outcome_tracker.py
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timezone

FEEDBACK_FILE   = os.path.join("logs", "feedback_log.json")
SYMBOL          = "DOGEUSDT"
INTERVAL        = "5m"
OUTCOME_CANDLES = 24   # look 24 × 5m = 2 hours forward
SLEEP_SECONDS   = 300  # check every 5 minutes


def fetch_klines(symbol: str, start_ms: int, limit: int = 30) -> pd.DataFrame:
    url    = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": INTERVAL,
               "startTime": start_ms, "limit": limit}
    try:
        r    = requests.get(url, params=params, timeout=10)
        data = r.json()
        df   = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df[["high", "low"]] = df[["high", "low"]].astype(float)
        df["timestamp"]     = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"❌ Fetch error: {e}")
        return pd.DataFrame()


def fill_outcomes():
    if not os.path.exists(FEEDBACK_FILE):
        return

    with open(FEEDBACK_FILE, "r") as f:
        try:
            data = json.load(f)
        except Exception:
            return

    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    modified = False

    for rec in data:
        # skip already resolved
        if rec.get("future_high") is not None:
            continue

        # parse signal time
        try:
            sig_time = datetime.strptime(rec["time"], "%Y-%m-%d %H:%M")
            sig_ms   = int(sig_time.replace(tzinfo=timezone.utc).timestamp() * 1000)
        except Exception:
            continue

        # only resolve if enough candles have passed
        elapsed_candles = (now_ms - sig_ms) / (5 * 60 * 1000)
        if elapsed_candles < OUTCOME_CANDLES:
            continue

        # fetch forward window
        df = fetch_klines(SYMBOL, sig_ms + 5 * 60 * 1000, limit=OUTCOME_CANDLES)
        if df.empty:
            continue

        rec["future_high"] = float(df["high"].max())
        rec["future_low"]  = float(df["low"].min())
        modified = True
        print(f"✅ Resolved {rec['signal_id']} — H:{rec['future_high']} L:{rec['future_low']}")

    if modified:
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print("💾 Feedback log updated")


if __name__ == "__main__":
    print("📡 Outcome tracker running...")
    while True:
        fill_outcomes()
        time.sleep(SLEEP_SECONDS)
