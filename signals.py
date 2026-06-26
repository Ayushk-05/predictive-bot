"""
signals.py — SMC + VSA signal engine with ML filtering

Key fixes over v1:
- ATR-based SL/TP (volatility-adaptive, not hardcoded 0.002)
- SMC score threshold raised: 4/5 instead of 3/5
- Features computed at signal time and stored in signal dict
- feedback_log stores features + entry/tp/sl for model retraining
- HTF trend filter uses EMA crossover (not raw close comparison)
- Signal cooldown: 1 signal per 3 candles max to avoid overtrading
"""

import time
import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

from features import build_features, compute_atr, compute_ema
from model import predict_signal_quality

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL          = "DOGEUSDT"
INTERVAL_5M     = "5m"
INTERVAL_1H     = "1h"
INTERVAL_4H     = "4h"
SIGNAL_THRESHOLD = 0.60      # raised from 0.50
SMC_MIN_SCORE   = 4          # raised from 3
ATR_SL_MULT     = 1.5        # SL = entry ± 1.5 × ATR(5m)
ATR_TP_MULT     = 2.5        # TP = entry ± 2.5 × ATR(5m) → RR ≈ 1.67
SLEEP_SECONDS   = 60
COOLDOWN_CANDLES = 3         # skip signal if last signal was < 3 candles ago

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

os.makedirs("logs", exist_ok=True)
SIGNALS_FILE  = os.path.join("logs", "signals.json")
FEEDBACK_FILE = os.path.join("logs", "feedback_log.json")


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_data(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df[["open", "high", "low", "close", "volume"]] = \
            df[["open", "high", "low", "close", "volume"]].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        print(f"❌ Fetch error ({interval}): {e}")
        return pd.DataFrame()


# ── HTF trend filter ──────────────────────────────────────────────────────────

def htf_trend_bullish(df_4h: pd.DataFrame) -> bool:
    ema20 = compute_ema(df_4h["close"], 20)
    ema50 = compute_ema(df_4h["close"], 50)
    return ema20.iloc[-1] > ema50.iloc[-1]


def htf_trend_bearish(df_4h: pd.DataFrame) -> bool:
    ema20 = compute_ema(df_4h["close"], 20)
    ema50 = compute_ema(df_4h["close"], 50)
    return ema20.iloc[-1] < ema50.iloc[-1]


# ── SMC detection ─────────────────────────────────────────────────────────────

def refined_smc(df: pd.DataFrame, lookback: int = 50, min_impulse: float = 1.5) -> dict | None:
    if len(df) < lookback:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Swing highs/lows
    swing_high = swing_low = None
    for i in range(len(df) - 2, len(df) - lookback, -1):
        if i - 2 < 0 or i + 2 >= len(df):
            continue
        neighbours_h = df["high"].iloc[max(0, i-2):i+3].drop(index=df.index[i])
        neighbours_l = df["low"].iloc[max(0, i-2):i+3].drop(index=df.index[i])
        if swing_high is None and df["high"].iloc[i] > neighbours_h.max():
            swing_high = df["high"].iloc[i]
        if swing_low is None and df["low"].iloc[i] < neighbours_l.min():
            swing_low = df["low"].iloc[i]
        if swing_high and swing_low:
            break

    bos_bull  = bool(swing_high and last["high"] > swing_high)
    bos_bear  = bool(swing_low  and last["low"]  < swing_low)
    choch_bull = bool(swing_low  and prev["close"] < swing_low  and last["close"] > prev["high"])
    choch_bear = bool(swing_high and prev["close"] > swing_high and last["close"] < prev["low"])
    liq_bull  = last["low"]  < df["low"].iloc[-10:].min()  and last["close"] > prev["low"]
    liq_bear  = last["high"] > df["high"].iloc[-10:].max() and last["close"] < prev["high"]
    str_bull  = last["close"] > last["open"] and (last["close"] - last["open"]) > 0.6 * (last["high"] - last["low"])
    str_bear  = last["open"]  > last["close"] and (last["open"] - last["close"]) > 0.6 * (last["high"] - last["low"])

    ob_bull, ob_bear = [], []
    for i in range(len(df) - 4, max(len(df) - lookback, 4), -1):
        if i + 5 >= len(df):
            continue
        c = df.iloc[i]
        nxt = df.iloc[i+1:i+5]
        imp_up   = nxt["high"].max() - c["high"]
        imp_down = c["low"] - nxt["low"].min()
        span = c["high"] - c["low"]
        if c["close"] < c["open"] and imp_up   > span * min_impulse:
            ob_bull.append(i)
        if c["close"] > c["open"] and imp_down > span * min_impulse:
            ob_bear.append(i)

    bull_sigs = {"BOS": bos_bull, "CHoCH": choch_bull, "LiqGrab": liq_bull,
                 "StrongCandle": str_bull, "OrderBlock": bool(ob_bull)}
    bear_sigs = {"BOS": bos_bear, "CHoCH": choch_bear, "LiqGrab": liq_bear,
                 "StrongCandle": str_bear, "OrderBlock": bool(ob_bear)}

    bull_score = sum(bull_sigs.values())
    bear_score = sum(bear_sigs.values())

    if bull_score >= SMC_MIN_SCORE:
        active = [k for k, v in bull_sigs.items() if v]
        return {"type": "bullish", "score": bull_score, "active": active,
                "reason": f"SMC Bullish {bull_score}/5: {', '.join(active)}"}
    if bear_score >= SMC_MIN_SCORE:
        active = [k for k, v in bear_sigs.items() if v]
        return {"type": "bearish", "score": bear_score, "active": active,
                "reason": f"SMC Bearish {bear_score}/5: {', '.join(active)}"}
    return None


# ── VSA detection ─────────────────────────────────────────────────────────────

def enhanced_vsa(df: pd.DataFrame) -> dict | None:
    if len(df) < 10:
        return None
    last    = df.iloc[-1]
    avg_vol = df["volume"].iloc[-10:-1].mean()
    spread  = last["high"] - last["low"]
    body    = abs(last["close"] - last["open"])

    bull_rev = (last["close"] > last["open"]
                and last["close"] > last["high"] - 0.3 * spread
                and body > 0.6 * spread
                and last["volume"] > 1.5 * avg_vol)
    bear_rev = (last["open"] > last["close"]
                and last["close"] < last["low"] + 0.3 * spread
                and body > 0.6 * spread
                and last["volume"] > 1.5 * avg_vol)
    absorb   = (last["volume"] > 2 * avg_vol and body < 0.3 * spread)

    if bull_rev:
        return {"type": "bullish", "reason": "VSA Bullish Reversal"}
    if bear_rev:
        return {"type": "bearish", "reason": "VSA Bearish Reversal"}
    if absorb:
        return {"type": "neutral", "reason": "VSA Absorption"}
    return None


# ── Signal builder ────────────────────────────────────────────────────────────

def build_signal(signal_type: str, reasoning: str,
                 df_5m: pd.DataFrame, atr: float) -> dict:
    entry = df_5m.iloc[-1]["close"]
    ts    = df_5m.iloc[-1]["timestamp"].strftime("%Y-%m-%d %H:%M")
    sid   = f"{signal_type}_{df_5m.iloc[-1]['timestamp'].timestamp():.0f}"

    if signal_type == "bullish":
        sl = round(entry - ATR_SL_MULT * atr, 6)
        tp = round(entry + ATR_TP_MULT * atr, 6)
    else:
        sl = round(entry + ATR_SL_MULT * atr, 6)
        tp = round(entry - ATR_TP_MULT * atr, 6)

    return {
        "signal_id": sid,
        "symbol":    SYMBOL,
        "time":      ts,
        "type":      signal_type,
        "entry":     round(entry, 6),
        "tp":        tp,
        "sl":        sl,
        "rr_ratio":  round(ATR_TP_MULT / ATR_SL_MULT, 2),
        "atr":       round(atr, 6),
        "reasoning": reasoning,
        # these get filled in later by outcome tracker
        "future_high": None,
        "future_low":  None,
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_signal(signal: dict) -> None:
    signals = []
    if os.path.exists(SIGNALS_FILE):
        with open(SIGNALS_FILE, "r") as f:
            try:
                signals = json.load(f)
            except Exception:
                signals = []
    signals.append(signal)
    with open(SIGNALS_FILE, "w") as f:
        json.dump(signals, f, indent=2)


def log_to_feedback(signal: dict, features: dict) -> None:
    """Store signal + features in feedback log for later model retraining."""
    record = {**signal, "features": features}
    data = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            try:
                data = json.load(f)
            except Exception:
                data = []
    data.append(record)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_alert(signal: dict, ml_score: float) -> None:
    emoji  = "🟢" if signal["type"] == "bullish" else "🔴"
    msg = (
        f"{emoji} *{signal['type'].upper()} Signal* {emoji}\n"
        f"Pair:     `{signal['symbol']}`\n"
        f"Entry:    `{signal['entry']}`\n"
        f"TP:       `{signal['tp']}`\n"
        f"SL:       `{signal['sl']}`\n"
        f"RR:       `1:{signal['rr_ratio']}`\n"
        f"ATR:      `{signal['atr']}`\n"
        f"ML Score: `{ml_score:.2f}`\n"
        f"Reason:   {signal['reasoning']}\n"
        f"Time:     {signal['time']}"
    )
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            print("✅ Telegram alert sent")
        else:
            print(f"❌ Telegram error {r.status_code}: {r.text}")
    except Exception as e:
        print(f"❌ Telegram exception: {e}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_once(last_5m_ts, cooldown_counter: int):
    df_5m = fetch_data(SYMBOL, INTERVAL_5M)
    df_1h = fetch_data(SYMBOL, INTERVAL_1H)
    df_4h = fetch_data(SYMBOL, INTERVAL_4H)

    if df_5m.empty or df_1h.empty or df_4h.empty:
        print("⚠️ Data fetch failed")
        return last_5m_ts, cooldown_counter

    current_ts = df_5m.iloc[-1]["timestamp"]
    if last_5m_ts is not None and current_ts <= last_5m_ts:
        print("⏳ No new candle yet")
        return last_5m_ts, cooldown_counter

    # Cooldown check
    if cooldown_counter > 0:
        print(f"⏸ Cooldown: {cooldown_counter} candles remaining")
        return current_ts, cooldown_counter - 1

    # ── Build features ────────────────────────────────────────────────────────
    feats = build_features(df_5m, df_1h, df_4h)
    atr   = feats["atr_5m"]

    # ── Detect signals ────────────────────────────────────────────────────────
    smc = refined_smc(df_1h)
    vsa = enhanced_vsa(df_5m)

    signal_type = None
    reasoning   = ""

    if smc and vsa and smc["type"] == vsa["type"] and smc["type"] != "neutral":
        signal_type = smc["type"]
        reasoning   = f"SMC+VSA Confluence | {smc['reason']} | {vsa['reason']}"
    elif smc and smc["type"] != "neutral":
        signal_type = smc["type"]
        reasoning   = smc["reason"]
    elif vsa and vsa["type"] != "neutral":
        signal_type = vsa["type"]
        reasoning   = vsa["reason"]

    if not signal_type:
        print("🔍 No signal this candle")
        return current_ts, 0

    # ── HTF trend filter ──────────────────────────────────────────────────────
    if signal_type == "bullish" and not htf_trend_bullish(df_4h):
        print("⛔ Bullish signal blocked — bearish HTF trend")
        return current_ts, 0
    if signal_type == "bearish" and not htf_trend_bearish(df_4h):
        print("⛔ Bearish signal blocked — bullish HTF trend")
        return current_ts, 0

    # ── ML filter ─────────────────────────────────────────────────────────────
    ml_score = predict_signal_quality(feats)
    print(f"📊 ML Score: {ml_score:.2f} | Signal: {signal_type} | {reasoning}")

    if ml_score < SIGNAL_THRESHOLD:
        print(f"⚠️ Signal skipped (ML score {ml_score:.2f} < {SIGNAL_THRESHOLD})")
        return current_ts, 0

    # ── Build, save, alert ────────────────────────────────────────────────────
    signal = build_signal(signal_type, reasoning, df_5m, atr)
    signal["features"] = feats          # store features for retraining
    log_to_feedback(signal, feats)
    save_signal(signal)
    send_alert(signal, ml_score)

    return current_ts, COOLDOWN_CANDLES


def main_loop():
    print(f"📡 Starting {SYMBOL} signal engine — SMC+VSA+ML")
    last_5m_ts      = None
    cooldown_counter = 0

    while True:
        last_5m_ts, cooldown_counter = run_once(last_5m_ts, cooldown_counter)
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main_loop()
