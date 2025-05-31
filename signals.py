import time
import requests
import pandas as pd
from model import predict_signal_quality
import json
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file

def save_signal(signal, filename="signals.json"):
    signals = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            try:
                signals = json.load(f)
            except Exception:
                signals = []
    signals.append(signal)
    with open(filename, "w") as f:
        json.dump(signals, f, indent=4)

SYMBOL = "DOGEUSDT"
INTERVAL_15M = "5m"
INTERVAL_1H = "15m"
INTERVAL_4H = "1h"
SIGNAL_THRESHOLD = 0.5
SLEEP_SECONDS = 60

# Load credentials from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def fetch_data(symbol, interval, limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close",
            "volume", "close_time", "quote_asset_volume",
            "number_of_trades", "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume", "ignore"
        ])
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        return df
    except Exception as e:
        print(f"❌ Error fetching {interval} data: {e}")
        return pd.DataFrame()

def detect_accumulation_zone(df, threshold=0.01):
    recent = df[-20:]
    high = recent['high'].max()
    low = recent['low'].min()
    range_size = high - low
    price = df['close'].iloc[-1]

    tight_range = range_size / price < threshold
    low_vol = recent['volume'].rolling(5).mean().iloc[-1] < recent['volume'].rolling(5).mean().iloc[0]
    spring_candle = df.iloc[-1]['low'] < low and df.iloc[-1]['close'] > df.iloc[-2]['close']

    if tight_range and low_vol and spring_candle:
        return True, {"range_high": high, "range_low": low}
    return False, {}

def confirm_accumulation_entry(df, zone):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    spring = last['low'] < zone['range_low'] and last['close'] > prev['close']
    wide_spread = (last['high'] - last['low']) > df['high'].rolling(5).mean().iloc[-1] - df['low'].rolling(5).mean().iloc[-1]
    strong_close = last['close'] > last['open']
    entry_ok = spring and strong_close and wide_spread

    return entry_ok

def refined_smc(df, lookback=50, min_impulse=1.5):
    if len(df) < lookback:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    swing_high = swing_low = None
    for i in range(len(df)-2, len(df)-lookback, -1):
        if i-2 < 0 or i+2 >= len(df):
            continue
        high = df['high'].iloc[i]
        low = df['low'].iloc[i]
        if high > df['high'].iloc[i-2:i+3].drop(i).max():
            swing_high = high
            break
    for i in range(len(df)-2, len(df)-lookback, -1):
        if i-2 < 0 or i+2 >= len(df):
            continue
        low = df['low'].iloc[i]
        if low < df['low'].iloc[i-2:i+3].drop(i).min():
            swing_low = low
            break

    bos_bull = swing_high and last['high'] > swing_high
    bos_bear = swing_low and last['low'] < swing_low
    choch_bull = prev['close'] < swing_low and last['close'] > prev['high']
    choch_bear = prev['close'] > swing_high and last['close'] < prev['low']

    liquidity_grab_bull = last['low'] < df['low'].iloc[-10:].min() and last['close'] > prev['low']
    liquidity_grab_bear = last['high'] > df['high'].iloc[-10:].max() and last['close'] < prev['high']
    strong_bull = last['close'] > last['open'] and (last['close'] - last['open']) > 0.6 * (last['high'] - last['low'])
    strong_bear = last['open'] > last['close'] and (last['open'] - last['close']) > 0.6 * (last['high'] - last['low'])

    order_blocks_bull = []
    order_blocks_bear = []
    for i in range(len(df)-4, len(df)-lookback, -1):
        if i+5 >= len(df):
            continue
        candle = df.iloc[i]
        next_candles = df.iloc[i+1:i+5]
        impulse_up = next_candles['high'].max() - candle['high']
        impulse_down = candle['low'] - next_candles['low'].min()

        if candle['close'] < candle['open'] and impulse_up > (candle['high'] - candle['low']) * min_impulse:
            order_blocks_bull.append(i)
        if candle['close'] > candle['open'] and impulse_down > (candle['high'] - candle['low']) * min_impulse:
            order_blocks_bear.append(i)

    bull_signals = {
        "BOS": bos_bull,
        "CHoCH": choch_bull,
        "Liquidity Grab": liquidity_grab_bull,
        "Strong Bull Candle": strong_bull,
        "Order Block": bool(order_blocks_bull),
    }
    bear_signals = {
        "BOS": bos_bear,
        "CHoCH": choch_bear,
        "Liquidity Grab": liquidity_grab_bear,
        "Strong Bear Candle": strong_bear,
        "Order Block": bool(order_blocks_bear),
    }

    bull_score = sum(bull_signals.values())
    bear_score = sum(bear_signals.values())

    if bull_score >= 3:
        active_bull = [k for k, v in bull_signals.items() if v]
        return {
            "type": "bullish",
            "score": bull_score,
            "active_signals": active_bull,
            "reason": f"Bullish Score {bull_score}/5 - Signals: {', '.join(active_bull)}",
            "description": f"Bullish OBs detected at: {order_blocks_bull}",
            "order_blocks": order_blocks_bull
        }
    elif bear_score >= 3:
        active_bear = [k for k, v in bear_signals.items() if v]
        return {
            "type": "bearish",
            "score": bear_score,
            "active_signals": active_bear,
            "reason": f"Bearish Score {bear_score}/5 - Signals: {', '.join(active_bear)}",
            "description": f"Bearish OBs detected at: {order_blocks_bear}",
            "order_blocks": order_blocks_bear
        }
    return None

def enhanced_vsa(df):
    if len(df) < 10:
        return None

    last = df.iloc[-1]
    avg_vol = df['volume'].iloc[-10:-1].mean()
    spread = last['high'] - last['low']
    body = abs(last['close'] - last['open'])

    bullish_reversal = (
        last['close'] > last['open'] and
        last['close'] > last['high'] - 0.3 * spread and
        body > 0.6 * spread and
        last['volume'] > 1.5 * avg_vol
    )
    bearish_reversal = (
        last['open'] > last['close'] and
        last['close'] < last['low'] + 0.3 * spread and
        body > 0.6 * spread and
        last['volume'] > 1.5 * avg_vol
    )
    absorption = (
        last['volume'] > 2 * avg_vol and
        body < 0.3 * spread
    )

    if bullish_reversal:
        return {
            "type": "bullish",
            "reason": "VSA Bullish Reversal",
            "description": (
                "Accumulation detected - strong buying pressure on high volume."
                "Price closed near the high with a large body and wide spread."
                "Potential reversal zone indicating buyers stepping in."
                "Watch for follow-through confirmation."
            )
        }
    elif bearish_reversal:
        return {
            "type": "bearish",
            "reason": "VSA Bearish Reversal",
            "description": (
                "Distribution detected - strong selling pressure on high volume."
                "Price closed near the low with a large body and wide spread."

                "Potential reversal zone indicating sellers dominating."
                "Beware of fake breakouts or trend changes."
            )
        }
    elif absorption:
        return {
            "type": "neutral",
            "reason": "Absorption or Trap Detected (VSA)",
            "description": (
                "Absorption volume detected - large volume with a small candle body."
                "Aggressive buying or selling absorbing opposing pressure."
                "Possible fake breakout or trap setup."
                "Price may be preparing to reverse or continue after this consolidation."
            )
        }
    return None

def run_once(last_15m_timestamp, last_1h_timestamp):
    current_15m_timestamp = last_15m_timestamp
    current_1h_timestamp = last_1h_timestamp

    df_15m = fetch_data(SYMBOL, INTERVAL_15M)
    df_1h = fetch_data(SYMBOL, INTERVAL_1H)
    df_4h = fetch_data(SYMBOL, INTERVAL_4H)

    if df_15m.empty or df_1h.empty or df_4h.empty:
        print("⚠️ Data fetch failed, skipping this cycle")
        return current_15m_timestamp, current_1h_timestamp

    current_15m_timestamp = df_15m.iloc[-1]["timestamp"]
    current_1h_timestamp = df_1h.iloc[-1]["timestamp"]

    new_15m = last_15m_timestamp is None or current_15m_timestamp > last_15m_timestamp
    new_1h = last_1h_timestamp is None or current_1h_timestamp > last_1h_timestamp

    if not new_15m and not new_1h:
        print("⏳ No new candles yet.")
        return current_15m_timestamp, current_1h_timestamp

    # 1. Accumulation zone entry logic
    accum_zone_detected, zone = detect_accumulation_zone(df_15m)
    if accum_zone_detected and confirm_accumulation_entry(df_15m, zone):
        entry_price = df_15m.iloc[-1]['close']
        signal = {
            "symbol": SYMBOL,
            "time": current_15m_timestamp.strftime("%Y-%m-%d %H:%M"),
            "signal_id": "accum_in_" + str(current_15m_timestamp),
            "entry": round(entry_price, 4),
            "type": "bullish",
            "sl": round(zone['range_low'] - 0.001, 4),
            "tp": round(zone['range_high'], 4),
            "tp_hit": 0,
            "sl_hit": 0,
            "reasoning": "Inside accumulation: spring + bullish confirmation",
            "description": f"Accumulation zone: {round(zone['range_low'], 4)} - {round(zone['range_high'], 4)}",
        }
        prob = predict_signal_quality(signal)
        print(f"📊 ML Confidence (accum.): {prob:.2f}")
        if prob >= SIGNAL_THRESHOLD:
            send_alert(signal)
            save_signal(signal, os.path.join("logs", "signals.json"))
        else:
            print("⚠️ Accumulation signal skipped (low ML score)")

    # 2. Original SMC + VSA logic (if implemented)
    smc = refined_smc(df_1h) if new_1h else None
    vsa = enhanced_vsa(df_15m) if new_15m else None

    if smc or vsa:
        reasoning = ""
        signal_type = ""
        entry_price = 0.0

        if smc and vsa and smc['type'] == vsa['type']:
            reasoning = f"SMC + VSA Confluence: {smc['reason']} and {vsa['reason']}"
            signal_type = smc['type']
            entry_price = df_15m.iloc[-1]['close']
        elif smc:
            reasoning = f"SMC only: {smc['reason']}"
            signal_type = smc['type']
            entry_price = df_1h.iloc[-1]['close']
        elif vsa:
            reasoning = f"VSA only: {vsa['reason']}"
            signal_type = vsa['type']
            entry_price = df_15m.iloc[-1]['close']

        if df_4h['close'].iloc[-1] < df_4h['close'].iloc[-3] and signal_type == "bullish":
            print("⚠️ Bearish HTF trend - skipping bullish signal")
            return current_15m_timestamp, current_1h_timestamp

        signal = {
            "symbol": SYMBOL,
            "time": current_15m_timestamp.strftime("%Y-%m-%d %H:%M"),
            "signal_id": str(current_15m_timestamp),
            "entry": round(entry_price, 4),
            "type": signal_type,
            "sl": round(entry_price - 0.002, 4) if signal_type == "bullish" else round(entry_price + 0.002, 4),
            "tp": round(entry_price + 0.004, 4) if signal_type == "bullish" else round(entry_price - 0.004, 4),
            "tp_hit": 0,
            "sl_hit": 0,
            "reasoning": reasoning,
            "description": smc['description'] if smc and smc['type'] == signal_type else vsa['description'] if vsa and vsa['type'] == signal_type else "",
        }

        prob = predict_signal_quality(signal)
        print(f"📊 ML Confidence: {prob:.2f}")

        if prob >= SIGNAL_THRESHOLD:
            send_alert(signal)
            save_signal(signal, os.path.join("logs", "signals.json"))
        else:
            print("⚠️ Signal skipped due to low ML score")

    return current_15m_timestamp, current_1h_timestamp

def send_alert(signal):
    message = f"""
🚨 *Trade Signal Alert* 🚨

coin: {signal['symbol']}
Type: {signal['type'].capitalize()}
Description: {signal['description']}
Entry: {signal['entry']}
TP: {signal['tp']}
SL: {signal['sl']}
Reason: {signal['reasoning']}
Time: {signal['time']}
id: {signal['signal_id']}
"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("✅ Signal sent to Telegram")
        else:
            print(f"❌ Telegram error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Exception while sending Telegram alert: {e}")

def main_loop():
    print("📡 Starting DOGEUSDT SMC + VSA observer with ML filtering...")
    last_15m_timestamp = None
    last_1h_timestamp = None
    while True:
        last_15m_timestamp, last_1h_timestamp = run_once(last_15m_timestamp, last_1h_timestamp)
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main_loop()
