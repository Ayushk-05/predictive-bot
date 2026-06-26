import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = compute_atr(df, period)
    plus_di = 100 * plus_dm.ewm(com=period - 1, min_periods=period).mean() / (atr + 1e-9)
    minus_di = 100 * minus_dm.ewm(com=period - 1, min_periods=period).mean() / (atr + 1e-9)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    return dx.ewm(com=period - 1, min_periods=period).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, min_periods=period).mean()


def compute_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    avg_vol = df["volume"].rolling(period).mean()
    return df["volume"] / (avg_vol + 1e-9)


def compute_bb_width(df: pd.DataFrame, period: int = 20) -> pd.Series:
    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (upper - lower) / (ma + 1e-9)


def build_features(df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> dict:
    """
    Build a flat feature dict from the latest candle across timeframes.
    All features are computed at signal time — zero lookahead.
    """
    features = {}

    # ── 5m features ──────────────────────────────────────────────────────────
    rsi_5m = compute_rsi(df_5m["close"])
    atr_5m = compute_atr(df_5m)
    adx_5m = compute_adx(df_5m)
    vol_ratio_5m = compute_volume_ratio(df_5m)
    bb_width_5m = compute_bb_width(df_5m)
    ema20_5m = compute_ema(df_5m["close"], 20)
    ema50_5m = compute_ema(df_5m["close"], 50)

    last5 = df_5m.iloc[-1]
    spread_5m = (last5["high"] - last5["low"]) / (last5["close"] + 1e-9)
    body_ratio_5m = abs(last5["close"] - last5["open"]) / (last5["high"] - last5["low"] + 1e-9)
    close_position_5m = (last5["close"] - last5["low"]) / (last5["high"] - last5["low"] + 1e-9)

    features["rsi_5m"] = rsi_5m.iloc[-1]
    features["atr_5m"] = atr_5m.iloc[-1]
    features["adx_5m"] = adx_5m.iloc[-1]
    features["vol_ratio_5m"] = vol_ratio_5m.iloc[-1]
    features["bb_width_5m"] = bb_width_5m.iloc[-1]
    features["spread_5m"] = spread_5m
    features["body_ratio_5m"] = body_ratio_5m
    features["close_position_5m"] = close_position_5m
    features["ema_diff_5m"] = (ema20_5m.iloc[-1] - ema50_5m.iloc[-1]) / (last5["close"] + 1e-9)
    features["price_vs_ema20_5m"] = (last5["close"] - ema20_5m.iloc[-1]) / (last5["close"] + 1e-9)

    # ── 1h features ──────────────────────────────────────────────────────────
    rsi_1h = compute_rsi(df_1h["close"])
    atr_1h = compute_atr(df_1h)
    adx_1h = compute_adx(df_1h)
    vol_ratio_1h = compute_volume_ratio(df_1h)
    ema20_1h = compute_ema(df_1h["close"], 20)

    last1h = df_1h.iloc[-1]
    features["rsi_1h"] = rsi_1h.iloc[-1]
    features["atr_1h"] = atr_1h.iloc[-1]
    features["adx_1h"] = adx_1h.iloc[-1]
    features["vol_ratio_1h"] = vol_ratio_1h.iloc[-1]
    features["price_vs_ema20_1h"] = (last1h["close"] - ema20_1h.iloc[-1]) / (last1h["close"] + 1e-9)

    # ── 4h trend features ────────────────────────────────────────────────────
    ema20_4h = compute_ema(df_4h["close"], 20)
    ema50_4h = compute_ema(df_4h["close"], 50)
    last4h = df_4h.iloc[-1]

    features["htf_trend"] = 1 if ema20_4h.iloc[-1] > ema50_4h.iloc[-1] else -1
    features["price_vs_ema20_4h"] = (last4h["close"] - ema20_4h.iloc[-1]) / (last4h["close"] + 1e-9)
    features["rsi_4h"] = compute_rsi(df_4h["close"]).iloc[-1]

    # ── Time features ────────────────────────────────────────────────────────
    ts = df_5m.iloc[-1]["timestamp"]
    features["hour"] = ts.hour
    features["day_of_week"] = ts.dayofweek

    # ── Session feature (crypto sessions matter) ─────────────────────────────
    hour = ts.hour
    if 8 <= hour < 16:
        features["session"] = 0   # London
    elif 13 <= hour < 22:
        features["session"] = 1   # New York
    else:
        features["session"] = 2   # Asia

    return features


FEATURE_COLUMNS = [
    "rsi_5m", "atr_5m", "adx_5m", "vol_ratio_5m", "bb_width_5m",
    "spread_5m", "body_ratio_5m", "close_position_5m", "ema_diff_5m",
    "price_vs_ema20_5m", "rsi_1h", "atr_1h", "adx_1h", "vol_ratio_1h",
    "price_vs_ema20_1h", "htf_trend", "price_vs_ema20_4h", "rsi_4h",
    "hour", "day_of_week", "session",
]
