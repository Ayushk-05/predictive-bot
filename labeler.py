"""
labeler.py  —  Triple Barrier labeling (Lopez de Prado style)

For each candle that has a qualifying setup (OB strength >= MIN_OB_STRENGTH),
we look forward LABEL_FORWARD_BARS candles and check:

  Did price hit TP (entry + 2×ATR) before SL (entry - 1×ATR)?
    → label = 1  (win)
  Did price hit SL first?
    → label = 0  (loss)
  Neither hit in the window?
    → label = 0  (treat as loss / skip — controlled by TIMEOUT_IS_LOSS)

This is the "ground truth" your model learns from.
"""

import pandas as pd
import numpy as np
from config import (LABEL_FORWARD_BARS, LABEL_TP_ATR_MULT,
                    LABEL_SL_ATR_MULT, MIN_OB_STRENGTH)
from smc_engine import _atr, FEATURE_COLS


TIMEOUT_IS_LOSS = True   # if neither barrier hit → count as loss


def label_setups(df: pd.DataFrame,
                 direction: str = "bull") -> pd.DataFrame:
    """
    direction: 'bull' (long setups from bullish OBs)
               'bear' (short setups from bearish OBs)

    Returns df with extra columns:
        label       — 1 = win, 0 = loss, -1 = not a setup (filtered out later)
        entry_price
        tp_price
        sl_price
        bars_to_outcome
    """
    df   = df.copy()
    a    = _atr(df)

    strength_col = f"ob_{direction}_strength"
    if strength_col not in df.columns:
        raise ValueError(f"Column {strength_col} not found. Run smc_engine.build_features first.")

    df["label"]          = -1
    df["entry_price"]    = np.nan
    df["tp_price"]       = np.nan
    df["sl_price"]       = np.nan
    df["bars_to_outcome"]= np.nan

    for i in range(len(df) - LABEL_FORWARD_BARS):
        strength = df[strength_col].iloc[i]
        if pd.isna(strength) or strength < MIN_OB_STRENGTH:
            continue

        entry = df["close"].iloc[i]
        atr_v = a.iloc[i]
        if pd.isna(atr_v) or atr_v == 0:
            continue

        if direction == "bull":
            tp = entry + LABEL_TP_ATR_MULT * atr_v
            sl = entry - LABEL_SL_ATR_MULT  * atr_v
        else:
            tp = entry - LABEL_TP_ATR_MULT * atr_v
            sl = entry + LABEL_SL_ATR_MULT  * atr_v

        df.at[df.index[i], "entry_price"] = entry
        df.at[df.index[i], "tp_price"]    = tp
        df.at[df.index[i], "sl_price"]    = sl

        outcome = -1   # default: timeout
        for j in range(i+1, i+1+LABEL_FORWARD_BARS):
            high = df["high"].iloc[j]
            low  = df["low"].iloc[j]

            if direction == "bull":
                if high >= tp:
                    outcome = 1; break
                if low  <= sl:
                    outcome = 0; break
            else:
                if low  <= tp:
                    outcome = 1; break
                if high >= sl:
                    outcome = 0; break

        if outcome == -1:
            outcome = 0 if TIMEOUT_IS_LOSS else -1

        df.at[df.index[i], "label"]          = outcome
        df.at[df.index[i], "bars_to_outcome"]= j - i if outcome >= 0 else LABEL_FORWARD_BARS

    return df


def prepare_dataset(df: pd.DataFrame,
                    direction: str = "bull") -> tuple:
    """
    Labels the df, drops rows with label == -1 (no setup),
    returns (X, y) ready for model training.
    """
    df = label_setups(df, direction)
    df = df[df["label"] >= 0].copy()

    # keep only feature columns that exist
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[cols].fillna(0)
    y = df["label"].astype(int)

    print(f"  Dataset: {len(X)} setups  |  wins: {y.sum()}  losses: {(y==0).sum()}")
    print(f"  Win rate in training data: {y.mean()*100:.1f}%")

    return X, y, df
