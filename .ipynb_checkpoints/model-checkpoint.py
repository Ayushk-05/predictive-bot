"""
model.py — LightGBM signal quality classifier

Key fixes over v1:
- No data leakage: tp_hit/sl_hit removed as input features
- Time-series split (no random shuffle) — preserves temporal order
- 21 real market features via features.py
- LightGBM instead of RandomForest
- SHAP feature importance plot saved to logs/
- Fallback: returns 0.5 if model missing (neutral, not always-pass)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from features import FEATURE_COLUMNS

FEEDBACK_FILE = os.path.join("logs", "feedback_log.json")
MODEL_FILE    = os.path.join("logs", "lgbm_model.pkl")
SCALER_FILE   = os.path.join("logs", "scaler.pkl")
SHAP_FILE     = os.path.join("logs", "shap_summary.png")

os.makedirs("logs", exist_ok=True)


# ── Label generation ──────────────────────────────────────────────────────────

def make_label(entry: float, tp: float, sl: float,
               future_high: float, future_low: float) -> int:
    """
    1 = TP hit before SL (good signal)
    0 = SL hit first or neither (bad signal)
    Uses future OHLC data recorded after signal — no lookahead at signal time.
    """
    tp_dist = abs(tp - entry)
    sl_dist = abs(sl - entry)
    if future_high >= tp and future_low <= sl:
        # both hit — whichever is closer to entry wins
        return 1 if tp_dist <= sl_dist else 0
    if future_high >= tp:
        return 1
    if future_low <= sl:
        return 0
    return 0  # neither hit = treat as loss (conservative)


# ── Training ──────────────────────────────────────────────────────────────────

def learn_from_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        print("No feedback log found.")
        return

    with open(FEEDBACK_FILE, "r") as f:
        data = json.load(f)

    # Filter: only records that have resolved (future_high/future_low filled in)
    resolved = [d for d in data if d.get("future_high") and d.get("future_low")]

    if len(resolved) < 30:
        print(f"Only {len(resolved)} resolved signals. Need 30+ to train.")
        return

    rows = []
    labels = []
    for rec in resolved:
        feats = rec.get("features")
        if not feats:
            continue
        row = [feats.get(col, np.nan) for col in FEATURE_COLUMNS]
        if any(np.isnan(v) for v in row):
            continue
        label = make_label(
            rec["entry"], rec["tp"], rec["sl"],
            rec["future_high"], rec["future_low"]
        )
        rows.append(row)
        labels.append(label)

    if len(rows) < 30:
        print("Not enough complete feature rows.")
        return

    X = np.array(rows)
    y = np.array(labels)

    # ── Time-series split (NO shuffle) ───────────────────────────────────────
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── LightGBM ─────────────────────────────────────────────────────────────
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print("✅ Model trained and saved.")

    # ── Evaluation ───────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred))
    if len(np.unique(y_test)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # ── SHAP ─────────────────────────────────────────────────────────────────
    if SHAP_AVAILABLE:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            vals = shap_values[1] if isinstance(shap_values, list) else shap_values
            shap.summary_plot(vals, X_test, feature_names=FEATURE_COLUMNS, show=False)
            plt.tight_layout()
            plt.savefig(SHAP_FILE, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"✅ SHAP plot saved → {SHAP_FILE}")
        except Exception as e:
            print(f"SHAP plot failed: {e}")
    else:
        print("⚠️  shap not installed — run: pip install shap")

        # Manual feature importance fallback
        importances = model.feature_importances_
        ranked = sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1])
        print("\nTop 10 features by importance:")
        for feat, imp in ranked[:10]:
            print(f"  {feat:<30} {imp:.1f}")


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_signal_quality(features: dict) -> float:
    """
    Takes a feature dict (from features.build_features) and returns
    probability [0, 1] that the signal is high quality.
    Returns 0.5 (neutral) if model not trained yet.
    """
    if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
        return 0.5  # neutral fallback, not always-pass

    try:
        model  = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        row = np.array([[features.get(col, np.nan) for col in FEATURE_COLUMNS]])
        row = scaler.transform(row)
        return float(model.predict_proba(row)[0][1])
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        return 0.5


if __name__ == "__main__":
    learn_from_feedback()