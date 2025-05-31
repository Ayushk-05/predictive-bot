import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

FEEDBACK_FILE = os.path.join("logs", "feedback_log.json")
MODEL_FILE = os.path.join("logs", "feedback_model.pkl")

def learn_from_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        print("No feedback yet.")
        return

    with open(FEEDBACK_FILE, "r") as f:
        data = json.load(f)

    if len(data) < 4:
        print(f"Only {len(data)} entries. Waiting for 4+ to train.")
        return

    df = pd.DataFrame(data)
    df['tp_hit'] = df['tp_hit'].astype(int)
    df['sl_hit'] = df['sl_hit'].astype(int)
    df['valid'] = df['valid'].astype(int)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['signal_code'] = LabelEncoder().fit_transform(df['signal_id'])

    X = df[['tp_hit', 'sl_hit', 'hour', 'signal_code']]
    y = df['valid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    print("✅ Model trained and saved.")
    print(classification_report(y_test, model.predict(X_test)))

def predict_signal_quality(signal):
    try:
        model = joblib.load(MODEL_FILE)
    except:
        return 1.0  # fallback prediction if model is missing

    hour = pd.to_datetime(signal["time"]).hour
    code = hash(signal["signal_id"]) % 100
    x = [[int(signal['tp_hit']), int(signal['sl_hit']), hour, code]]
    return model.predict_proba(x)[0][1]
