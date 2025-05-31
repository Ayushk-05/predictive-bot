import json
import os
from datetime import datetime

FEEDBACK_FILE = os.path.join("logs", "feedback_log.json")

def log_feedback(signal_id, valid, tp_hit, sl_hit, notes, tp,sl):
    feedback = {
        "signal_id": signal_id,
        "valid": valid,
        "tp_hit": tp_hit,
        "sl_hit": sl_hit,
        "notes": notes,
        "tp": tp,
        "sl": sl,
        "timestamp": datetime.now().isoformat()
    }
    data = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    data.append(feedback)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)
