import streamlit as st
import os
import pandas as pd
from feedback_logger import log_feedback
from model import learn_from_feedback

FEEDBACK_FILE = os.path.join("logs", "feedback_log.json")
SIGNALS_FILE = os.path.join("logs", "signals.json")

st.set_page_config(page_title="ML Bot Feedback", page_icon="🧠")
st.title("📊 ML Signal Feedback")

# Load signals for dropdown if exists
if os.path.exists(SIGNALS_FILE):
    signals_df = pd.read_json(SIGNALS_FILE)
    signal_options = signals_df["signal_id"].tolist()
else:
    signals_df = pd.DataFrame()
    signal_options = []

# If signals exist, show dropdown for signal selection
if signal_options:
    signal_id = st.selectbox("Select Signal ID", options=signal_options)

    # Show TP and SL for selected signal
    selected_signal = signals_df[signals_df["signal_id"] == signal_id]
    if not selected_signal.empty:
        tp_value = selected_signal.iloc[0].get("tp", "N/A")
        sl_value = selected_signal.iloc[0].get("sl", "N/A")
        st.info(f"Target Price (TP): {tp_value} | Stop Loss (SL): {sl_value}")
else:
    # No signals found, allow manual signal ID input
    st.info("No saved signals found. Please enter signal details manually.")
    signal_id = st.text_input("Enter Signal ID (Timestamp or Bot Code)")
    tp_value = st.text_input("Enter Target Price (TP)")
    sl_value = st.text_input("Enter Stop Loss (SL)")

is_valid = st.radio("Was the signal valid?", ["Yes", "No"])
tp_hit = st.radio("Was Target Price hit?", ["Yes", "No"])
sl_hit = st.radio("Was Stop Loss hit?", ["Yes", "No"])
notes = st.text_area("Notes / Comments")

if st.button("Submit Feedback"):
    if not signal_id:
        st.warning("Please enter or select a Signal ID.")
    else:
        notes_safe = notes if notes else ""
        log_feedback(
            signal_id=signal_id,
            valid=is_valid == "Yes",
            tp_hit=tp_hit == "Yes",
            sl_hit=sl_hit == "Yes",
            notes=notes_safe,
            tp=tp_value if 'tp_value' in locals() else None,
            sl=sl_value if 'sl_value' in locals() else None
        )
        st.success("✅ Feedback submitted successfully!")

# Show feedback stats if available
if os.path.exists(FEEDBACK_FILE):
    df = pd.read_json(FEEDBACK_FILE)
    if not df.empty:
        st.subheader("📈 Feedback Stats")
        st.metric("Total", len(df))
        st.metric("Valid %", f"{df['valid'].mean() * 100:.1f}%")
        st.metric("TP Hit %", f"{df['tp_hit'].mean() * 100:.1f}%")
        st.metric("SL Hit %", f"{df['sl_hit'].mean() * 100:.1f}%")

        if st.button("Retrain Now"):
            learn_from_feedback()
            st.success("✅ Model retrained.")
