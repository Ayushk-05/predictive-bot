from flask import Flask
from threading import Thread
import signals  # your existing signals.py module

app = Flask(__name__)

@app.route('/')
def home():
    return "SMC + VSA Signals Bot is running!"

def run_bot():
    signals.main_loop()

if __name__ == '__main__':
    # Start the signals bot in a background thread
    bot_thread = Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    # Start Flask server (Render will look for this)
    app.run(host='0.0.0.0', port=10000)
