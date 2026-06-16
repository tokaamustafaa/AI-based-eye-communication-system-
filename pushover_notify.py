"""
pushover_notify.py — Send a Pushover notification to the caregiver's phone.

Setup:
  1. Create a free account at https://pushover.net
  2. Copy your User Key from the dashboard → paste into PUSHOVER_USER_KEY
  3. Go to https://pushover.net/apps/build → create an app → copy the API Token
     → paste into PUSHOVER_API_TOKEN
  4. Install the requests library if not already installed:
       pip install requests
"""

import threading
import requests

# ── Credentials ────────────────────────────────────────────────────────────────
PUSHOVER_API_TOKEN = ""   # <-- paste your app token here
PUSHOVER_USER_KEY  = ""        # <-- paste your user key here
# ───────────────────────────────────────────────────────────────────────────────

_PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def send_notification(choice: str) -> None:
    """
    Send 'Patient needs: <choice>' to the caregiver's phone via Pushover.
    Runs in a background thread so it never blocks the UI.
    """
    if PUSHOVER_API_TOKEN == "YOUR_APP_API_TOKEN" or PUSHOVER_USER_KEY == "YOUR_USER_KEY":
        print("[Pushover] Credentials not set — skipping notification.")
        return

    def _send():
        try:
            response = requests.post(_PUSHOVER_URL, data={
                "token":   PUSHOVER_API_TOKEN,
                "user":    PUSHOVER_USER_KEY,
                "title":   "Patient Alert",
                "message": f"Patient needs: {choice}",
                "priority": 1,   # high priority — makes a sound even in quiet hours
            }, timeout=5)
            if response.status_code == 200:
                print(f"[Pushover] Notification sent → Patient needs: {choice}")
            else:
                print(f"[Pushover] Failed ({response.status_code}): {response.text}")
        except Exception as exc:
            print(f"[Pushover] Error: {exc}")

    threading.Thread(target=_send, daemon=True).start()
