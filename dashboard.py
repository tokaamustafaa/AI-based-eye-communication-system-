"""
dashboard.py — Web dashboard for caregiver.
Run: python dashboard.py
Open browser: http://localhost:5000
"""

from functools import wraps
from flask import Flask, render_template_string, jsonify, request, Response
import json
import os
import datetime

app = Flask(__name__)
LOG_FILE = "medical_log.json"

DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "")
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS", "")


def _require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.authorization
        if (not auth
                or auth.username != DASHBOARD_USER
                or auth.password != DASHBOARD_PASS):
            return Response(
                "Authentication required.",
                401,
                {"WWW-Authenticate": 'Basic realm="Patient Dashboard"'},
            )
        return f(*args, **kwargs)
    return wrapper

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Patient Dashboard</title>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="5">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0d1117; color: #e6edf3; font-family: Arial, sans-serif; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 20px; font-size: 28px; }

        .stats { display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }
        .stat-card {
            background: #161b22; border: 1px solid #30363d;
            border-radius: 12px; padding: 16px 24px; min-width: 160px;
        }
        .stat-card .num { font-size: 36px; font-weight: bold; color: #58a6ff; }
        .stat-card .label { color: #8b949e; font-size: 14px; margin-top: 4px; }

        .urgent-banner {
            background: #3d1a1a; border: 2px solid #f85149;
            border-radius: 12px; padding: 16px 24px; margin-bottom: 24px;
            display: none;
        }
        .urgent-banner.show { display: block; }
        .urgent-banner h2 { color: #f85149; font-size: 22px; }
        .urgent-banner p  { color: #ffa198; margin-top: 8px; }

        table { width: 100%; border-collapse: collapse; background: #161b22; border-radius: 12px; overflow: hidden; }
        th { background: #21262d; padding: 12px 16px; text-align: left; color: #8b949e; font-size: 14px; }
        td { padding: 12px 16px; border-top: 1px solid #21262d; }
        tr:hover td { background: #1c2128; }

        .badge {
            padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: bold;
        }
        .badge-urgent  { background: #3d1a1a; color: #f85149; }
        .badge-normal  { background: #1a2d1a; color: #3fb950; }

        .item-pain, .item-call   { color: #f85149; font-weight: bold; }
        .item-toilet, .item-food, .item-water { color: #d29922; font-weight: bold; }
        .item-yes, .item-no, .item-sleep, .item-light { color: #3fb950; }

        .refresh { color: #8b949e; font-size: 13px; margin-bottom: 16px; }
    </style>
</head>
<body>
    <h1>🏥 Patient Dashboard</h1>
    <p class="refresh">Auto-refreshes every 5 seconds — Last update: {{ now }}</p>

    {% if urgent %}
    <div class="urgent-banner show">
        <h2>🚨 URGENT — Patient needs attention!</h2>
        <p>Last urgent request: <strong>{{ urgent.item }}</strong> at {{ urgent.time }}</p>
    </div>
    {% endif %}

    <div class="stats">
        <div class="stat-card">
            <div class="num">{{ total }}</div>
            <div class="label">Total requests today</div>
        </div>
        <div class="stat-card">
            <div class="num" style="color:#f85149">{{ urgent_count }}</div>
            <div class="label">Urgent requests</div>
        </div>
        <div class="stat-card">
            <div class="num" style="color:#3fb950">{{ last_time }}</div>
            <div class="label">Last activity</div>
        </div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Time</th>
                <th>Request</th>
                <th>Priority</th>
            </tr>
        </thead>
        <tbody>
            {% for log in logs %}
            <tr>
                <td>{{ log.time }}</td>
                <td class="item-{{ log.item.lower() }}">{{ log.item }}</td>
                <td>
                    {% if log.urgent %}
                    <span class="badge badge-urgent">🚨 URGENT</span>
                    {% else %}
                    <span class="badge badge-normal">Normal</span>
                    {% endif %}
                </td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""

@app.route("/")
@_require_auth
def index():
    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                logs = json.load(f)
        except:
            logs = []

    # فلتر النهارده بس
    today = datetime.date.today().strftime("%Y-%m-%d")
    today_logs = [l for l in logs if l["time"].startswith(today)]
    today_logs.reverse()  # الأحدث فوق

    urgent      = next((l for l in today_logs if l["urgent"]), None)
    urgent_count = sum(1 for l in today_logs if l["urgent"])
    last_time   = today_logs[0]["time"].split(" ")[1][:5] if today_logs else "—"

    return render_template_string(HTML,
        logs         = today_logs,
        total        = len(today_logs),
        urgent       = urgent,
        urgent_count = urgent_count,
        last_time    = last_time,
        now          = datetime.datetime.now().strftime("%H:%M:%S"),
    )

@app.route("/api/logs")
@_require_auth
def api_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return jsonify(json.load(f))
    return jsonify([])

if __name__ == "__main__":
    print("Dashboard running at http://localhost:5000")
    print(f"Login: user={DASHBOARD_USER}  "
          f"(set DASHBOARD_USER / DASHBOARD_PASS env vars to change)")
    app.run(host="0.0.0.0", port=5000, debug=False)
