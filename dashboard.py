#!/usr/bin/env python3
"""
TV Watch Tracker - Web Dashboard

視聴データをグラフで可視化するWebダッシュボード
"""
import os
import csv
import json
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, render_template_string, jsonify

app = Flask(__name__)

# 設定読み込み
CONFIG_PATH = os.path.expanduser("~/config.json")

def load_config():
    defaults = {
        "log_path": "~/tv_watch_log.csv",
        "interval_sec": 5,
        "target_names": ["mio", "yu", "tsubasa"],
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_config = json.load(f)
            defaults.update(user_config)
    return defaults

def load_log_data(days=7):
    """過去N日間のログデータを読み込む"""
    config = load_config()
    log_path = os.path.expanduser(config["log_path"])
    interval_sec = config["interval_sec"]
    target_names = config["target_names"]

    cutoff = datetime.now() - timedelta(days=days)
    daily_minutes = defaultdict(lambda: defaultdict(float))
    recent_entries = []

    if not os.path.exists(log_path):
        return daily_minutes, recent_entries

    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                name = row["name"]

                if ts < cutoff:
                    continue

                if name in target_names:
                    date_str = ts.strftime("%Y-%m-%d")
                    daily_minutes[date_str][name] += interval_sec / 60.0

                # 直近のエントリを保持
                recent_entries.append({
                    "timestamp": row["timestamp"],
                    "name": name
                })
            except (ValueError, KeyError):
                continue

    # 直近50件のみ
    recent_entries = recent_entries[-50:][::-1]

    return daily_minutes, recent_entries

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TV Watch Tracker</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: #00d4ff;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .card {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .card h2 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #fff;
        }
        .stat-label {
            color: #888;
            margin-top: 5px;
        }
        .chart-container {
            position: relative;
            height: 300px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        th { color: #00d4ff; }
        .name-mio { color: #ff6b6b; }
        .name-yu { color: #4ecdc4; }
        .name-tsubasa { color: #ffe66d; }
        .name-unknown { color: #888; }
        .name-none { color: #444; }
        .refresh-btn {
            display: block;
            margin: 20px auto;
            padding: 10px 30px;
            background: #00d4ff;
            color: #1a1a2e;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
        }
        .refresh-btn:hover { background: #00b8e6; }
    </style>
</head>
<body>
    <div class="container">
        <h1>TV Watch Tracker</h1>

        <div class="grid">
            <div class="card">
                <h2>Today's Total</h2>
                <div class="stat-value" id="today-total">--</div>
                <div class="stat-label">minutes</div>
            </div>
            <div class="card">
                <h2>This Week</h2>
                <div class="stat-value" id="week-total">--</div>
                <div class="stat-label">minutes</div>
            </div>
            <div class="card">
                <h2>Most Active</h2>
                <div class="stat-value" id="most-active">--</div>
                <div class="stat-label">this week</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 20px;">
            <h2>Daily Viewing Time (Last 7 Days)</h2>
            <div class="chart-container">
                <canvas id="dailyChart"></canvas>
            </div>
        </div>

        <div class="card">
            <h2>Recent Activity</h2>
            <table>
                <thead>
                    <tr><th>Time</th><th>Person</th></tr>
                </thead>
                <tbody id="recent-table"></tbody>
            </table>
        </div>

        <button class="refresh-btn" onclick="location.reload()">Refresh</button>
    </div>

    <script>
        const colors = {
            'mio': '#ff6b6b',
            'yu': '#4ecdc4',
            'tsubasa': '#ffe66d',
            'unknown': '#888888',
            'none': '#444444'
        };

        fetch('/api/stats')
            .then(res => res.json())
            .then(data => {
                // Today's total
                const today = new Date().toISOString().slice(0, 10);
                let todayTotal = 0;
                if (data.daily[today]) {
                    todayTotal = Object.values(data.daily[today]).reduce((a, b) => a + b, 0);
                }
                document.getElementById('today-total').textContent = Math.round(todayTotal);

                // Week total
                let weekTotal = 0;
                let personTotals = {};
                Object.values(data.daily).forEach(day => {
                    Object.entries(day).forEach(([name, mins]) => {
                        weekTotal += mins;
                        personTotals[name] = (personTotals[name] || 0) + mins;
                    });
                });
                document.getElementById('week-total').textContent = Math.round(weekTotal);

                // Most active
                let mostActive = '--';
                let maxMins = 0;
                Object.entries(personTotals).forEach(([name, mins]) => {
                    if (mins > maxMins) {
                        maxMins = mins;
                        mostActive = name;
                    }
                });
                document.getElementById('most-active').textContent = mostActive;

                // Chart
                const dates = Object.keys(data.daily).sort();
                const names = data.target_names;
                const datasets = names.map(name => ({
                    label: name,
                    data: dates.map(d => Math.round(data.daily[d]?.[name] || 0)),
                    backgroundColor: colors[name] || '#888',
                    borderColor: colors[name] || '#888',
                    borderWidth: 1
                }));

                new Chart(document.getElementById('dailyChart'), {
                    type: 'bar',
                    data: {
                        labels: dates.map(d => d.slice(5)),
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: { stacked: true, ticks: { color: '#888' }, grid: { color: '#333' } },
                            y: { stacked: true, ticks: { color: '#888' }, grid: { color: '#333' },
                                 title: { display: true, text: 'Minutes', color: '#888' } }
                        },
                        plugins: {
                            legend: { labels: { color: '#eee' } }
                        }
                    }
                });

                // Recent table
                const tbody = document.getElementById('recent-table');
                data.recent.slice(0, 20).forEach(entry => {
                    const tr = document.createElement('tr');
                    const nameClass = 'name-' + entry.name;
                    tr.innerHTML = `<td>${entry.timestamp}</td><td class="${nameClass}">${entry.name}</td>`;
                    tbody.appendChild(tr);
                });
            });
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/stats")
def api_stats():
    config = load_config()
    daily, recent = load_log_data(days=7)
    return jsonify({
        "daily": {k: dict(v) for k, v in daily.items()},
        "recent": recent,
        "target_names": config["target_names"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
