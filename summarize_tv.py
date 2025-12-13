#!/usr/bin/env python3
import os
import csv
import json
import datetime as dt
from collections import defaultdict

# 設定ファイル読み込み
CONFIG_PATH = os.path.expanduser("~/config.json")

def load_config():
    """設定ファイルを読み込む。なければデフォルト値を返す"""
    defaults = {
        "interval_sec": 5,
        "log_path": "~/tv_watch_log.csv",
        "target_names": ["mio", "yu", "tsubasa"],
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_config = json.load(f)
            defaults.update(user_config)
    return defaults

config = load_config()

LOG_PATH = os.path.expanduser(config["log_path"])
OUT_PATH = os.path.expanduser("~/tv_watch_summary.csv")
INTERVAL_SEC = config["interval_sec"]
TARGET_NAMES = config["target_names"]

# 日付×名前で分数を集計
minutes = defaultdict(float)

with open(LOG_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["name"]
        if name not in TARGET_NAMES:
            continue  # unknown / none は無視

        ts = dt.datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        date_str = ts.date().isoformat()
        key = (date_str, name)
        minutes[key] += INTERVAL_SEC / 60.0

with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "name", "minutes"])
    for (date_str, name), m in sorted(minutes.items()):
        writer.writerow([date_str, name, round(m, 1)])

print("書き出し完了:", OUT_PATH)
