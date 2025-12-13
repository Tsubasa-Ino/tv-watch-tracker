#!/usr/bin/env python3
import os
import csv
import datetime as dt
from collections import defaultdict

LOG_PATH = os.path.expanduser("~/tv_watch_log.csv")
OUT_PATH = os.path.expanduser("~/tv_watch_summary.csv")

INTERVAL_SEC = 5  # watch_faces.py と合わせる
TARGET_NAMES = ["mio", "yu", "tsubasa"]  # ★ ここに tsuabsa 追加

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
