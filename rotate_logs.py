#!/usr/bin/env python3
"""
ログローテーションスクリプト

古いログを月別にアーカイブし、メインログファイルをクリアします。
cronで月初に実行することを想定。

例: 0 0 1 * * /home/pi/venv/bin/python /home/pi/rotate_logs.py
"""
import os
import csv
import gzip
import shutil
import json
import datetime as dt
from collections import defaultdict

# 設定ファイル読み込み
CONFIG_PATH = os.path.expanduser("~/config.json")
ARCHIVE_DIR = os.path.expanduser("~/tv_watch_archives")

def load_config():
    defaults = {
        "log_path": "~/tv_watch_log.csv",
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_config = json.load(f)
            defaults.update(user_config)
    return defaults

def rotate_log():
    config = load_config()
    log_path = os.path.expanduser(config["log_path"])

    if not os.path.exists(log_path):
        print("ログファイルが存在しません:", log_path)
        return

    # アーカイブディレクトリ作成
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    # ログを月別に分割
    monthly_data = defaultdict(list)
    header = None

    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            if len(row) < 2:
                continue
            try:
                ts = dt.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                month_key = ts.strftime("%Y-%m")
                monthly_data[month_key].append(row)
            except ValueError:
                continue

    if not monthly_data:
        print("ログデータがありません")
        return

    # 今月のデータは残す
    current_month = dt.datetime.now().strftime("%Y-%m")
    archived_count = 0

    for month_key, rows in sorted(monthly_data.items()):
        if month_key == current_month:
            continue  # 今月は残す

        # アーカイブファイル作成（gzip圧縮）
        archive_path = os.path.join(ARCHIVE_DIR, f"tv_watch_log_{month_key}.csv.gz")

        # 既存アーカイブがあれば追記用に読み込む
        existing_rows = []
        if os.path.exists(archive_path):
            with gzip.open(archive_path, "rt", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                existing_rows = list(reader)

        # アーカイブに書き込み
        with gzip.open(archive_path, "wt", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(existing_rows)
            writer.writerows(rows)

        print(f"アーカイブ: {archive_path} ({len(rows)} 件追加)")
        archived_count += len(rows)

    # 今月のデータだけでログファイルを再作成
    current_rows = monthly_data.get(current_month, [])
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(current_rows)

    print(f"ログローテーション完了: {archived_count} 件アーカイブ, {len(current_rows)} 件残存")

if __name__ == "__main__":
    rotate_log()
