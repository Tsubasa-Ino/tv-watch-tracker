#!/usr/bin/env python3
import os
import time
import csv
import json
import datetime as dt

import cv2
import face_recognition
import pickle

# 設定ファイル読み込み
CONFIG_PATH = os.path.expanduser("~/config.json")

def load_config():
    """設定ファイルを読み込む。なければデフォルト値を返す"""
    defaults = {
        "camera_device": 0,
        "interval_sec": 5,
        "tolerance": 0.5,
        "face_model": "cnn",
        "upsample": 0,
        "encodings_path": "~/encodings.pkl",
        "log_path": "~/tv_watch_log.csv",
    }
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            user_config = json.load(f)
            defaults.update(user_config)
    return defaults

config = load_config()

# 設定値を変数に展開
CAMERA_DEVICE = config["camera_device"]
ENC_PATH = os.path.expanduser(config["encodings_path"])
LOG_PATH = os.path.expanduser(config["log_path"])
INTERVAL_SEC = config["interval_sec"]
TOLERANCE = config["tolerance"]
FACE_MODEL = config["face_model"]
UPSAMPLE = config["upsample"]

# 既知の顔エンコーディング読み込み
data = pickle.load(open(ENC_PATH, "rb"))
known_names = data["names"]
known_encodings = data["encodings"]

print("encodings loaded:", len(known_names), "faces")
print("persons:", sorted(set(known_names)))

# ログファイル作成（なければ）
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "name"])
    print("created log file:", LOG_PATH)

# カメラオープン
cap = cv2.VideoCapture(CAMERA_DEVICE)
if not cap.isOpened():
    raise RuntimeError("カメラが開けませんでした（/dev/video0？）")

print("camera opened. Start watching... (Ctrl+C で停止)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("フレーム取得に失敗。少し待ちます…")
            time.sleep(1)
            continue

        # ★ ここを修正：dlib が扱いやすいように cv2.cvtColor で RGB 化
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 顔検出（CNN）
        face_locations = face_recognition.face_locations(
            rgb,
            model=FACE_MODEL,
            number_of_times_to_upsample=UPSAMPLE,
        )

        # 顔エンコーディング
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        seen_names = set()

        for enc in face_encodings:
            matches = face_recognition.compare_faces(
                known_encodings, enc, tolerance=TOLERANCE
            )
            face_distances = face_recognition.face_distance(known_encodings, enc)

            if len(face_distances) == 0:
                continue

            best_index = face_distances.argmin()
            name = "unknown"
            if matches[best_index]:
                name = known_names[best_index]

            seen_names.add(name)

        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if seen_names:
                for name in sorted(seen_names):
                    writer.writerow([ts, name])
                print(f"{ts} -> {', '.join(sorted(seen_names))}")
            else:
                writer.writerow([ts, "none"])
                print(f"{ts} -> none")

        time.sleep(INTERVAL_SEC)

except KeyboardInterrupt:
    print("\n停止要求を受けました。終了します。")

finally:
    cap.release()
    print("bye")
