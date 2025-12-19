#!/usr/bin/env python3
import os
import sys
import time
import csv
import json
import logging
import datetime as dt

import cv2
import face_recognition
import pickle

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 設定ファイル読み込み
CONFIG_PATH = os.path.expanduser("~/config.json")

def load_config():
    """設定ファイルを読み込む。なければデフォルト値を返す"""
    defaults = {
        "camera_device": 0,
        "interval_sec": 5,
        "tolerance": 0.5,
        "face_model": "hog",
        "upsample": 2,
        "resize_width": 640,
        "roi": None,  # {"x": 0, "y": 0, "w": 100, "h": 100} in percent
        "use_roi": True,
        "encodings_path": "~/encodings.pkl",
        "log_path": "~/tv_watch_log.csv",
        "camera_retry_sec": 5,
        "max_camera_retries": 10,
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_config = json.load(f)
                defaults.update(user_config)
                logger.info("設定ファイルを読み込みました: %s", CONFIG_PATH)
        except json.JSONDecodeError as e:
            logger.error("設定ファイルのJSON解析エラー: %s", e)
            logger.info("デフォルト設定を使用します")
    else:
        logger.info("設定ファイルなし。デフォルト設定を使用: %s", CONFIG_PATH)
    return defaults

def load_encodings(path):
    """顔エンコーディングを読み込む"""
    if not os.path.exists(path):
        logger.error("エンコーディングファイルが見つかりません: %s", path)
        logger.error("先に build_encodings.py を実行してください")
        sys.exit(1)

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)

        if "names" not in data or "encodings" not in data:
            logger.error("エンコーディングファイルの形式が不正です")
            sys.exit(1)

        if len(data["names"]) == 0:
            logger.error("登録された顔がありません")
            sys.exit(1)

        return data["names"], data["encodings"]

    except Exception as e:
        logger.error("エンコーディングファイルの読み込みエラー: %s", e)
        sys.exit(1)

def open_camera(device, retry_sec, max_retries):
    """カメラを開く。失敗時はリトライ"""
    for attempt in range(max_retries):
        cap = cv2.VideoCapture(device)
        if cap.isOpened():
            logger.info("カメラを開きました: device=%s", device)
            return cap

        logger.warning(
            "カメラを開けません (試行 %d/%d)。%d秒後にリトライ...",
            attempt + 1, max_retries, retry_sec
        )
        time.sleep(retry_sec)

    logger.error("カメラを開けませんでした。終了します。")
    sys.exit(1)

def ensure_log_file(path):
    """ログファイルが存在しなければ作成"""
    if not os.path.exists(path):
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "name"])
            logger.info("ログファイルを作成しました: %s", path)
        except IOError as e:
            logger.error("ログファイルを作成できません: %s", e)
            sys.exit(1)

def write_log(path, timestamp, names):
    """ログに書き込む"""
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if names:
                for name in sorted(names):
                    writer.writerow([timestamp, name])
            else:
                writer.writerow([timestamp, "none"])
    except IOError as e:
        logger.error("ログ書き込みエラー: %s", e)

def main():
    # 設定読み込み
    config = load_config()

    CAMERA_DEVICE = config["camera_device"]
    ENC_PATH = os.path.expanduser(config["encodings_path"])
    LOG_PATH = os.path.expanduser(config["log_path"])
    INTERVAL_SEC = config["interval_sec"]
    TOLERANCE = config["tolerance"]
    FACE_MODEL = config["face_model"]
    UPSAMPLE = config["upsample"]
    RESIZE_WIDTH = config.get("resize_width", 640)
    ROI = config.get("roi")
    USE_ROI = config.get("use_roi", True)
    CAMERA_RETRY_SEC = config["camera_retry_sec"]
    MAX_CAMERA_RETRIES = config["max_camera_retries"]

    logger.info("検出設定: model=%s, upsample=%d, resize=%d, ROI=%s",
                FACE_MODEL, UPSAMPLE, RESIZE_WIDTH, "有効" if (USE_ROI and ROI) else "無効")

    # 顔エンコーディング読み込み
    known_names, known_encodings = load_encodings(ENC_PATH)
    logger.info("登録済み顔数: %d, 人物: %s", len(known_names), sorted(set(known_names)))

    # ログファイル準備
    ensure_log_file(LOG_PATH)

    # カメラ初期化
    cap = open_camera(CAMERA_DEVICE, CAMERA_RETRY_SEC, MAX_CAMERA_RETRIES)

    logger.info("監視を開始します (Ctrl+C で停止)")

    consecutive_failures = 0
    max_consecutive_failures = 30

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                logger.warning("フレーム取得失敗 (%d/%d)", consecutive_failures, max_consecutive_failures)

                if consecutive_failures >= max_consecutive_failures:
                    logger.error("連続失敗が上限に達しました。カメラを再接続します...")
                    cap.release()
                    time.sleep(CAMERA_RETRY_SEC)
                    cap = open_camera(CAMERA_DEVICE, CAMERA_RETRY_SEC, MAX_CAMERA_RETRIES)
                    consecutive_failures = 0
                else:
                    time.sleep(1)
                continue

            consecutive_failures = 0

            try:
                # ROI適用（ピクセル値）
                if USE_ROI and ROI:
                    x1 = ROI["x"]
                    y1 = ROI["y"]
                    x2 = x1 + ROI["w"]
                    y2 = y1 + ROI["h"]
                    frame = frame[y1:y2, x1:x2]

                # 縮小処理（メモリ節約）
                if RESIZE_WIDTH and RESIZE_WIDTH > 0:
                    h, w = frame.shape[:2]
                    if w > RESIZE_WIDTH:
                        scale = RESIZE_WIDTH / w
                        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

                # BGR -> RGB 変換
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 顔検出
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
                write_log(LOG_PATH, ts, seen_names)

                if seen_names:
                    logger.info("%s -> %s", ts, ", ".join(sorted(seen_names)))
                else:
                    logger.debug("%s -> none", ts)

            except Exception as e:
                logger.error("顔認識処理中にエラー: %s", e)

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        logger.info("停止要求を受けました")

    finally:
        cap.release()
        logger.info("カメラを解放しました。終了します。")

if __name__ == "__main__":
    main()
