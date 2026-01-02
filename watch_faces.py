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
        "save_detections": True,
        "detections_dir": "~/detections",
        "max_detection_images": 100,
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

    # ROI設定：roi_indexとroi_presetsから適用するROIを決定
    ROI = config.get("roi")
    roi_index = config.get("roi_index", "")
    roi_presets = config.get("roi_presets", [])
    if roi_index and roi_presets:
        try:
            idx = int(roi_index) - 1  # 1-based to 0-based
            if 0 <= idx < len(roi_presets):
                ROI = roi_presets[idx]
                logger.info("ROIプリセット %s を適用: %s", roi_index, ROI.get("name", ""))
        except (ValueError, IndexError) as e:
            logger.warning("ROIプリセットの適用に失敗: %s", e)

    USE_ROI = config.get("use_roi", True)
    CAMERA_RETRY_SEC = config["camera_retry_sec"]
    MAX_CAMERA_RETRIES = config["max_camera_retries"]
    SAVE_DETECTIONS = config.get("save_detections", True)
    DETECTIONS_DIR = os.path.expanduser(config.get("detections_dir", "~/detections"))
    MAX_DETECTION_IMAGES = config.get("max_detection_images", 100)

    # 検出画像保存ディレクトリ作成
    if SAVE_DETECTIONS:
        os.makedirs(DETECTIONS_DIR, exist_ok=True)

    # 適用中の設定を保存（管理画面で参照用）
    applied_config = {
        "face_model": FACE_MODEL,
        "upsample": UPSAMPLE,
        "interval_sec": INTERVAL_SEC,
        "tolerance": TOLERANCE,
        "roi_index": config.get("roi_index", "")
    }
    try:
        import json
        with open(os.path.expanduser("~/tv_watch_applied_config.json"), 'w') as f:
            json.dump(applied_config, f)
    except Exception as e:
        logger.warning("適用設定の保存に失敗: %s", e)

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
                # 元フレームを保存（ROI適用前）
                full_frame = frame.copy()

                # ROI適用（ピクセル値）
                roi_info = None  # メタデータ保存用
                roi_offset_x = 0
                roi_offset_y = 0
                if USE_ROI and ROI:
                    roi_info = {"x": ROI["x"], "y": ROI["y"], "w": ROI["w"], "h": ROI["h"]}
                    x1 = ROI["x"]
                    y1 = ROI["y"]
                    x2 = x1 + ROI["w"]
                    y2 = y1 + ROI["h"]
                    roi_offset_x = x1
                    roi_offset_y = y1
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
                face_results = []  # [(name, location), ...]

                for i, enc in enumerate(face_encodings):
                    matches = face_recognition.compare_faces(
                        known_encodings, enc, tolerance=TOLERANCE
                    )
                    face_distances = face_recognition.face_distance(known_encodings, enc)

                    if len(face_distances) == 0:
                        continue

                    best_index = face_distances.argmin()
                    best_distance = float(face_distances[best_index])
                    name = "unknown"
                    if matches[best_index]:
                        name = known_names[best_index]

                    seen_names.add(name)
                    if i < len(face_locations):
                        face_results.append((name, face_locations[i], best_distance))

                ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                write_log(LOG_PATH, ts, seen_names)

                # 最新フレームを常に保存（フルフレームにROI枠とBBox付き）
                latest_frame = full_frame.copy()
                # ROI枠を描画（オレンジ色）
                if roi_info:
                    cv2.rectangle(latest_frame,
                                  (roi_info["x"], roi_info["y"]),
                                  (roi_info["x"] + roi_info["w"], roi_info["y"] + roi_info["h"]),
                                  (0, 165, 255), 2)
                # BBox描画（座標をフルフレーム基準に変換）
                for name, (top, right, bottom, left), dist in face_results:
                    color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
                    similarity = max(0, (1 - dist) * 100)
                    # ROIオフセットを加算してフルフレーム座標に変換
                    abs_left = left + roi_offset_x
                    abs_top = top + roi_offset_y
                    abs_right = right + roi_offset_x
                    abs_bottom = bottom + roi_offset_y
                    cv2.rectangle(latest_frame, (abs_left, abs_top), (abs_right, abs_bottom), color, 2)
                    cv2.putText(latest_frame, f"{name} ({similarity:.0f}%)", (abs_left, abs_top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                latest_path = os.path.join(DETECTIONS_DIR, "latest_frame.jpg")
                cv2.imwrite(latest_path, latest_frame)

                # クリーンなフレームも保存（撮影用、オーバーレイなし）
                clean_path = os.path.join(DETECTIONS_DIR, "latest_frame_clean.jpg")
                cv2.imwrite(clean_path, full_frame)

                # latest_frame用のメタデータを保存（BBox表示用）
                import json
                latest_meta = {
                    "roi": roi_info,
                    "faces": []
                }
                for name, (top, right, bottom, left), dist in face_results:
                    latest_meta["faces"].append({
                        "name": name,
                        "bbox": {
                            "top": top + roi_offset_y,
                            "right": right + roi_offset_x,
                            "bottom": bottom + roi_offset_y,
                            "left": left + roi_offset_x
                        },
                        "similarity": max(0, (1 - dist) * 100)
                    })
                latest_meta_path = os.path.join(DETECTIONS_DIR, "latest_frame_meta.json")
                with open(latest_meta_path, 'w') as f:
                    json.dump(latest_meta, f)

                if seen_names:
                    logger.info("%s -> %s", ts, ", ".join(sorted(seen_names)))

                    # 検出画像を保存
                    if SAVE_DETECTIONS and face_results:
                        timestamp_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

                        # 元画像を保存（フルフレーム、オーバーレイなし）
                        orig_filename = f"detection_{timestamp_str}_original.jpg"
                        orig_filepath = os.path.join(DETECTIONS_DIR, orig_filename)
                        cv2.imwrite(orig_filepath, full_frame)

                        # メタデータを保存（座標はフルフレーム基準）
                        import json
                        meta = {
                            "timestamp": timestamp_str,
                            "roi": roi_info,
                            "faces": []
                        }
                        for name, (top, right, bottom, left), dist in face_results:
                            # ROIオフセットを加算してフルフレーム座標に変換
                            meta["faces"].append({
                                "name": name,
                                "bbox": {
                                    "top": top + roi_offset_y,
                                    "right": right + roi_offset_x,
                                    "bottom": bottom + roi_offset_y,
                                    "left": left + roi_offset_x
                                },
                                "distance": dist,
                                "similarity": max(0, (1 - dist) * 100)
                            })
                        meta_filename = f"detection_{timestamp_str}_meta.json"
                        meta_filepath = os.path.join(DETECTIONS_DIR, meta_filename)
                        with open(meta_filepath, 'w') as f:
                            json.dump(meta, f)

                        # 各検出者ごとにファイル名を記録（互換性のため）
                        for name in seen_names:
                            if name != "unknown":
                                filename = f"detection_{timestamp_str}_{name}.jpg"
                                # シンボリックリンクまたはコピーの代わりにメタファイルを参照

                        # 古いファイルを削除
                        import glob as glob_module
                        all_files = sorted(glob_module.glob(os.path.join(DETECTIONS_DIR, "detection_*")))
                        # original.jpg, meta.json, name.jpg のセットを数える
                        timestamps = set()
                        for f in all_files:
                            base = os.path.basename(f)
                            if base.startswith("detection_") and "_" in base[10:]:
                                ts_part = base[10:25]  # YYYYMMDD_HHMMSS
                                timestamps.add(ts_part)
                        if len(timestamps) > MAX_DETECTION_IMAGES:
                            old_timestamps = sorted(timestamps)[:-MAX_DETECTION_IMAGES]
                            for old_ts in old_timestamps:
                                for old_file in glob_module.glob(os.path.join(DETECTIONS_DIR, f"detection_{old_ts}*")):
                                    try:
                                        os.remove(old_file)
                                    except:
                                        pass
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
