#!/usr/bin/env python3
"""
顔管理Web UI
- リアルタイムプレビュー＆撮影
- ROI設定（複数保存対応）
- 顔抽出（画像から顔を検出・保存）
- 顔登録（ラベリング・自動エンコーディング）
- 顔認識テスト
- ダッシュボード
"""
import os
import sys
import json
import time
import glob
import shutil
import cv2
import face_recognition
import pickle
from flask import Flask, render_template_string, jsonify, request, Response, send_file

app = Flask(__name__)

# パス設定
BASE_DIR = os.path.expanduser("~")
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
FACES_DIR = os.path.join(BASE_DIR, "faces")
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pkl")

os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

camera = None

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

def is_service_running():
    """顔認識サービスが稼働中かチェック"""
    import subprocess
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "tv-watch-tracker"],
            capture_output=True, text=True
        )
        return result.stdout.strip() == "active"
    except:
        return False

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def stop_service_and_get_camera():
    """顔認識サービスを停止してカメラを取得"""
    global camera
    os.system("sudo systemctl stop tv-watch-tracker 2>/dev/null")
    time.sleep(0.5)
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    return camera

def get_roi_by_index(roi_index):
    """ROIインデックスからROIを取得"""
    if roi_index == "" or roi_index is None:
        return None
    try:
        idx = int(roi_index)
        config = load_config()
        presets = config.get("roi_presets", [])
        if 0 <= idx < len(presets):
            return presets[idx]
    except (ValueError, TypeError):
        pass
    return None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>顔管理システム</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: #1a1a2e;
            color: #fff;
            min-height: 100vh;
        }
        .tabs {
            display: flex;
            background: #16213e;
            border-bottom: 2px solid #00d4ff;
            flex-wrap: wrap;
        }
        .tab {
            padding: 15px 20px;
            cursor: pointer;
            border: none;
            background: transparent;
            color: #888;
            font-size: 1em;
            transition: all 0.3s;
        }
        .tab:hover { color: #fff; }
        .tab.active { color: #00d4ff; background: #0f3460; }
        .content { padding: 20px; max-width: 900px; margin: 0 auto; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .card { background: #16213e; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
        h2 { color: #00d4ff; margin-bottom: 15px; font-size: 1.3em; }
        h3 { color: #ffe66d; margin: 15px 0 10px; font-size: 1.1em; }
        .preview-container { position: relative; width: 100%; background: #000; border-radius: 8px; overflow: hidden; }
        .preview-container img, .preview-container canvas { width: 100%; display: block; }
        #roiCanvas { position: absolute; top: 0; left: 0; cursor: crosshair; }
        .btn {
            padding: 12px 24px;
            border: 2px solid transparent;
            border-radius: 8px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            margin: 5px;
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 6px 12px rgba(0,0,0,0.4); }
        .btn:active { transform: translateY(0); box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .btn-primary { background: linear-gradient(135deg, #00d4ff, #0099cc); color: #1a1a2e; border-color: #00b8e6; }
        .btn-success { background: linear-gradient(135deg, #4ecdc4, #3db8b0); color: #1a1a2e; border-color: #45c4bb; }
        .btn-danger { background: linear-gradient(135deg, #ff6b6b, #e55555); color: #fff; border-color: #ff5555; }
        .btn-secondary { background: linear-gradient(135deg, #666, #555); color: #fff; border-color: #777; }
        .btn-small { padding: 8px 16px; font-size: 0.85em; }
        .status { padding: 10px; border-radius: 8px; margin: 10px 0; text-align: center; }
        .status.success { background: #4ecdc4; color: #1a1a2e; }
        .status.error { background: #ff6b6b; }
        .status.info { background: #0f3460; color: #00d4ff; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); gap: 10px; }
        .grid-item { position: relative; aspect-ratio: 1; background: #0f3460; border-radius: 8px; overflow: hidden; cursor: pointer; }
        .grid-item img { width: 100%; height: 100%; object-fit: cover; }
        .grid-item .delete-btn {
            position: absolute; top: 5px; right: 5px;
            background: rgba(255,107,107,0.9); color: #fff;
            border: none; border-radius: 50%; width: 24px; height: 24px;
            cursor: pointer; display: none; font-size: 14px; line-height: 24px; text-align: center;
        }
        .grid-item:hover .delete-btn { display: block; }
        .grid-item.selected { outline: 3px solid #00d4ff; }
        .grid-item .filename { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.7); padding: 3px; font-size: 0.7em; text-align: center; color: #fff; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .grid-item.registered { outline: 3px solid #4ecdc4; }
        .grid-item.unregistered { outline: 3px solid #ffe66d; }
        .face-item { position: relative; display: inline-block; margin: 5px; }
        .face-item img { width: 80px; height: 80px; object-fit: cover; border-radius: 8px; cursor: pointer; }
        .face-item.selected img { outline: 3px solid #00d4ff; }
        .face-item .delete-btn {
            position: absolute; top: -5px; right: -5px;
            background: rgba(255,107,107,0.9); color: #fff;
            border: none; border-radius: 50%; width: 20px; height: 20px;
            cursor: pointer; display: none; font-size: 12px; line-height: 20px; text-align: center;
        }
        .face-item:hover .delete-btn { display: block; }
        .face-item .badge {
            position: absolute; top: 3px; left: 3px;
            padding: 2px 6px; border-radius: 4px; font-size: 0.6em;
        }
        .badge-registered { background: #4ecdc4; color: #000; }
        .badge-unregistered { background: #ffe66d; color: #000; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; color: #00d4ff; }
        .form-group input, .form-group select {
            width: 100%; padding: 10px; border: none; border-radius: 8px;
            font-size: 1em; background: #0f3460; color: #fff;
        }
        .roi-info { background: #0f3460; padding: 10px; border-radius: 8px; margin-top: 10px; font-family: monospace; }
        .face-list { max-height: 400px; overflow-y: auto; }
        .modal {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.9); justify-content: center; align-items: center; z-index: 100;
        }
        .modal.active { display: flex; flex-direction: column; }
        .modal img { max-width: 90%; max-height: 70%; border-radius: 8px; }
        .modal-close { position: absolute; top: 20px; right: 20px; color: #fff; font-size: 2em; cursor: pointer; }
        .modal-controls { margin-top: 20px; }
        .detection-result { margin-top: 15px; }
        .detection-result .face-box {
            display: inline-block; margin: 5px; padding: 10px;
            background: #0f3460; border-radius: 8px; text-align: center;
        }
        .detection-result .face-box img { width: 100px; height: 100px; object-fit: cover; border-radius: 4px; }
        .params { display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 15px; }
        .params .form-group { flex: 1; min-width: 150px; }
        .roi-preset { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 15px; }
        .roi-preset-item {
            background: #0f3460; padding: 10px 15px; border-radius: 8px;
            display: flex; align-items: center; gap: 10px; transition: all 0.2s;
        }
        .roi-preset-item:hover { background: #16213e; }
        .roi-preset-item.selected { background: #ffe66d; color: #1a1a2e; }
        .roi-preset-item.selected small { color: #333 !important; }
        .roi-preset-item .delete-roi { color: #ff6b6b; cursor: pointer; font-size: 1.2em; }
        .label-group { background: #0f3460; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
        .label-group h4 { color: #ffe66d; margin-bottom: 10px; }

        /* スマホ用レスポンシブ */
        @media (max-width: 768px) {
            .tabs {
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: none;
            }
            .tabs::-webkit-scrollbar { display: none; }
            .tab {
                padding: 12px 14px;
                font-size: 0.85em;
                white-space: nowrap;
                flex-shrink: 0;
            }
            .content { padding: 12px; }
            .card { padding: 15px; margin-bottom: 15px; }
            h2 { font-size: 1.1em; margin-bottom: 12px; }
            h3 { font-size: 1em; }
            .btn {
                padding: 10px 16px;
                font-size: 0.9em;
                margin: 3px;
            }
            .btn-small { padding: 8px 12px; font-size: 0.8em; }
            .grid { grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 8px; }
            .params { gap: 10px; }
            .params .form-group { min-width: 120px; }
            .form-group input, .form-group select { padding: 8px; font-size: 0.9em; }
            .modal img { max-width: 95%; max-height: 50%; }
            .modal-close { top: 10px; right: 10px; font-size: 1.5em; }
            #detectionControls { font-size: 0.85em; }
            #detectionControls label { margin-right: 8px; }
            #relabelControls { width: 95%; padding: 12px; }
            #relabelControls h4 { font-size: 0.95em; }
            .roi-preset { gap: 8px; }
            .roi-preset-item { padding: 8px 12px; font-size: 0.9em; }
            .detection-result .face-box { padding: 8px; margin: 3px; }
            .detection-result .face-box img { width: 70px; height: 70px; }
            .face-item img { width: 60px; height: 60px; }
        }

        @media (max-width: 480px) {
            .tab { padding: 10px 10px; font-size: 0.8em; }
            .content { padding: 8px; }
            .card { padding: 12px; }
            h2 { font-size: 1em; }
            .btn { padding: 8px 12px; font-size: 0.85em; }
            .grid { grid-template-columns: repeat(auto-fill, minmax(70px, 1fr)); gap: 6px; }
            .params .form-group { min-width: 100%; }
            #todayByLabel > div, #weekByLabel > div {
                min-width: calc(50% - 5px) !important;
                padding: 8px 10px !important;
            }
            #todayByLabel > div > div:last-child,
            #weekByLabel > div > div:last-child {
                font-size: 1.2em !important;
            }
        }
    </style>
</head>
<body>
    <div class="tabs">
        <button class="tab active" onclick="showTab('camera')">撮影</button>
        <button class="tab" onclick="showTab('roi')">ROI設定</button>
        <button class="tab" onclick="showTab('extract')">顔抽出</button>
        <button class="tab" onclick="showTab('register')">顔登録</button>
        <button class="tab" onclick="showTab('test')">テスト</button>
        <button class="tab" onclick="showTab('settings')">顔認識</button>
        <button class="tab" onclick="showTab('dashboard')">ダッシュボード</button>
    </div>
    <div class="content">
        <!-- カメラタブ -->
        <div id="camera" class="tab-content active">
            <div class="card">
                <h2 id="cameraTitle">プレビュー</h2>
                <!-- サービス稼働中: 検出画像 -->
                <div id="serviceImageContainer" style="display:none;">
                    <p style="color:#4ecdc4;margin-bottom:10px;">サービス稼働中 - 検出画像を表示</p>
                    <div style="margin-bottom:10px;display:flex;gap:20px;align-items:center;flex-wrap:wrap;">
                        <label><input type="checkbox" id="camShowRoi" checked onchange="updateServiceImage()"> ROI表示</label>
                        <label><input type="checkbox" id="camShowBbox" checked onchange="updateServiceImage()"> BBox表示</label>
                    </div>
                    <div class="preview-container">
                        <img id="serviceImage" src="" style="width:100%;">
                    </div>
                    <p id="serviceImageTime" style="color:#888;font-size:0.9em;margin-top:10px;text-align:center;"></p>
                    <div style="margin-top:15px; text-align:center;">
                        <button class="btn btn-success" onclick="captureServiceFrame()">この画像を保存</button>
                    </div>
                </div>
                <!-- サービス停止中: リアルタイムストリーム -->
                <div id="cameraContainer">
                    <div class="preview-container">
                        <img id="cameraPreview" src="/stream">
                    </div>
                    <div style="margin-top:15px; text-align:center;">
                        <button class="btn btn-success" onclick="capture()">撮影</button>
                    </div>
                </div>
                <div id="captureStatus"></div>
            </div>
            <div class="card">
                <h2>撮影済み画像</h2>
                <div class="grid" id="captureGrid"></div>
            </div>
        </div>

        <!-- ROI設定タブ -->
        <div id="roi" class="tab-content">
            <div class="card">
                <h2>ROI（検出領域）設定</h2>
                <p style="color:#888;margin-bottom:15px;">撮影画像を選択し、マウスでドラッグして検出領域を指定</p>
                <h3>画像を選択</h3>
                <div class="grid" id="roiImageGrid" style="margin-bottom:15px;"></div>
                <div class="preview-container" id="roiContainer" style="display:none;">
                    <img id="roiImage" src="">
                    <canvas id="roiCanvas"></canvas>
                </div>
                <div id="roiEditControls" style="display:none;margin-top:15px;">
                    <button class="btn btn-success" onclick="saveRoiPreset()">ROI追加保存</button>
                    <button class="btn btn-danger" onclick="clearRoiDraw()">描画クリア</button>
                </div>
                <div class="roi-info" id="roiInfo">ROI: 未設定</div>
            </div>
            <div class="card">
                <h2>保存済みROI一覧</h2>
                <div id="roiPresetList" class="roi-preset"></div>
                <div id="roiPresetStatus"></div>
            </div>
        </div>

        <!-- 顔抽出タブ -->
        <div id="extract" class="tab-content">
            <div class="card">
                <h2>顔抽出</h2>
                <p style="color:#888;margin-bottom:15px;">撮影画像から顔を検出して抽出</p>
                <div class="params">
                    <div class="form-group">
                        <label>検出モデル</label>
                        <select id="extractModel">
                            <option value="hog" selected>HOG（軽量）</option>
                            <option value="cnn">CNN（高精度）</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Upsample</label>
                        <select id="extractUpsample">
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2" selected>2</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>ROI選択</label>
                        <select id="extractRoiSelect" onchange="loadExtractImages()">
                            <option value="">使用しない</option>
                        </select>
                    </div>
                </div>
                <h3>画像を選択（複数可）</h3>
                <div class="grid" id="extractImageGrid"></div>
                <div style="margin-top:15px;">
                    <button class="btn btn-primary" onclick="extractFaces()">選択画像から顔を検出</button>
                </div>
                <div id="extractStatus"></div>
            </div>
            <div class="card">
                <h2>抽出済み顔一覧</h2>
                <p style="color:#888;margin-bottom:10px;"><span style="background:#4ecdc4;color:#000;padding:2px 6px;border-radius:4px;font-size:0.8em;">登録済</span> <span style="background:#ffe66d;color:#000;padding:2px 6px;border-radius:4px;font-size:0.8em;">未登録</span></p>
                <div id="extractedFacesList"></div>
            </div>
        </div>

        <!-- 顔登録タブ -->
        <div id="register" class="tab-content">
            <div class="card">
                <h2>未登録顔のラベリング</h2>
                <p style="color:#888;margin-bottom:15px;">顔を選択し、名前を付けて登録（登録後自動エンコード）</p>
                <div class="form-group" style="max-width:300px;">
                    <label>登録する人の名前</label>
                    <input type="text" id="labelName" placeholder="例: tsubasa">
                </div>
                <div id="unregisteredFaces"></div>
                <div style="margin-top:15px;">
                    <button class="btn btn-success" onclick="registerSelectedFaces()">選択した顔を登録</button>
                    <button class="btn btn-secondary" onclick="selectAllUnregistered()">全選択</button>
                    <button class="btn btn-secondary" onclick="deselectAllUnregistered()">全解除</button>
                </div>
                <div id="registerStatus"></div>
            </div>
            <div class="card">
                <h2>登録済み顔一覧</h2>
                <div id="registeredFaces"></div>
            </div>
            <div class="card">
                <h2>ラベル管理</h2>
                <p style="color:#888;margin-bottom:15px;">画像未登録のラベルを表示・削除</p>
                <div id="labelStatus"></div>
            </div>
        </div>

        <!-- テストタブ -->
        <div id="test" class="tab-content">
            <!-- テスト種別選択 -->
            <div style="display:flex;gap:10px;margin-bottom:15px;">
                <button class="btn btn-secondary" id="testTypeDetect" onclick="switchTestType('detect')" style="flex:1;">顔検出</button>
                <button class="btn btn-secondary" id="testTypeRecog" onclick="switchTestType('recog')" style="flex:1;">顔判定</button>
                <button class="btn btn-primary" id="testTypeAll" onclick="switchTestType('all')" style="flex:1;">顔認識</button>
            </div>

            <!-- 顔検出テスト -->
            <div id="testDetect" class="card" style="display:none;">
                <h2>顔検出テスト</h2>
                <p style="color:#888;margin-bottom:15px;">カメラ画像から顔を検出しBBoxを表示</p>
                <div class="params">
                    <div class="form-group">
                        <label>検出モデル</label>
                        <select id="detectModel">
                            <option value="hog">HOG（軽量）</option>
                            <option value="cnn">CNN（高精度）</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Upsample</label>
                        <select id="detectUpsample">
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2" selected>2</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>ROI選択</label>
                        <select id="detectRoiSelect" onchange="loadDetectImages()">
                            <option value="">使用しない</option>
                        </select>
                    </div>
                </div>
                <h3>テスト画像を選択</h3>
                <div class="grid" id="detectImageGrid"></div>
                <input type="hidden" id="detectImage" value="">
                <div style="margin-top:15px;">
                    <button class="btn btn-primary" onclick="runDetection(this)">顔検出実行</button>
                </div>
                <div id="detectStatus"></div>
                <div class="detection-result" id="detectResult"></div>
            </div>

            <!-- 顔判定テスト -->
            <div id="testRecog" class="card" style="display:none;">
                <h2>顔判定テスト</h2>
                <p style="color:#888;margin-bottom:15px;">抽出済みの顔画像から誰か判定</p>
                <div class="params">
                    <div class="form-group">
                        <label>類似度閾値</label>
                        <select id="recogOnlyTolerance">
                            <option value="60">60%（厳密）</option>
                            <option value="50" selected>50%（標準）</option>
                            <option value="40">40%（緩め）</option>
                        </select>
                    </div>
                </div>
                <h3>顔画像を選択</h3>
                <div id="recogFaceGrid" style="display:flex;flex-wrap:wrap;gap:10px;"></div>
                <input type="hidden" id="recogFaceFile" value="">
                <div style="margin-top:15px;">
                    <button class="btn btn-primary" onclick="runRecogOnly()">顔判定実行</button>
                </div>
                <div id="recogOnlyStatus"></div>
                <div id="recogOnlyResult" style="margin-top:15px;"></div>
            </div>

            <!-- 顔認識テスト -->
            <div id="testAll" class="card">
                <h2>顔認識テスト</h2>
                <p style="color:#888;margin-bottom:15px;">カメラ画像から顔検出＋判定を実行</p>
                <div class="params">
                    <div class="form-group">
                        <label>検出モデル</label>
                        <select id="recogModel">
                            <option value="hog">HOG（軽量）</option>
                            <option value="cnn">CNN（高精度）</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Upsample</label>
                        <select id="recogUpsample">
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2" selected>2</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>類似度閾値</label>
                        <select id="recogTolerance">
                            <option value="60">60%（厳密）</option>
                            <option value="50" selected>50%（標準）</option>
                            <option value="40">40%（緩め）</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>ROI選択</label>
                        <select id="recogRoiSelect" onchange="loadRecogImages()">
                            <option value="">使用しない</option>
                        </select>
                    </div>
                </div>
                <h3>テスト画像を選択</h3>
                <div class="grid" id="recogImageGrid"></div>
                <input type="hidden" id="recogImage" value="">
                <div style="margin-top:15px;">
                    <button class="btn btn-primary" onclick="runRecognition(this)">顔認識実行</button>
                </div>
                <div id="recogStatus"></div>
                <div class="detection-result" id="recogResult"></div>
            </div>
        </div>

        <!-- 顔認識タブ -->
        <div id="settings" class="tab-content">
            <div class="card">
                <h2>顔認識サービス</h2>
                <div style="display:flex;align-items:center;gap:20px;margin-bottom:20px;">
                    <div>
                        <button class="btn btn-success" onclick="serviceControl('start')">開始</button>
                        <button class="btn btn-danger" onclick="serviceControl('stop')" style="margin-left:10px;">停止</button>
                    </div>
                    <div style="display:flex;align-items:center;gap:10px;">
                        <span style="color:#888;">状態:</span>
                        <span id="cfgServiceStatus" style="padding:5px 15px;border-radius:4px;font-weight:bold;">確認中...</span>
                    </div>
                </div>
                <div style="background:#0f3460;padding:15px;border-radius:8px;margin-bottom:20px;">
                    <h4 style="margin:0 0 10px 0;color:#4ecdc4;">適用中の設定</h4>
                    <div id="appliedConfigDisplay" style="font-size:0.9em;color:#ccc;">読込中...</div>
                </div>
            </div>
            <div class="card">
                <h2>設定変更</h2>
                <div class="params">
                    <div class="form-group">
                        <label>検出モデル</label>
                        <select id="cfgModel">
                            <option value="hog">HOG（高速）</option>
                            <option value="cnn">CNN（高精度）</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>UpSample</label>
                        <select id="cfgUpsample">
                            <option value="0">0（高速）</option>
                            <option value="1">1</option>
                            <option value="2">2（小顔検出）</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>撮影間隔</label>
                        <select id="cfgInterval">
                            <option value="3">3秒</option>
                            <option value="5">5秒</option>
                            <option value="10">10秒</option>
                            <option value="30">30秒</option>
                            <option value="60">1分</option>
                            <option value="300">5分</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>類似度閾値</label>
                        <select id="cfgTolerance">
                            <option value="60">60%（厳密）</option>
                            <option value="50">50%（標準）</option>
                            <option value="40">40%（緩め）</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>ROI選択</label>
                        <select id="cfgRoiSelect">
                            <option value="">使用しない</option>
                        </select>
                    </div>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <button class="btn btn-primary" onclick="saveAndApplyConfig()">設定を保存して反映</button>
                    <span id="configStatus" style="margin-left:10px;"></span>
                </div>
            </div>
        </div>

        <!-- ダッシュボードタブ -->
        <div id="dashboard" class="tab-content">
            <div class="card">
                <h2>直近の画像</h2>
                <div style="margin-bottom:10px;display:flex;gap:20px;align-items:center;flex-wrap:wrap;">
                    <label><input type="checkbox" id="showRoi" checked onchange="updateLatestImage()"> ROI表示</label>
                    <label><input type="checkbox" id="showBbox" checked onchange="updateLatestImage()"> BBox表示</label>
                    <span id="roiNameDisplay" style="color:#4ecdc4;font-size:0.9em;"></span>
                </div>
                <div id="latestImageContainer" style="text-align:center;">
                    <img id="latestImage" src="" style="max-width:100%;border-radius:8px;display:none;">
                    <p id="noLatestImage" style="color:#888;">画像なし</p>
                </div>
            </div>
            <div class="card">
                <h2>視聴時間</h2>
                <div style="display:flex;gap:20px;flex-wrap:wrap;">
                    <div style="flex:1;min-width:200px;">
                        <h3 style="color:#ffe66d;margin-bottom:10px;">本日</h3>
                        <div id="todayByLabel" style="display:flex;flex-wrap:wrap;gap:10px;"></div>
                    </div>
                    <div style="flex:1;min-width:200px;">
                        <h3 style="color:#ffe66d;margin-bottom:10px;">今週</h3>
                        <div id="weekByLabel" style="display:flex;flex-wrap:wrap;gap:10px;"></div>
                    </div>
                </div>
            </div>
            <div class="card">
                <h2>検出状況（直近3時間）</h2>
                <div id="detection3h"></div>
            </div>
            <div class="card">
                <h2>視聴時間分布</h2>
                <div style="margin-bottom:10px;">
                    <input type="date" id="distributionDate" onchange="loadDistribution()">
                </div>
                <div style="height:200px;"><canvas id="distributionChart"></canvas></div>
            </div>
            <div class="card">
                <h2>視聴時間推移</h2>
                <div style="margin-bottom:10px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                    <input type="date" id="trendStartDate" onchange="loadTrend()">
                    <span>〜</span>
                    <input type="date" id="trendEndDate" onchange="loadTrend()">
                </div>
                <div style="height:200px;"><canvas id="trendChart"></canvas></div>
            </div>
            <div class="card">
                <h2>検出ログ</h2>
                <div id="recentActivity" style="max-height:300px;overflow-y:auto;"></div>
            </div>
        </div>
    </div>

    <div class="modal" id="modal">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <div id="detectionControls" style="display:none;margin-bottom:10px;">
            <label style="color:#fff;margin-right:15px;"><input type="checkbox" id="detModalBbox" onchange="updateDetectionImage()" checked> BBox</label>
            <label style="color:#fff;margin-right:15px;"><input type="checkbox" id="detModalRoi" onchange="updateDetectionImage()" checked> ROI</label>
            <label style="color:#fff;"><input type="checkbox" id="detModalScore" onchange="updateDetectionImage()" checked> スコア</label>
        </div>
        <img id="modalImage" src="">
        <div id="relabelControls" style="display:none;margin-top:15px;background:#16213e;padding:15px;border-radius:8px;max-width:90%;width:400px;">
            <h4 style="color:#ffe66d;margin-bottom:10px;">検出された顔の再ラベリング</h4>
            <div id="relabelFaces"></div>
            <div style="margin-top:15px;text-align:center;">
                <button class="btn btn-success btn-small" onclick="saveRelabels()">変更を保存</button>
                <button class="btn btn-danger btn-small" onclick="deleteDetection()">この検出を削除</button>
            </div>
            <div id="relabelStatus" style="margin-top:10px;text-align:center;"></div>
        </div>
        <div class="modal-controls" id="modalControls">
            <button class="btn btn-danger" onclick="deleteModalImage()">削除</button>
        </div>
    </div>

    <script>
        let currentRoi = null;
        let roiDrawing = false;
        let roiStart = {x: 0, y: 0};
        let modalImagePath = '';
        let modalImageType = 'capture';  // 'capture' or 'face'
        let selectedRoiImage = '';
        let roiPresets = [];
        let currentTab = 'camera';

        // タブ切り替え
        function showTab(tabId) {
            currentTab = tabId;
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`.tab[onclick="showTab('${tabId}')"]`).classList.add('active');
            document.getElementById(tabId).classList.add('active');

            if (tabId === 'camera') { checkCameraStatus(); loadCaptures(); }
            if (tabId === 'roi') { loadRoiImages(); loadRoiPresets(); }
            if (tabId === 'extract') { populateRoiDropdown('extractRoiSelect'); loadExtractImages(); loadExtractedFaces(); }
            if (tabId === 'register') { loadUnregisteredFaces(); loadRegisteredFaces(); loadLabelStatus(); }
            if (tabId === 'test') { initTestTab(); }
            if (tabId === 'settings') { populateRoiDropdown('cfgRoiSelect'); loadConfig(); }
            if (tabId === 'dashboard') { initDashboardDates(); loadDashboard(); loadServiceStatus(); startDashboardRefresh(); }
            else { stopDashboardRefresh(); }
        }

        // カメラ状態チェック
        let serviceImageInterval = null;
        function checkCameraStatus() {
            fetch('/camera_status').then(r => r.json()).then(data => {
                const serviceContainer = document.getElementById('serviceImageContainer');
                const cameraContainer = document.getElementById('cameraContainer');
                const title = document.getElementById('cameraTitle');
                if (data.service_running) {
                    serviceContainer.style.display = 'block';
                    cameraContainer.style.display = 'none';
                    title.textContent = '検出画像';
                    updateServiceImage();
                    startServiceImageRefresh();
                } else {
                    serviceContainer.style.display = 'none';
                    cameraContainer.style.display = 'block';
                    title.textContent = 'リアルタイムプレビュー';
                    stopServiceImageRefresh();
                }
            });
        }

        function updateServiceImage() {
            // サービスの最新フレームを取得（ROI/BBox表示切替対応）
            const img = document.getElementById('serviceImage');
            const showRoi = document.getElementById('camShowRoi')?.checked ?? true;
            const showBbox = document.getElementById('camShowBbox')?.checked ?? true;
            const newSrc = `/api/latest_image?roi=${showRoi}&bbox=${showBbox}&t=${Date.now()}`;
            // 画像が読み込めるか確認
            fetch(newSrc).then(r => {
                if (r.ok) {
                    img.src = newSrc;
                    document.getElementById('serviceImageTime').textContent =
                        `更新: ${new Date().toLocaleTimeString('ja-JP')}`;
                }
            });
        }

        function startServiceImageRefresh() {
            if (!serviceImageInterval) {
                serviceImageInterval = setInterval(updateServiceImage, 5000);
            }
        }

        function stopServiceImageRefresh() {
            if (serviceImageInterval) {
                clearInterval(serviceImageInterval);
                serviceImageInterval = null;
            }
        }

        function showStatus(elementId, message, type) {
            const el = document.getElementById(elementId);
            el.className = 'status ' + type;
            el.textContent = message;
            if (type !== 'info') setTimeout(() => { el.className = ''; el.textContent = ''; }, 5000);
        }

        // 撮影
        function capture() {
            fetch('/capture', {method: 'POST'}).then(r => r.json()).then(data => {
                if (data.success) {
                    showStatus('captureStatus', '撮影完了: ' + data.filename, 'success');
                    loadCaptures();
                } else {
                    showStatus('captureStatus', 'エラー: ' + data.error, 'error');
                }
            });
        }

        function captureServiceFrame() {
            fetch('/capture_service_frame', {method: 'POST'}).then(r => r.json()).then(data => {
                if (data.success) {
                    showStatus('captureStatus', '保存完了: ' + data.filename, 'success');
                    loadCaptures();
                } else {
                    showStatus('captureStatus', 'エラー: ' + data.error, 'error');
                }
            });
        }

        function loadCaptures() {
            fetch('/captures').then(r => r.json()).then(data => {
                const grid = document.getElementById('captureGrid');
                if (data.length === 0) {
                    grid.innerHTML = '<p style="color:#888;">撮影画像なし</p>';
                    return;
                }
                grid.innerHTML = data.map(f => `
                    <div class="grid-item" onclick="showModal('/capture_image/${f}', '${f}')">
                        <img src="/capture_image/${f}">
                        <button class="delete-btn" onclick="event.stopPropagation();deleteCapture('${f}')">&times;</button>
                    </div>
                `).join('');
            });
        }

        function deleteCapture(filename) {
            if (!confirm('削除しますか？')) return;
            fetch('/delete_capture', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: filename})
            }).then(() => loadCaptures());
        }

        // ROI設定
        function loadRoiImages() {
            fetch('/captures').then(r => r.json()).then(data => {
                const grid = document.getElementById('roiImageGrid');
                if (data.length === 0) {
                    grid.innerHTML = '<p style="color:#888;">撮影画像なし</p>';
                    return;
                }
                grid.innerHTML = data.map(f => `
                    <div class="grid-item" onclick="selectRoiImage('${f}', this)">
                        <img src="/capture_image/${f}">
                        <div class="filename">${f}</div>
                    </div>
                `).join('');
            });
        }

        function selectRoiImage(filename, element) {
            document.querySelectorAll('#roiImageGrid .grid-item').forEach(el => el.classList.remove('selected'));
            element.classList.add('selected');
            selectedRoiImage = filename;
            const img = document.getElementById('roiImage');
            img.src = '/capture_image/' + filename;
            img.onload = setupRoiCanvas;
            document.getElementById('roiContainer').style.display = 'block';
            document.getElementById('roiEditControls').style.display = 'block';
            currentRoi = null;
            updateRoiInfo();
        }

        function setupRoiCanvas() {
            const img = document.getElementById('roiImage');
            const canvas = document.getElementById('roiCanvas');
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
            drawRoi();
        }

        function loadRoiPresets() {
            fetch('/api/roi_presets').then(r => r.json()).then(data => {
                roiPresets = data.presets || [];
                renderRoiPresets();
            });
        }

        function populateRoiDropdown(selectId) {
            fetch('/api/roi_presets').then(r => r.json()).then(data => {
                const select = document.getElementById(selectId);
                const currentValue = select.value;
                select.innerHTML = '<option value="">使用しない</option>';
                (data.presets || []).forEach((p, i) => {
                    const opt = document.createElement('option');
                    opt.value = i;
                    opt.textContent = p.name || ('ROI ' + (i+1));
                    select.appendChild(opt);
                });
                if (currentValue && select.querySelector(`option[value="${currentValue}"]`)) {
                    select.value = currentValue;
                }
            });
        }

        let selectedPresetIndex = -1;
        function renderRoiPresets() {
            const container = document.getElementById('roiPresetList');
            if (roiPresets.length === 0) {
                container.innerHTML = '<p style="color:#888;">保存済みROIなし</p>';
                return;
            }
            container.innerHTML = roiPresets.map((p, i) => `
                <div class="roi-preset-item ${selectedPresetIndex === i ? 'selected' : ''}" onclick="selectRoiPreset(${i})" style="cursor:pointer;">
                    <span>${p.name || 'ROI ' + (i+1)}</span>
                    <small style="color:#888;">(${p.x},${p.y} ${p.w}x${p.h})</small>
                    <span class="delete-roi" onclick="event.stopPropagation();deleteRoiPreset(${i})">&times;</span>
                </div>
            `).join('');
        }

        function selectRoiPreset(index) {
            selectedPresetIndex = (selectedPresetIndex === index) ? -1 : index;
            renderRoiPresets();
            drawRoi();
        }

        function deleteRoiPreset(index) {
            if (!confirm('このROIを削除しますか？')) return;
            fetch('/api/roi_presets/delete', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({index: index})
            }).then(() => loadRoiPresets());
        }

        function saveRoiPreset() {
            if (!currentRoi) { alert('ROIを描画してください'); return; }
            const name = 'ROI ' + (roiPresets.length + 1);
            fetch('/api/roi_presets/add', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({roi: {...currentRoi, name: name}})
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    currentRoi = null;
                    drawRoi();
                    updateRoiInfo();
                    loadRoiPresets();
                    showStatus('roiPresetStatus', 'ROI "' + data.name + '" を保存しました', 'success');
                }
            });
        }

        function clearRoiDraw() {
            currentRoi = null;
            updateRoiInfo();
            drawRoi();
        }

        function updateRoiInfo() {
            const el = document.getElementById('roiInfo');
            el.textContent = currentRoi ? `描画中ROI: x=${currentRoi.x}, y=${currentRoi.y}, w=${currentRoi.w}, h=${currentRoi.h}` : 'ROI: 未描画';
        }

        function drawRoi() {
            const canvas = document.getElementById('roiCanvas');
            const ctx = canvas.getContext('2d');
            const img = document.getElementById('roiImage');
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (img.naturalWidth === 0) return;
            const scaleX = canvas.width / img.naturalWidth;
            const scaleY = canvas.height / img.naturalHeight;

            // 選択された保存済みROIを描画
            if (selectedPresetIndex >= 0 && roiPresets[selectedPresetIndex]) {
                const p = roiPresets[selectedPresetIndex];
                const x = p.x * scaleX, y = p.y * scaleY;
                const w = p.w * scaleX, h = p.h * scaleY;
                ctx.strokeStyle = '#ffe66d';
                ctx.lineWidth = 3;
                ctx.setLineDash([]);
                ctx.strokeRect(x, y, w, h);
                ctx.fillStyle = '#ffe66d';
                ctx.font = 'bold 16px sans-serif';
                ctx.fillText(p.name || 'ROI ' + (selectedPresetIndex+1), x + 5, y - 8);
            }

            // 現在描画中のROI
            if (currentRoi) {
                ctx.strokeStyle = '#00d4ff';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 5]);
                ctx.strokeRect(currentRoi.x * scaleX, currentRoi.y * scaleY, currentRoi.w * scaleX, currentRoi.h * scaleY);
                ctx.fillStyle = 'rgba(0,0,0,0.5)';
                ctx.fillRect(0, 0, canvas.width, currentRoi.y * scaleY);
                ctx.fillRect(0, (currentRoi.y + currentRoi.h) * scaleY, canvas.width, canvas.height);
                ctx.fillRect(0, currentRoi.y * scaleY, currentRoi.x * scaleX, currentRoi.h * scaleY);
                ctx.fillRect((currentRoi.x + currentRoi.w) * scaleX, currentRoi.y * scaleY, canvas.width, currentRoi.h * scaleY);
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            const canvas = document.getElementById('roiCanvas');
            canvas.addEventListener('mousedown', (e) => {
                roiDrawing = true;
                const rect = canvas.getBoundingClientRect();
                roiStart = {x: e.clientX - rect.left, y: e.clientY - rect.top};
            });
            canvas.addEventListener('mousemove', (e) => {
                if (!roiDrawing) return;
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                const img = document.getElementById('roiImage');
                const scaleX = img.naturalWidth / canvas.width;
                const scaleY = img.naturalHeight / canvas.height;
                currentRoi = {
                    x: Math.round(Math.min(roiStart.x, x) * scaleX),
                    y: Math.round(Math.min(roiStart.y, y) * scaleY),
                    w: Math.round(Math.abs(x - roiStart.x) * scaleX),
                    h: Math.round(Math.abs(y - roiStart.y) * scaleY)
                };
                updateRoiInfo();
                drawRoi();
            });
            canvas.addEventListener('mouseup', () => { roiDrawing = false; });
            canvas.addEventListener('mouseleave', () => { roiDrawing = false; });
            canvas.addEventListener('touchstart', (e) => {
                e.preventDefault();
                const touch = e.touches[0];
                const rect = canvas.getBoundingClientRect();
                roiDrawing = true;
                roiStart = {x: touch.clientX - rect.left, y: touch.clientY - rect.top};
            });
            canvas.addEventListener('touchmove', (e) => {
                e.preventDefault();
                if (!roiDrawing) return;
                const touch = e.touches[0];
                const rect = canvas.getBoundingClientRect();
                const x = touch.clientX - rect.left;
                const y = touch.clientY - rect.top;
                const img = document.getElementById('roiImage');
                const scaleX = img.naturalWidth / canvas.width;
                const scaleY = img.naturalHeight / canvas.height;
                currentRoi = {
                    x: Math.round(Math.min(roiStart.x, x) * scaleX),
                    y: Math.round(Math.min(roiStart.y, y) * scaleY),
                    w: Math.round(Math.abs(x - roiStart.x) * scaleX),
                    h: Math.round(Math.abs(y - roiStart.y) * scaleY)
                };
                updateRoiInfo();
                drawRoi();
            });
            canvas.addEventListener('touchend', () => { roiDrawing = false; });
            loadServiceStatus();
            checkCameraStatus();
            loadCaptures();
        });

        // 顔抽出
        let selectedExtractImages = new Set();

        function loadExtractImages() {
            const roiIndex = document.getElementById('extractRoiSelect').value;
            fetch('/captures').then(r => r.json()).then(data => {
                const grid = document.getElementById('extractImageGrid');
                if (data.length === 0) {
                    grid.innerHTML = '<p style="color:#888;">撮影画像なし</p>';
                    return;
                }
                grid.innerHTML = data.map(f => `
                    <div class="grid-item" onclick="toggleExtractImage('${f}', this)">
                        <img src="/thumbnail_roi/${f}?roi_index=${roiIndex}&${Date.now()}">
                        <div class="filename">${f}</div>
                    </div>
                `).join('');
                selectedExtractImages.clear();
            });
        }

        function toggleExtractImage(filename, element) {
            if (selectedExtractImages.has(filename)) {
                selectedExtractImages.delete(filename);
                element.classList.remove('selected');
            } else {
                selectedExtractImages.add(filename);
                element.classList.add('selected');
            }
        }

        function extractFaces() {
            if (selectedExtractImages.size === 0) { alert('画像を選択してください'); return; }
            const model = document.getElementById('extractModel').value;
            const upsample = parseInt(document.getElementById('extractUpsample').value);
            const roiIndex = document.getElementById('extractRoiSelect').value;
            const images = Array.from(selectedExtractImages);
            const msg = model === 'cnn' ? '検出中（CNNは時間がかかります）...' : '検出中...';
            showStatus('extractStatus', msg, 'info');

            let completed = 0;
            let totalFaces = 0;
            let hasError = false;

            images.forEach(image => {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 300000);

                fetch('/extract_and_save_faces', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image, model, upsample, roi_index: roiIndex}),
                    signal: controller.signal
                }).then(r => r.json()).then(data => {
                    clearTimeout(timeoutId);
                    if (data.success) totalFaces += data.count;
                    else hasError = true;
                    completed++;
                    showStatus('extractStatus', `処理中... (${completed}/${images.length})`, 'info');
                    if (completed === images.length) {
                        if (hasError) {
                            showStatus('extractStatus', `完了（一部エラー）: ${totalFaces}個の顔を抽出`, 'error');
                        } else {
                            showStatus('extractStatus', `${totalFaces}個の顔を抽出しました`, 'success');
                        }
                        loadExtractedFaces();
                        selectedExtractImages.clear();
                        loadExtractImages();
                    }
                }).catch(err => {
                    clearTimeout(timeoutId);
                    hasError = true;
                    completed++;
                    if (completed === images.length) {
                        showStatus('extractStatus', `エラー: ${err.message}`, 'error');
                        loadExtractedFaces();
                    }
                });
            });
        }

        function loadExtractedFaces() {
            fetch('/all_faces_status').then(r => r.json()).then(data => {
                const container = document.getElementById('extractedFacesList');
                if (data.length === 0) {
                    container.innerHTML = '<p style="color:#888;">抽出済み顔なし</p>';
                    return;
                }
                container.innerHTML = data.map(f => `
                    <div class="face-item">
                        <img src="/face_image/${f.filename}" onclick="openFaceModal('${f.filename}')">
                        <span class="badge ${f.label ? 'badge-registered' : 'badge-unregistered'}">${f.label || '未登録'}</span>
                        <button class="delete-btn" onclick="event.stopPropagation();deleteFace('${f.filename}')">&times;</button>
                    </div>
                `).join('');
            });
        }

        function openFaceModal(filename) {
            modalImagePath = filename;
            modalImageType = 'face';
            document.getElementById('modalImage').src = '/face_image/' + filename;
            document.getElementById('modalControls').style.display = 'block';
            document.getElementById('detectionControls').style.display = 'none';
            document.getElementById('modal').classList.add('active');
        }

        // 顔登録
        let selectedUnregisteredFaces = new Set();

        function loadUnregisteredFaces() {
            fetch('/unregistered_faces').then(r => r.json()).then(data => {
                const container = document.getElementById('unregisteredFaces');
                if (data.length === 0) {
                    container.innerHTML = '<p style="color:#888;">未登録の顔なし</p>';
                    return;
                }
                container.innerHTML = data.map(f => `
                    <div class="face-item" data-file="${f}" onclick="toggleUnregisteredFace('${f}', this)">
                        <img src="/face_image/${f}">
                    </div>
                `).join('');
                selectedUnregisteredFaces.clear();
            });
        }

        function toggleUnregisteredFace(filename, element) {
            if (selectedUnregisteredFaces.has(filename)) {
                selectedUnregisteredFaces.delete(filename);
                element.classList.remove('selected');
            } else {
                selectedUnregisteredFaces.add(filename);
                element.classList.add('selected');
            }
        }

        function selectAllUnregistered() {
            document.querySelectorAll('#unregisteredFaces .face-item').forEach(el => {
                el.classList.add('selected');
                selectedUnregisteredFaces.add(el.dataset.file);
            });
        }

        function deselectAllUnregistered() {
            document.querySelectorAll('#unregisteredFaces .face-item').forEach(el => el.classList.remove('selected'));
            selectedUnregisteredFaces.clear();
        }

        function registerSelectedFaces() {
            const label = document.getElementById('labelName').value.trim().toLowerCase();
            if (!label) { alert('名前を入力してください'); return; }
            if (selectedUnregisteredFaces.size === 0) { alert('顔を選択してください'); return; }
            showStatus('registerStatus', '登録中...', 'info');

            fetch('/register_faces', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({files: Array.from(selectedUnregisteredFaces), label: label})
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    showStatus('registerStatus', `${data.count}件登録・エンコード完了`, 'success');
                    loadUnregisteredFaces();
                    loadRegisteredFaces();
                    loadLabelStatus();
                } else {
                    showStatus('registerStatus', 'エラー: ' + data.error, 'error');
                }
            });
        }

        function loadRegisteredFaces() {
            fetch('/registered_faces_by_label').then(r => r.json()).then(data => {
                const container = document.getElementById('registeredFaces');
                if (Object.keys(data).length === 0) {
                    container.innerHTML = '<p style="color:#888;">登録済み顔なし</p>';
                    return;
                }
                container.innerHTML = Object.entries(data).map(([label, info]) => {
                    const files = info.files || [];
                    const encoded = info.encoded;
                    const statusIcon = encoded ?
                        '<span style="color:#4ecdc4;margin-left:8px;" title="エンコード済み">&#10003;</span>' :
                        '<span style="color:#ff6b6b;margin-left:8px;" title="未エンコード">&#9888;</span>';
                    return `
                        <div class="label-group">
                            <h4>${label} (${files.length}枚) ${statusIcon}</h4>
                            <div>${files.map(f => `
                                <div class="face-item">
                                    <img src="/face_image/${f}" onclick="openFaceModal('${f}')">
                                    <button class="delete-btn" onclick="event.stopPropagation();deleteFace('${f}')">&times;</button>
                                </div>
                            `).join('')}</div>
                        </div>
                    `;
                }).join('');
            });
        }

        function deleteFace(filename) {
            if (!confirm('この写真を削除しますか？')) return;
            fetch('/delete_face', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename})
            }).then(() => {
                loadUnregisteredFaces();
                loadRegisteredFaces();
                loadExtractedFaces();
                loadLabelStatus();
            });
        }

        // ラベル管理
        let editingLabel = null;

        function loadLabelStatus() {
            fetch('/api/label_status').then(r => r.json()).then(data => {
                const container = document.getElementById('labelStatus');
                if (!data.labels || data.labels.length === 0) {
                    container.innerHTML = '<p style="color:#888;">登録済みラベルなし</p>';
                    return;
                }
                let html = '<div style="display:flex;flex-direction:column;gap:8px;">';
                data.labels.forEach(label => {
                    const color = nameColors[label.name] || '#888';
                    const hasImages = label.count > 0;
                    const isEditing = editingLabel === label.name;

                    if (isEditing) {
                        html += `<div style="background:#0f3460;padding:10px 15px;border-radius:8px;border-left:3px solid ${color};display:flex;align-items:center;justify-content:space-between;">
                            <div style="display:flex;align-items:center;gap:10px;flex:1;">
                                <input type="text" id="editLabelInput" value="${label.name}" style="background:#1a1a2e;border:1px solid #4ecdc4;color:#fff;padding:5px 10px;border-radius:4px;width:120px;">
                                <span style="color:#888;font-size:0.9em;">${label.count}枚</span>
                            </div>
                            <div style="display:flex;gap:5px;">
                                <button class="btn btn-primary btn-small" onclick="saveLabel('${label.name}')" style="padding:5px 10px;font-size:0.8em;">保存</button>
                                <button class="btn btn-secondary btn-small" onclick="cancelEditLabel()" style="padding:5px 10px;font-size:0.8em;">取消</button>
                            </div>
                        </div>`;
                    } else {
                        html += `<div style="background:#0f3460;padding:10px 15px;border-radius:8px;border-left:3px solid ${color};display:flex;align-items:center;justify-content:space-between;">
                            <div style="display:flex;align-items:center;gap:15px;">
                                <span style="color:${color};font-weight:bold;min-width:80px;">${label.name}</span>
                                <span style="color:#888;font-size:0.9em;">${label.count}枚</span>
                            </div>
                            <div style="display:flex;gap:5px;">
                                <button class="btn btn-secondary btn-small" onclick="editLabel('${label.name}')" style="padding:5px 10px;font-size:0.8em;">編集</button>
                                <button class="btn btn-danger btn-small" onclick="deleteLabel('${label.name}')" style="padding:5px 10px;font-size:0.8em;">削除</button>
                            </div>
                        </div>`;
                    }
                });
                html += '</div>';
                container.innerHTML = html;

                if (editingLabel) {
                    const input = document.getElementById('editLabelInput');
                    if (input) {
                        input.focus();
                        input.select();
                    }
                }
            });
        }

        function editLabel(name) {
            editingLabel = name;
            loadLabelStatus();
        }

        function cancelEditLabel() {
            editingLabel = null;
            loadLabelStatus();
        }

        function saveLabel(oldName) {
            const input = document.getElementById('editLabelInput');
            const newName = input ? input.value.trim().toLowerCase() : '';

            if (!newName) {
                alert('ラベル名を入力してください');
                return;
            }

            fetch('/api/rename_label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({old_name: oldName, new_name: newName})
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    editingLabel = null;
                    loadLabelStatus();
                    loadRegisteredFaces();
                } else {
                    alert('エラー: ' + (data.error || '変更に失敗しました'));
                }
            }).catch(err => {
                alert('エラー: ' + err.message);
            });
        }

        function deleteLabel(name) {
            if (!confirm(`ラベル "${name}" を削除しますか？\\n・登録済み顔画像のラベルが解除されます\\n・エンコードデータも削除されます`)) return;
            fetch('/api/delete_label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name})
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    loadLabelStatus();
                    loadRegisteredFaces();
                    loadExtractedFaces();
                } else {
                    alert('エラー: ' + (data.error || '削除に失敗しました'));
                }
            });
        }

        // テストタブ
        let currentTestType = 'all';

        function initTestTab() {
            switchTestType('all');
            populateRoiDropdown('detectRoiSelect');
            populateRoiDropdown('recogRoiSelect');
        }

        function switchTestType(type) {
            currentTestType = type;
            document.getElementById('testDetect').style.display = type === 'detect' ? 'block' : 'none';
            document.getElementById('testRecog').style.display = type === 'recog' ? 'block' : 'none';
            document.getElementById('testAll').style.display = type === 'all' ? 'block' : 'none';
            document.getElementById('testTypeDetect').className = 'btn ' + (type === 'detect' ? 'btn-primary' : 'btn-secondary');
            document.getElementById('testTypeRecog').className = 'btn ' + (type === 'recog' ? 'btn-primary' : 'btn-secondary');
            document.getElementById('testTypeAll').className = 'btn ' + (type === 'all' ? 'btn-primary' : 'btn-secondary');
            if (type === 'detect') loadDetectImages();
            if (type === 'recog') loadRecogFaces();
            if (type === 'all') loadRecogImages();
        }

        // 顔検出テスト
        function loadDetectImages() {
            const roiIndex = document.getElementById('detectRoiSelect').value;
            fetch('/captures').then(r => r.json()).then(data => {
                const grid = document.getElementById('detectImageGrid');
                if (data.length === 0) {
                    grid.innerHTML = '<p style="color:#888;">撮影画像なし</p>';
                    return;
                }
                grid.innerHTML = data.map(f => `
                    <div class="grid-item" onclick="selectDetectImage('${f}', this)">
                        <img src="/thumbnail_roi/${f}?roi_index=${roiIndex}&${Date.now()}">
                        <div class="filename">${f}</div>
                    </div>
                `).join('');
            });
        }

        function selectDetectImage(filename, element) {
            document.querySelectorAll('#detectImageGrid .grid-item').forEach(el => el.classList.remove('selected'));
            element.classList.add('selected');
            document.getElementById('detectImage').value = filename;
        }

        let detectInProgress = false;
        let lastDetectHasRoi = false;

        function runDetection(btn) {
            if (detectInProgress) { alert('処理中です。しばらくお待ちください。'); return; }
            const image = document.getElementById('detectImage').value;
            const model = document.getElementById('detectModel').value;
            const upsample = document.getElementById('detectUpsample').value;
            const roiIndex = document.getElementById('detectRoiSelect').value;
            if (!image) { alert('画像を選択してください'); return; }
            const msg = model === 'cnn' ? '検出中（CNNは2-3分かかる場合があります）...' : '検出中...';
            showStatus('detectStatus', msg, 'info');
            if (btn) { btn.disabled = true; btn.textContent = '処理中...'; }
            detectInProgress = true;

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 300000);

            fetch('/detect_only', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image, model, upsample: parseInt(upsample), roi_index: roiIndex}),
                signal: controller.signal
            }).then(r => r.json()).then(data => {
                clearTimeout(timeoutId);
                detectInProgress = false;
                if (btn) { btn.disabled = false; btn.textContent = '顔検出実行'; }
                if (data.success) {
                    lastDetectHasRoi = data.roi_used;
                    const roiText = data.roi_used ? ' [ROI適用]' : '';
                    showStatus('detectStatus', `検出完了: ${data.count}人検出 (${data.time}秒)${roiText}`, 'success');
                    renderDetectResult(data);
                } else {
                    showStatus('detectStatus', 'エラー: ' + data.error, 'error');
                }
            }).catch(err => {
                clearTimeout(timeoutId);
                detectInProgress = false;
                if (btn) { btn.disabled = false; btn.textContent = '顔検出実行'; }
                if (err.name === 'AbortError') {
                    showStatus('detectStatus', 'タイムアウト: 処理に時間がかかりすぎました', 'error');
                } else {
                    showStatus('detectStatus', 'エラー: ' + err.message, 'error');
                }
            });
        }

        let lastDetectData = null;
        function renderDetectResult(data) {
            lastDetectData = data;
            const result = document.getElementById('detectResult');
            const showBbox = document.getElementById('detectShowBbox')?.checked ?? true;
            const showRoi = document.getElementById('detectShowRoi')?.checked ?? true;
            const ts = Date.now();

            let html = `
                <div style="display:flex;gap:20px;margin-bottom:10px;justify-content:center;">
                    <label style="display:flex;align-items:center;gap:5px;cursor:pointer;">
                        <input type="checkbox" id="detectShowBbox" onchange="updateDetectImage()" ${showBbox ? 'checked' : ''}> BBox表示
                    </label>
                    <label style="display:flex;align-items:center;gap:5px;cursor:pointer;${lastDetectHasRoi ? '' : 'opacity:0.5;'}">
                        <input type="checkbox" id="detectShowRoi" onchange="updateDetectImage()" ${showRoi ? 'checked' : ''} ${lastDetectHasRoi ? '' : 'disabled'}> ROI表示
                    </label>
                </div>
                <img id="detectResultImg" src="/detect_result_render?show_bbox=${showBbox}&show_roi=${showRoi}&t=${ts}" style="width:100%;border-radius:8px;">
            `;

            if (data.count === 0) {
                html += '<p style="color:#ff6b6b;margin-top:10px;">顔が検出されませんでした</p>';
            } else {
                html += `<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;">
                    ${data.faces.map((f, i) => `
                        <div class="face-box">
                            <img src="/detect_face/${i}?${ts}">
                            <div style="font-size:0.9em;color:#4ecdc4;">顔 ${i + 1}</div>
                            <div style="font-size:0.8em;color:#888;">${f.width}x${f.height}</div>
                        </div>
                    `).join('')}
                </div>`;
            }
            result.innerHTML = html;
        }

        function updateDetectImage() {
            const showBbox = document.getElementById('detectShowBbox')?.checked ?? true;
            const showRoi = document.getElementById('detectShowRoi')?.checked ?? true;
            const img = document.getElementById('detectResultImg');
            if (img) {
                img.src = `/detect_result_render?show_bbox=${showBbox}&show_roi=${showRoi}&t=${Date.now()}`;
            }
        }

        // 顔認識テスト（顔画像入力）
        function loadRecogFaces() {
            fetch('/all_faces_status').then(r => r.json()).then(data => {
                const grid = document.getElementById('recogFaceGrid');
                if (data.length === 0) {
                    grid.innerHTML = '<p style="color:#888;">抽出済み顔なし</p>';
                    return;
                }
                grid.innerHTML = data.map(f => `
                    <div class="face-item" onclick="selectRecogFace('${f.filename}', this)">
                        <img src="/face_image/${f.filename}">
                        <span class="badge ${f.label ? 'badge-registered' : 'badge-unregistered'}">${f.label || '未登録'}</span>
                    </div>
                `).join('');
            });
        }

        function selectRecogFace(filename, element) {
            document.querySelectorAll('#recogFaceGrid .face-item').forEach(el => el.classList.remove('selected'));
            element.classList.add('selected');
            document.getElementById('recogFaceFile').value = filename;
        }

        function runRecogOnly() {
            const faceFile = document.getElementById('recogFaceFile').value;
            const similarityThreshold = document.getElementById('recogOnlyTolerance').value;
            const tolerance = 1 - parseFloat(similarityThreshold) / 100;
            if (!faceFile) { alert('顔画像を選択してください'); return; }
            showStatus('recogOnlyStatus', '認識中...', 'info');

            fetch('/recognize_face', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({face_file: faceFile, tolerance: tolerance})
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    const color = nameColors[data.name] || '#888';
                    const similarity = Math.max(0, (1 - data.distance) * 100).toFixed(1);
                    showStatus('recogOnlyStatus', '認識完了', 'success');

                    // 全ラベルの類似度をソート（類似度高い順）
                    const allDist = data.all_distances || {};
                    const sortedLabels = Object.entries(allDist)
                        .map(([label, dist]) => ({label, dist, similarity: Math.max(0, (1 - dist) * 100)}))
                        .sort((a, b) => b.similarity - a.similarity);

                    let labelsHtml = sortedLabels.map(item => {
                        const labelColor = nameColors[item.label] || '#888';
                        const isMatch = item.label === data.name && data.name !== 'unknown';
                        return `<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;background:${isMatch ? '#1a4a3a' : '#1a1a2e'};border-radius:4px;border-left:3px solid ${labelColor};">
                            <span style="color:${labelColor};font-weight:${isMatch ? 'bold' : 'normal'};">${item.label}</span>
                            <span style="color:${item.similarity >= 50 ? '#4ecdc4' : '#888'};">${item.similarity.toFixed(1)}%</span>
                        </div>`;
                    }).join('');

                    document.getElementById('recogOnlyResult').innerHTML = `
                        <div style="display:flex;gap:20px;background:#0f3460;padding:20px;border-radius:8px;">
                            <div style="flex-shrink:0;">
                                <img src="/face_image/${faceFile}" style="width:100px;height:100px;object-fit:cover;border-radius:8px;">
                            </div>
                            <div style="flex:1;">
                                <div style="font-size:1.3em;font-weight:bold;color:${color};margin-bottom:10px;">
                                    判定結果: ${data.name} (${similarity}%)
                                </div>
                                <div style="font-size:0.9em;color:#888;margin-bottom:8px;">各ラベルとの類似度:</div>
                                <div style="display:flex;flex-direction:column;gap:6px;">
                                    ${labelsHtml}
                                </div>
                            </div>
                        </div>
                    `;
                } else {
                    showStatus('recogOnlyStatus', 'エラー: ' + data.error, 'error');
                }
            });
        }

        // 総合テスト（既存の顔認識テスト）
        function loadRecogImages() {
            const roiIndex = document.getElementById('recogRoiSelect').value;
            fetch('/captures').then(r => r.json()).then(data => {
                const grid = document.getElementById('recogImageGrid');
                if (data.length === 0) {
                    grid.innerHTML = '<p style="color:#888;">撮影画像なし</p>';
                    return;
                }
                grid.innerHTML = data.map(f => `
                    <div class="grid-item" onclick="selectRecogImage('${f}', this)">
                        <img src="/thumbnail_roi/${f}?roi_index=${roiIndex}&${Date.now()}">
                        <div class="filename">${f}</div>
                    </div>
                `).join('');
            });
        }

        function selectRecogImage(filename, element) {
            document.querySelectorAll('#recogImageGrid .grid-item').forEach(el => el.classList.remove('selected'));
            element.classList.add('selected');
            document.getElementById('recogImage').value = filename;
        }

        let recogInProgress = false;
        let lastRecogHasRoi = false;
        let lastRecogData = null;

        function runRecognition(btn) {
            if (recogInProgress) { alert('処理中です。しばらくお待ちください。'); return; }
            const image = document.getElementById('recogImage').value;
            const model = document.getElementById('recogModel').value;
            const upsample = document.getElementById('recogUpsample').value;
            const similarityThreshold = document.getElementById('recogTolerance').value;
            const tolerance = 1 - parseFloat(similarityThreshold) / 100;
            const roiIndex = document.getElementById('recogRoiSelect').value;
            if (!image) { alert('画像を選択してください'); return; }
            const msg = model === 'cnn' ? '認識中（CNNは2-3分かかる場合があります）...' : '認識中...';
            showStatus('recogStatus', msg, 'info');
            if (btn) { btn.disabled = true; btn.textContent = '処理中...'; }
            recogInProgress = true;

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 300000);

            fetch('/recognize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image, model, upsample: parseInt(upsample), tolerance: tolerance, roi_index: roiIndex}),
                signal: controller.signal
            }).then(r => r.json()).then(data => {
                clearTimeout(timeoutId);
                recogInProgress = false;
                if (btn) { btn.disabled = false; btn.textContent = '顔認識実行'; }
                if (data.success) {
                    lastRecogHasRoi = data.roi_used;
                    const roiText = data.roi_used ? ' [ROI適用]' : '';
                    showStatus('recogStatus', `認識完了: ${data.faces.length}人検出 (${data.time}秒)${roiText}`, 'success');
                    renderRecogResult(data);
                } else {
                    showStatus('recogStatus', 'エラー: ' + data.error, 'error');
                }
            }).catch(err => {
                clearTimeout(timeoutId);
                recogInProgress = false;
                if (btn) { btn.disabled = false; btn.textContent = '顔認識実行'; }
                if (err.name === 'AbortError') {
                    showStatus('recogStatus', 'タイムアウト: 処理に時間がかかりすぎました', 'error');
                } else {
                    showStatus('recogStatus', 'エラー: ' + err.message, 'error');
                }
            });
        }

        function renderRecogResult(data) {
            lastRecogData = data;
            const result = document.getElementById('recogResult');
            const showBbox = document.getElementById('recogShowBbox')?.checked ?? true;
            const showRoi = document.getElementById('recogShowRoi')?.checked ?? true;
            const ts = Date.now();
            const nameColors = {'mio': '#ff6b6b', 'yu': '#4ecdc4', 'tsubasa': '#ffe66d', 'unknown': '#888'};

            let html = `
                <div style="display:flex;gap:20px;margin-bottom:10px;justify-content:center;">
                    <label style="display:flex;align-items:center;gap:5px;cursor:pointer;">
                        <input type="checkbox" id="recogShowBbox" onchange="updateRecogImage()" ${showBbox ? 'checked' : ''}> BBox表示
                    </label>
                    <label style="display:flex;align-items:center;gap:5px;cursor:pointer;${lastRecogHasRoi ? '' : 'opacity:0.5;'}">
                        <input type="checkbox" id="recogShowRoi" onchange="updateRecogImage()" ${showRoi ? 'checked' : ''} ${lastRecogHasRoi ? '' : 'disabled'}> ROI表示
                    </label>
                </div>
                <img id="recogResultImg" src="/recog_result_render?show_bbox=${showBbox}&show_roi=${showRoi}&t=${ts}" style="width:100%;border-radius:8px;">
            `;

            if (data.faces.length === 0) {
                html += '<p style="color:#ff6b6b;margin-top:10px;">顔が検出されませんでした</p>';
            } else {
                html += `<div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;">
                    ${data.faces.map((f, i) => `
                        <div class="face-box" style="border-left:4px solid ${nameColors[f.name] || '#888'};">
                            <img src="/recog_face/${i}?${ts}">
                            <div style="color:${nameColors[f.name] || '#888'};font-weight:bold;">${f.name}</div>
                            <div style="font-size:0.8em;color:#888;">類似度: ${Math.max(0, (1 - f.distance) * 100).toFixed(1)}%</div>
                        </div>
                    `).join('')}
                </div>`;
            }
            result.innerHTML = html;
        }

        function updateRecogImage() {
            const showBbox = document.getElementById('recogShowBbox')?.checked ?? true;
            const showRoi = document.getElementById('recogShowRoi')?.checked ?? true;
            const img = document.getElementById('recogResultImg');
            if (img) {
                img.src = `/recog_result_render?show_bbox=${showBbox}&show_roi=${showRoi}&t=${Date.now()}`;
            }
        }

        // モーダル
        function showModal(src, path) {
            modalImagePath = path;
            modalImageType = 'capture';
            document.getElementById('modalImage').src = src;
            document.getElementById('modalControls').style.display = 'block';
            document.getElementById('detectionControls').style.display = 'none';
            document.getElementById('modal').classList.add('active');
        }

        function closeModal() { document.getElementById('modal').classList.remove('active'); }

        let currentDetectionTimestamp = '';

        let currentDetectionMeta = null;
        let registeredLabels = [];

        function showDetectionModal(images, timestamp) {
            if (!images || images.length === 0) {
                alert('この検出の画像がありません');
                return;
            }
            // タイムスタンプを抽出 (detection_YYYYMMDD_HHMMSS_name.jpg -> YYYYMMDD_HHMMSS)
            const firstImage = images[0];
            const match = firstImage.match(/detection_(\d{8}_\d{6})_/);
            if (match) {
                currentDetectionTimestamp = match[1];
                // チェックボックスをリセット
                document.getElementById('detModalBbox').checked = true;
                document.getElementById('detModalRoi').checked = true;
                document.getElementById('detModalScore').checked = true;
                updateDetectionImage();
                document.getElementById('detectionControls').style.display = 'block';
                // 再ラベリング用データを読み込む
                loadRelabelData();
            } else {
                // 旧形式の場合はそのまま表示
                document.getElementById('modalImage').src = '/detection_image/' + firstImage + '?t=' + Date.now();
                document.getElementById('detectionControls').style.display = 'none';
                document.getElementById('relabelControls').style.display = 'none';
            }
            modalImagePath = firstImage;
            modalImageType = 'detection';
            document.getElementById('modalControls').style.display = 'none';
            document.getElementById('relabelStatus').textContent = '';
            document.getElementById('modal').classList.add('active');
        }

        function loadRelabelData() {
            // 登録済みラベル一覧を取得
            fetch('/api/label_status').then(r => r.json()).then(data => {
                registeredLabels = Object.keys(data);
                // メタデータを取得
                fetch(`/api/detection_meta/${currentDetectionTimestamp}`).then(r => r.json()).then(meta => {
                    currentDetectionMeta = meta;
                    renderRelabelFaces();
                    document.getElementById('relabelControls').style.display = 'block';
                }).catch(() => {
                    document.getElementById('relabelControls').style.display = 'none';
                });
            });
        }

        function renderRelabelFaces() {
            const container = document.getElementById('relabelFaces');
            if (!currentDetectionMeta || !currentDetectionMeta.faces || currentDetectionMeta.faces.length === 0) {
                container.innerHTML = '<p style="color:#888;">顔データがありません</p>';
                return;
            }
            const options = registeredLabels.map(l => `<option value="${l}">${l}</option>`).join('') + '<option value="unknown">unknown</option><option value="__new__">+ 新規入力...</option>';
            container.innerHTML = currentDetectionMeta.faces.map((face, i) => `
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;padding:10px;background:#0f3460;border-radius:6px;flex-wrap:wrap;">
                    <div style="min-width:60px;text-align:center;">
                        <div style="color:${nameColors[face.name] || '#888'};font-weight:bold;">${face.name}</div>
                        <div style="font-size:0.8em;color:#888;">${face.similarity?.toFixed(0) || 0}%</div>
                    </div>
                    <span style="color:#888;">→</span>
                    <select id="relabel_${i}" onchange="toggleNewLabelInput(${i})" style="flex:1;min-width:100px;padding:8px;border-radius:4px;background:#1a1a2e;color:#fff;border:1px solid #333;">
                        ${options.replace(`value="${face.name}"`, `value="${face.name}" selected`)}
                    </select>
                    <input type="text" id="relabel_new_${i}" placeholder="新しいラベル名" style="display:none;flex:1;min-width:100px;padding:8px;border-radius:4px;background:#1a1a2e;color:#fff;border:1px solid #4ecdc4;">
                </div>
            `).join('');
        }

        function toggleNewLabelInput(index) {
            const select = document.getElementById(`relabel_${index}`);
            const input = document.getElementById(`relabel_new_${index}`);
            if (select.value === '__new__') {
                input.style.display = 'block';
                input.focus();
            } else {
                input.style.display = 'none';
                input.value = '';
            }
        }

        function saveRelabels() {
            if (!currentDetectionMeta || !currentDetectionTimestamp) return;
            const updates = [];
            currentDetectionMeta.faces.forEach((face, i) => {
                const select = document.getElementById(`relabel_${i}`);
                const newInput = document.getElementById(`relabel_new_${i}`);
                let newName = select.value;
                if (newName === '__new__') {
                    newName = newInput.value.trim();
                    if (!newName) {
                        return; // 空の場合はスキップ
                    }
                }
                if (newName !== face.name) {
                    updates.push({index: i, old_name: face.name, new_name: newName});
                }
            });
            if (updates.length === 0) {
                document.getElementById('relabelStatus').innerHTML = '<span style="color:#888;">変更なし</span>';
                return;
            }
            fetch('/api/relabel_detection', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({timestamp: currentDetectionTimestamp, updates: updates})
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    document.getElementById('relabelStatus').innerHTML = '<span style="color:#4ecdc4;">保存しました</span>';
                    loadRelabelData();
                    updateDetectionImage();
                    loadDashboardFast();
                } else {
                    document.getElementById('relabelStatus').innerHTML = `<span style="color:#ff6b6b;">エラー: ${data.error}</span>`;
                }
            });
        }

        function deleteDetection() {
            if (!currentDetectionTimestamp) return;
            if (!confirm('この検出記録を削除しますか？\\nCSVログとメタデータが削除されます。')) return;
            fetch('/api/delete_detection', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({timestamp: currentDetectionTimestamp})
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    closeModal();
                    loadDashboardFast();
                } else {
                    alert('削除エラー: ' + data.error);
                }
            });
        }

        function updateDetectionImage() {
            if (!currentDetectionTimestamp) return;
            const bbox = document.getElementById('detModalBbox').checked;
            const roi = document.getElementById('detModalRoi').checked;
            const score = document.getElementById('detModalScore').checked;
            document.getElementById('modalImage').src = `/detection_render/${currentDetectionTimestamp}?bbox=${bbox}&roi=${roi}&score=${score}&t=${Date.now()}`;
        }

        function deleteModalImage() {
            if (!confirm('削除しますか？')) return;
            if (modalImageType === 'face') {
                fetch('/delete_face', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({filename: modalImagePath})
                }).then(() => {
                    closeModal();
                    loadExtractedFaces();
                    loadUnregisteredFaces();
                    loadRegisteredFaces();
                });
            } else {
                fetch('/delete_capture', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({filename: modalImagePath})
                }).then(() => { closeModal(); loadCaptures(); });
            }
        }

        // ダッシュボード
        let dashboardFastInterval = null;
        let dashboardMinuteInterval = null;
        let dashboardHourInterval = null;
        const nameColors = {'mio': '#ff6b6b', 'yu': '#4ecdc4', 'tsubasa': '#ffe66d', 'unknown': '#888'};
        let distributionChart = null, trendChart = null;
        let latestImageFilename = '';

        function startDashboardRefresh() {
            stopDashboardRefresh();
            // 直近の画像と検出ログ: 10秒周期
            dashboardFastInterval = setInterval(() => {
                if (currentTab === 'dashboard') { loadDashboardFast(); }
            }, 10000);
            // 視聴時間・検出状況・視聴時間分布: 1分周期
            dashboardMinuteInterval = setInterval(() => {
                if (currentTab === 'dashboard') { loadDashboardMinute(); loadDistribution(); }
            }, 60000);
            // 視聴時間推移: 1時間周期
            dashboardHourInterval = setInterval(() => {
                if (currentTab === 'dashboard') { loadTrend(); }
            }, 3600000);
        }

        function stopDashboardRefresh() {
            if (dashboardFastInterval) { clearInterval(dashboardFastInterval); dashboardFastInterval = null; }
            if (dashboardMinuteInterval) { clearInterval(dashboardMinuteInterval); dashboardMinuteInterval = null; }
            if (dashboardHourInterval) { clearInterval(dashboardHourInterval); dashboardHourInterval = null; }
        }

        function initDashboardDates() {
            const today = new Date().toISOString().slice(0, 10);
            const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
            document.getElementById('distributionDate').value = today;
            document.getElementById('trendStartDate').value = weekAgo;
            document.getElementById('trendEndDate').value = today;
        }

        function updateLatestImage() {
            if (!latestImageFilename) return;
            const showRoi = document.getElementById('showRoi').checked;
            const showBbox = document.getElementById('showBbox').checked;
            document.getElementById('latestImage').src = `/api/latest_image?roi=${showRoi}&bbox=${showBbox}&t=${Date.now()}`;
        }

        // 直近の画像と検出ログ（10秒周期）
        function loadDashboardFast() {
            fetch('/api/dashboard').then(r => r.json()).then(data => {
                // ROI名称表示
                const roiName = data.roi_name || '';
                document.getElementById('roiNameDisplay').textContent = roiName ? `ROI: ${roiName}` : '';

                // 直近の画像
                if (data.latest_image) {
                    latestImageFilename = data.latest_image;
                    document.getElementById('latestImage').style.display = 'block';
                    document.getElementById('noLatestImage').style.display = 'none';
                    updateLatestImage();
                } else {
                    document.getElementById('latestImage').style.display = 'none';
                    document.getElementById('noLatestImage').style.display = 'block';
                }

                // 検出ログ（同時検出は1レコードにまとめ）
                const recentHtml = (data.recent_grouped || []).slice(0, 30).map(e => {
                    const namesHtml = e.names.map(n => `<span style="color:${nameColors[n] || '#888'};margin-left:8px;">${n}</span>`).join('');
                    const images = JSON.stringify(e.images || []);
                    return `<div style="padding:6px 10px;border-bottom:1px solid #333;display:flex;justify-content:space-between;align-items:center;cursor:pointer;" onclick='showDetectionModal(${images}, "${e.timestamp}")'><span style="color:#888;">${e.timestamp}</span><div>${namesHtml}</div></div>`;
                }).join('');
                document.getElementById('recentActivity').innerHTML = recentHtml || '<p style="color:#888;padding:10px;">データなし</p>';
            });
        }

        // 視聴時間と検出状況（1分周期）
        function loadDashboardMinute() {
            fetch('/api/dashboard').then(r => r.json()).then(data => {
                const today = new Date().toISOString().slice(0, 10);
                const names = data.registered_labels || [];

                // 視聴時間（本日・今週）
                let todayHtml = '';
                names.forEach(name => {
                    const mins = data.daily[today]?.[name] || 0;
                    const color = nameColors[name] || '#888';
                    todayHtml += `<div style="background:#0f3460;padding:10px 15px;border-radius:8px;text-align:center;border-left:3px solid ${color};">
                        <div style="color:${color};font-weight:bold;font-size:0.9em;">${name}</div>
                        <div style="font-size:1.5em;font-weight:bold;">${Math.round(mins)}<span style="font-size:0.5em;color:#888;">分</span></div>
                    </div>`;
                });
                document.getElementById('todayByLabel').innerHTML = todayHtml || '<p style="color:#888;">データなし</p>';

                let weekHtml = '';
                names.forEach(name => {
                    let total = 0;
                    Object.values(data.daily).forEach(day => { total += day[name] || 0; });
                    const color = nameColors[name] || '#888';
                    weekHtml += `<div style="background:#0f3460;padding:10px 15px;border-radius:8px;text-align:center;border-left:3px solid ${color};">
                        <div style="color:${color};font-weight:bold;font-size:0.9em;">${name}</div>
                        <div style="font-size:1.5em;font-weight:bold;">${Math.round(total)}<span style="font-size:0.5em;color:#888;">分</span></div>
                    </div>`;
                });
                document.getElementById('weekByLabel').innerHTML = weekHtml || '<p style="color:#888;">データなし</p>';

                // 検出状況（直近3時間）- データがなくても構造を表示
                let html3h = '';
                if (names.length === 0) {
                    const emptyBars = Array(180).fill(0).map(() => '<div style="width:2px;height:24px;background:#333;"></div>').join('');
                    html3h = `<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;padding:8px;background:#0f3460;border-radius:6px;">
                        <div style="color:#888;font-weight:bold;width:60px;">-</div>
                        <div style="display:flex;gap:1px;flex:1;align-items:center;">
                            <span style="color:#666;font-size:0.7em;width:30px;">3h前</span>
                            ${emptyBars}
                            <span style="color:#666;font-size:0.7em;width:25px;text-align:right;">now</span>
                        </div>
                    </div>`;
                } else {
                    names.forEach(name => {
                        const color = nameColors[name] || '#888';
                        const bars = data.detection_3h?.[name] || Array(180).fill(false);
                        const barsHtml = bars.map(v => `<div style="width:2px;height:24px;background:${v ? color : '#333'};"></div>`).join('');
                        html3h += `<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;padding:8px;background:#0f3460;border-radius:6px;">
                            <div style="color:${color};font-weight:bold;width:60px;">${name}</div>
                            <div style="display:flex;gap:1px;flex:1;align-items:center;">
                                <span style="color:#666;font-size:0.7em;width:30px;">3h前</span>
                                ${barsHtml}
                                <span style="color:#666;font-size:0.7em;width:25px;text-align:right;">now</span>
                            </div>
                        </div>`;
                    });
                }
                document.getElementById('detection3h').innerHTML = html3h;
            });
        }

        // 初回読み込み用（全て読み込む）
        function loadDashboard() {
            loadDashboardFast();
            loadDashboardMinute();
            loadDistribution();
            loadTrend();
        }

        function loadDistribution() {
            const date = document.getElementById('distributionDate').value;
            if (!date) return;
            fetch(`/api/distribution?date=${date}`).then(r => r.json()).then(data => {
                const names = data.labels || [];
                const hours = Array.from({length: 24}, (_, i) => String(i).padStart(2, '0'));
                const datasets = names.map(name => ({
                    label: name, data: hours.map(h => Math.round(data.hourly[h]?.[name] || 0)),
                    borderColor: nameColors[name] || '#888', backgroundColor: 'transparent', tension: 0.3
                }));
                if (distributionChart) distributionChart.destroy();
                distributionChart = new Chart(document.getElementById('distributionChart'), {
                    type: 'line', data: { labels: hours.map(h => h + ':00'), datasets },
                    options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { color: '#888' }, grid: { color: '#333' } }, y: { min: 0, ticks: { color: '#888' }, grid: { color: '#333' } } }, plugins: { legend: { labels: { color: '#eee' } } } }
                });
            });
        }

        function loadTrend() {
            const start = document.getElementById('trendStartDate').value;
            const end = document.getElementById('trendEndDate').value;
            if (!start || !end) return;
            fetch(`/api/trend?start=${start}&end=${end}`).then(r => r.json()).then(data => {
                const names = data.labels || [];
                const dates = data.dates || [];
                const datasets = names.map(name => ({
                    label: name, data: dates.map(d => Math.round(data.daily[d]?.[name] || 0)),
                    borderColor: nameColors[name] || '#888', backgroundColor: 'transparent', tension: 0.3
                }));
                if (trendChart) trendChart.destroy();
                trendChart = new Chart(document.getElementById('trendChart'), {
                    type: 'line', data: { labels: dates.map(d => d.slice(5)), datasets },
                    options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { color: '#888' }, grid: { color: '#333' } }, y: { min: 0, ticks: { color: '#888' }, grid: { color: '#333' } } }, plugins: { legend: { labels: { color: '#eee' } } } }
                });
            });
        }

        function loadServiceStatus() {
            fetch('/api/service_status').then(r => r.json()).then(data => {
                const el = document.getElementById('serviceStatus');
                if (el) {
                    if (data.running) { el.textContent = '稼働中'; el.style.background = '#4ecdc4'; el.style.color = '#000'; }
                    else { el.textContent = '停止中'; el.style.background = '#ff6b6b'; el.style.color = '#fff'; }
                }
                updateCfgServiceStatus(data.running);
            });
        }

        function serviceControl(action) {
            fetch('/api/service_control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action}) })
            .then(r => r.json()).then(data => {
                setTimeout(() => {
                    loadServiceStatus();
                    loadConfig();
                }, 2000);
                if (data.error) {
                    alert(data.error);
                } else if (action === 'stop') {
                    // サービス停止後、カメラタブにいる場合はプレビューを再開
                    setTimeout(() => {
                        if (currentTab === 'camera') {
                            // カメラプレビューを再開
                            fetch('/start_camera', { method: 'POST' }).then(r => r.json()).then(d => {
                                if (d.success) {
                                    checkCameraStatus();
                                    document.getElementById('cameraPreview').src = '/stream?' + Date.now();
                                }
                            });
                        }
                    }, 500);
                } else if (action === 'start') {
                    // サービス開始後、カメラタブにいる場合は検出画像表示に切り替え
                    setTimeout(() => {
                        if (currentTab === 'camera') {
                            checkCameraStatus();
                        }
                    }, 500);
                }
            });
        }

        function formatConfigDisplay(cfg) {
            const interval = cfg.interval_sec || 5;
            const intervalText = interval >= 60 ? `${interval / 60}分` : `${interval}秒`;
            const tolerance = cfg.tolerance || 0.5;
            const similarity = Math.round((1 - tolerance) * 100);
            const roiText = cfg.roi_index ? `ROI ${cfg.roi_index}` : 'なし';
            return `検出モデル: ${cfg.face_model || 'hog'}<br>UpSample: ${cfg.upsample || 0}<br>撮影間隔: ${intervalText}<br>類似度閾値: ${similarity}%<br>ROI: ${roiText}`;
        }

        function updateCfgServiceStatus(running) {
            const el = document.getElementById('cfgServiceStatus');
            if (el) {
                if (running) {
                    el.textContent = '稼働中';
                    el.style.background = '#4ecdc4';
                    el.style.color = '#000';
                } else {
                    el.textContent = '停止中';
                    el.style.background = '#ff6b6b';
                    el.style.color = '#fff';
                }
            }
        }

        function loadConfig() {
            // 保存済み設定をフォームに読み込む
            fetch('/api/config').then(r => r.json()).then(cfg => {
                document.getElementById('cfgModel').value = cfg.face_model || 'hog';
                document.getElementById('cfgUpsample').value = cfg.upsample || 0;
                document.getElementById('cfgInterval').value = cfg.interval_sec || 5;
                const tolerance = cfg.tolerance || 0.5;
                const similarity = Math.round((1 - tolerance) * 100);
                document.getElementById('cfgTolerance').value = similarity;
                if (cfg.roi_index !== undefined && cfg.roi_index !== null && cfg.roi_index !== '') {
                    setTimeout(() => { document.getElementById('cfgRoiSelect').value = cfg.roi_index; }, 500);
                }
            });
            // 適用中の設定とサービス状態を読み込む
            fetch('/api/applied_config').then(r => r.json()).then(data => {
                updateCfgServiceStatus(data.running);
                if (data.running && data.config) {
                    document.getElementById('appliedConfigDisplay').innerHTML = formatConfigDisplay(data.config);
                } else {
                    document.getElementById('appliedConfigDisplay').innerHTML = '<span style="color:#888;">サービス停止中</span>';
                }
            });
        }

        function saveAndApplyConfig() {
            if (!confirm('設定を保存してサービスを再起動しますか？')) return;
            const st = document.getElementById('configStatus');
            const similarityThreshold = parseFloat(document.getElementById('cfgTolerance').value);
            const tolerance = 1 - similarityThreshold / 100;
            const cfg = {
                face_model: document.getElementById('cfgModel').value,
                upsample: parseInt(document.getElementById('cfgUpsample').value),
                interval_sec: parseInt(document.getElementById('cfgInterval').value),
                tolerance: tolerance,
                roi_index: document.getElementById('cfgRoiSelect').value
            };

            st.textContent = '保存中...';
            st.style.color = '#ffe66d';

            // 現在のタイムスタンプを記録（新しい設定ファイルがこれより新しいか確認用）
            const restartTime = Date.now() / 1000;

            fetch('/api/config', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(cfg) })
            .then(r => r.json()).then(data => {
                if (!data.success) {
                    st.textContent = 'エラー: ' + data.error;
                    st.style.color = '#ff6b6b';
                    return null;
                }
                st.textContent = '再起動中...';
                return fetch('/api/service_control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action: 'restart'}) });
            })
            .then(r => r ? r.json() : null)
            .then(data => {
                if (!data) return;
                if (data.success) {
                    st.textContent = 'サービス起動待機中...';
                    // サービス起動を待つ（ポーリング）- 設定ファイルが更新されるまで待機
                    let retryCount = 0;
                    const checkService = () => {
                        fetch('/api/applied_config?since=' + restartTime).then(r => r.json()).then(result => {
                            if (result.running && result.config && !result.waiting) {
                                st.textContent = '設定を反映しました';
                                st.style.color = '#4ecdc4';
                                document.getElementById('appliedConfigDisplay').innerHTML = formatConfigDisplay(result.config);
                                updateCfgServiceStatus(true);
                                setTimeout(() => st.textContent = '', 3000);
                            } else if (retryCount < 15) {
                                retryCount++;
                                setTimeout(checkService, 1000);
                            } else {
                                st.textContent = 'サービス起動待機タイムアウト';
                                st.style.color = '#ff6b6b';
                                loadServiceStatus();
                            }
                        });
                    };
                    setTimeout(checkService, 2000);
                } else {
                    st.textContent = 'エラー: ' + (data.error || '再起動失敗');
                    st.style.color = '#ff6b6b';
                    setTimeout(() => st.textContent = '', 5000);
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/manual.html")
def manual():
    manual_path = os.path.join(BASE_DIR, "manual.html")
    if os.path.exists(manual_path):
        with open(manual_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Manual not found", 404

@app.route("/start_camera", methods=["POST"])
def start_camera():
    if is_service_running():
        os.system("sudo systemctl stop tv-watch-tracker 2>/dev/null")
        time.sleep(0.5)
    get_camera()
    return jsonify({"success": True})

@app.route("/camera_status")
def camera_status():
    return jsonify({"service_running": is_service_running(), "camera_available": camera is not None and camera.isOpened()})

def gen_frames():
    global camera
    while True:
        if camera is None or not camera.isOpened():
            placeholder = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
            yield (b'--frame\r\nContent-Type: image/png\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(1)
            continue
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/stream")
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/capture", methods=["POST"])
def capture():
    if is_service_running():
        return jsonify({"success": False, "error": "顔認識サービス稼働中"})
    cam = get_camera()
    ret, frame = cam.read()
    if not ret:
        return jsonify({"success": False, "error": "カメラエラー"})
    filename = f"capture_{int(time.time())}.jpg"
    cv2.imwrite(os.path.join(CAPTURES_DIR, filename), frame)
    return jsonify({"success": True, "filename": filename})

@app.route("/capture_service_frame", methods=["POST"])
def capture_service_frame():
    """サービスの最新検出画像を撮影画像として保存（オーバーレイなし）"""
    # クリーンなフレーム（オーバーレイなし）を優先
    clean_path = os.path.join(DETECTIONS_DIR, "latest_frame_clean.jpg")
    latest_path = os.path.join(DETECTIONS_DIR, "latest_frame.jpg")

    src_path = clean_path if os.path.exists(clean_path) else latest_path
    if not os.path.exists(src_path):
        return jsonify({"success": False, "error": "最新画像がありません"})

    # 最新画像を撮影フォルダにコピー
    filename = f"capture_{int(time.time())}.jpg"
    dst_path = os.path.join(CAPTURES_DIR, filename)
    import shutil
    shutil.copy2(src_path, dst_path)
    return jsonify({"success": True, "filename": filename})

@app.route("/captures")
def captures():
    files = sorted(glob.glob(os.path.join(CAPTURES_DIR, "*.jpg")), reverse=True)
    return jsonify([os.path.basename(f) for f in files])

@app.route("/capture_image/<filename>")
def capture_image(filename):
    path = os.path.join(CAPTURES_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return "Not found", 404

@app.route("/delete_capture", methods=["POST"])
def delete_capture():
    filename = request.json.get("filename")
    path = os.path.join(CAPTURES_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
    return jsonify({"success": True})

@app.route("/thumbnail_roi/<filename>")
def thumbnail_roi(filename):
    path = os.path.join(CAPTURES_DIR, filename)
    if not os.path.exists(path):
        return "Not found", 404
    roi_index = request.args.get("roi_index", "")
    roi = get_roi_by_index(roi_index)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    thumb_size = 200
    scale = thumb_size / max(h, w)
    thumb = cv2.resize(img, (int(w * scale), int(h * scale)))
    if roi:
        x = int(roi["x"] * scale)
        y = int(roi["y"] * scale)
        rw = int(roi["w"] * scale)
        rh = int(roi["h"] * scale)
        overlay = thumb.copy()
        cv2.rectangle(overlay, (0, 0), (thumb.shape[1], thumb.shape[0]), (0, 0, 0), -1)
        mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
        mask[y:y+rh, x:x+rw] = 0
        mask[mask > 0] = 128
        dark = thumb.copy()
        dark[mask > 0] = (dark[mask > 0] * 0.4).astype('uint8')
        thumb = dark
        cv2.rectangle(thumb, (x, y), (x + rw, y + rh), (0, 212, 255), 2)
    _, jpeg = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

# ROI API
@app.route("/api/roi_presets")
def api_roi_presets():
    config = load_config()
    presets = config.get("roi_presets", [])
    return jsonify({"presets": presets})

@app.route("/api/roi_presets/add", methods=["POST"])
def api_roi_preset_add():
    config = load_config()
    roi = request.json.get("roi")
    if not roi:
        return jsonify({"success": False, "error": "ROIが必要です"})
    presets = config.get("roi_presets", [])
    max_num = 0
    for p in presets:
        name = p.get("name", "")
        if name.startswith("ROI "):
            try:
                num = int(name[4:])
                max_num = max(max_num, num)
            except:
                pass
    roi["name"] = f"ROI {max_num + 1}"
    presets.append(roi)
    config["roi_presets"] = presets
    save_config(config)
    return jsonify({"success": True, "name": roi["name"]})

@app.route("/api/roi_presets/delete", methods=["POST"])
def api_roi_preset_delete():
    config = load_config()
    index = request.json.get("index", -1)
    presets = config.get("roi_presets", [])
    if 0 <= index < len(presets):
        presets.pop(index)
        config["roi_presets"] = presets
        save_config(config)
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "無効なインデックス"})

# 顔抽出
@app.route("/extract_and_save_faces", methods=["POST"])
def extract_and_save_faces():
    data = request.json
    image = data.get("image")
    model = data.get("model", "hog")
    upsample = data.get("upsample", 2)
    roi_index = data.get("roi_index", "")

    path = os.path.join(CAPTURES_DIR, image)
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "画像が見つかりません"})

    img = cv2.imread(path)
    roi = get_roi_by_index(roi_index)

    if roi:
        x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        img_roi = img[y:y+rh, x:x+rw]
        roi_offset = (x, y)
    else:
        img_roi = img
        roi_offset = (0, 0)

    rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, model=model, number_of_times_to_upsample=upsample)

    count = 0
    import uuid
    for (top, right, bottom, left) in face_locations:
        top += roi_offset[1]
        right += roi_offset[0]
        bottom += roi_offset[1]
        left += roi_offset[0]

        margin = int((bottom - top) * 0.3)
        top = max(0, top - margin)
        left = max(0, left - margin)
        bottom = min(img.shape[0], bottom + margin)
        right = min(img.shape[1], right + margin)

        face_img = img[top:bottom, left:right]
        filename = f"face_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
        cv2.imwrite(os.path.join(FACES_DIR, filename), face_img)
        # メタデータ（未登録状態）
        with open(os.path.join(FACES_DIR, filename + ".json"), "w") as f:
            json.dump({"source": image, "label": ""}, f)
        count += 1

    return jsonify({"success": True, "count": count})

@app.route("/all_faces_status")
def all_faces_status():
    files = sorted(glob.glob(os.path.join(FACES_DIR, "*.jpg")), reverse=True)
    result = []
    for f in files:
        filename = os.path.basename(f)
        meta_path = f + ".json"
        label = ""
        if os.path.exists(meta_path):
            with open(meta_path) as mf:
                label = json.load(mf).get("label", "")
        result.append({"filename": filename, "label": label})
    return jsonify(result)

@app.route("/unregistered_faces")
def unregistered_faces():
    files = sorted(glob.glob(os.path.join(FACES_DIR, "*.jpg")), reverse=True)
    result = []
    for f in files:
        meta_path = f + ".json"
        if os.path.exists(meta_path):
            with open(meta_path) as mf:
                if not json.load(mf).get("label"):
                    result.append(os.path.basename(f))
        else:
            result.append(os.path.basename(f))
    return jsonify(result)

@app.route("/registered_faces_by_label")
def registered_faces_by_label():
    files = glob.glob(os.path.join(FACES_DIR, "*.jpg"))
    result = {}
    for f in files:
        meta_path = f + ".json"
        if os.path.exists(meta_path):
            with open(meta_path) as mf:
                label = json.load(mf).get("label", "")
                if label:
                    if label not in result:
                        result[label] = []
                    result[label].append(os.path.basename(f))

    # エンコーディング状態を確認
    encoded_labels = set()
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, 'rb') as f:
                enc_data = pickle.load(f)
                encoded_labels = set(enc_data.get('names', []))
        except:
            pass

    # 各ラベルのエンコーディング状態を追加
    result_with_status = {}
    for label, face_files in result.items():
        result_with_status[label] = {
            "files": face_files,
            "encoded": label in encoded_labels
        }

    return jsonify(result_with_status)

@app.route("/register_faces", methods=["POST"])
def register_faces():
    data = request.json
    files = data.get("files", [])
    label = data.get("label", "").strip().lower()

    if not label:
        return jsonify({"success": False, "error": "ラベルが必要です"})
    if not files:
        return jsonify({"success": False, "error": "ファイルを選択してください"})

    count = 0
    for filename in files:
        meta_path = os.path.join(FACES_DIR, filename + ".json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["label"] = label
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            count += 1

    # 自動エンコード
    build_encoding_for_label_internal(label)

    return jsonify({"success": True, "count": count})

def build_encoding_for_label_internal(target_label):
    existing_data = {"names": [], "encodings": [], "files": {}}
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                existing_data = pickle.load(f)
                if "files" not in existing_data:
                    existing_data["files"] = {}
        except:
            pass

    new_names = []
    new_encodings = []
    new_files = {}

    for i, name in enumerate(existing_data.get("names", [])):
        if name != target_label:
            new_names.append(name)
            new_encodings.append(existing_data["encodings"][i])

    for label, filelist in existing_data.get("files", {}).items():
        if label != target_label:
            new_files[label] = filelist

    files = glob.glob(os.path.join(FACES_DIR, "*.jpg"))
    encoded_files_list = []

    for f in files:
        meta_path = f + ".json"
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as mf:
            label = json.load(mf).get("label", "")
        if label != target_label:
            continue

        filename = os.path.basename(f)
        img = cv2.imread(f)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=1)

        if len(face_locations) == 0:
            h, w = rgb.shape[:2]
            face_locations = [(0, w, h, 0)]
        elif len(face_locations) > 1:
            face_locations = [max(face_locations, key=lambda x: (x[2]-x[0]) * (x[1]-x[3]))]

        try:
            enc = face_recognition.face_encodings(rgb, face_locations)[0]
            new_names.append(target_label)
            new_encodings.append(enc)
            encoded_files_list.append(filename)
        except:
            continue

    if encoded_files_list:
        new_files[target_label] = encoded_files_list

    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"names": new_names, "encodings": new_encodings, "files": new_files}, f)

@app.route("/face_image/<filename>")
def face_image(filename):
    path = os.path.join(FACES_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return "Not found", 404

@app.route("/delete_face", methods=["POST"])
def delete_face():
    filename = request.json.get("filename")
    path = os.path.join(FACES_DIR, filename)
    meta_path = path + ".json"
    if os.path.exists(path):
        os.remove(path)
    if os.path.exists(meta_path):
        os.remove(meta_path)
    return jsonify({"success": True})

# テスト機能
last_recog_result = None
last_recog_faces = []
last_recog_original = None
last_recog_locations = []
last_recog_roi = None
last_recog_names = []
last_detect_result = None
last_detect_faces = []
last_detect_original = None
last_detect_locations = []
last_detect_roi = None

@app.route("/detect_only", methods=["POST"])
def detect_only():
    """顔検出のみ（認識なし）"""
    global last_detect_result, last_detect_faces, last_detect_original, last_detect_locations, last_detect_roi
    data = request.json
    image = data.get("image")
    model = data.get("model", "hog")
    upsample = data.get("upsample", 2)
    roi_index = data.get("roi_index", "")

    path = os.path.join(CAPTURES_DIR, image)
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "画像が見つかりません"})

    img = cv2.imread(path)
    last_detect_original = img.copy()
    roi = get_roi_by_index(roi_index)
    last_detect_roi = roi

    if roi:
        x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        img_roi = img[y:y+rh, x:x+rw]
        roi_offset = (x, y)
    else:
        img_roi = img
        roi_offset = (0, 0)

    rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)

    start = time.time()
    face_locations = face_recognition.face_locations(rgb, model=model, number_of_times_to_upsample=upsample)
    elapsed = round(time.time() - start, 2)

    # 顔の切り抜きと座標を保存
    last_detect_faces = []
    last_detect_locations = []
    faces_info = []
    for (top, right, bottom, left) in face_locations:
        orig_top = top + roi_offset[1]
        orig_right = right + roi_offset[0]
        orig_bottom = bottom + roi_offset[1]
        orig_left = left + roi_offset[0]

        # 座標を保存
        last_detect_locations.append((orig_top, orig_right, orig_bottom, orig_left))

        # 切り抜き画像を保存
        face_img = last_detect_original[orig_top:orig_bottom, orig_left:orig_right]
        last_detect_faces.append(face_img)
        faces_info.append({
            "width": orig_right - orig_left,
            "height": orig_bottom - orig_top
        })

    # デフォルト表示用（BBox・ROI両方表示）
    for (orig_top, orig_right, orig_bottom, orig_left) in last_detect_locations:
        cv2.rectangle(img, (orig_left, orig_top), (orig_right, orig_bottom), (0, 255, 0), 2)
    if roi:
        cv2.rectangle(img, (roi["x"], roi["y"]), (roi["x"]+roi["w"], roi["y"]+roi["h"]), (0, 212, 255), 2)

    last_detect_result = img

    return jsonify({
        "success": True,
        "count": len(face_locations),
        "time": elapsed,
        "roi_used": roi is not None,
        "faces": faces_info
    })

@app.route("/detect_result")
def detect_result():
    if last_detect_result is None:
        return "No result", 404
    _, jpeg = cv2.imencode('.jpg', last_detect_result)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route("/detect_result_render")
def detect_result_render():
    """BBox/ROI表示を切替えて検出結果画像を返す"""
    global last_detect_original, last_detect_locations, last_detect_roi
    if last_detect_original is None:
        return "No result", 404

    show_bbox = request.args.get('show_bbox', 'true').lower() == 'true'
    show_roi = request.args.get('show_roi', 'true').lower() == 'true'

    img = last_detect_original.copy()

    if show_bbox:
        for (top, right, bottom, left) in last_detect_locations:
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    if show_roi and last_detect_roi:
        roi = last_detect_roi
        cv2.rectangle(img, (roi["x"], roi["y"]), (roi["x"]+roi["w"], roi["y"]+roi["h"]), (0, 212, 255), 2)

    _, jpeg = cv2.imencode('.jpg', img)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route("/detect_face/<int:index>")
def detect_face(index):
    """検出した顔の切り抜き画像を返す"""
    global last_detect_faces
    if last_detect_faces is None or index >= len(last_detect_faces):
        return "Not found", 404
    face_img = last_detect_faces[index]
    if face_img is None or face_img.size == 0:
        return "Invalid face", 404
    _, jpeg = cv2.imencode('.jpg', face_img)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    """単一顔画像の認識"""
    data = request.json
    face_file = data.get("face_file")
    tolerance = data.get("tolerance", 0.5)

    path = os.path.join(FACES_DIR, face_file)
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "顔画像が見つかりません"})

    if not os.path.exists(ENCODINGS_PATH):
        return jsonify({"success": False, "error": "エンコーディングファイルがありません"})

    try:
        with open(ENCODINGS_PATH, "rb") as f:
            enc_data = pickle.load(f)
        known_names = enc_data.get("names", [])
        known_encodings = enc_data.get("encodings", [])
        if not known_names:
            return jsonify({"success": False, "error": "登録された顔がありません"})
    except:
        return jsonify({"success": False, "error": "エンコーディングの読み込みに失敗しました"})

    img = face_recognition.load_image_file(path)

    # まず通常の顔検出を試みる
    encodings = face_recognition.face_encodings(img)

    # 検出できない場合、画像全体を顔として扱う（既に切り抜き済みの顔画像のため）
    if len(encodings) == 0:
        h, w = img.shape[:2]
        face_location = [(0, w, h, 0)]  # top, right, bottom, left
        encodings = face_recognition.face_encodings(img, face_location)

    if len(encodings) == 0:
        return jsonify({"success": False, "error": "顔のエンコードに失敗しました"})

    enc = encodings[0]
    distances = face_recognition.face_distance(known_encodings, enc)

    if len(distances) == 0:
        return jsonify({"success": True, "name": "unknown", "distance": 1.0, "all_distances": {}})

    # 各ラベルごとの最小距離を計算
    label_distances = {}
    for i, (dist, label_name) in enumerate(zip(distances, known_names)):
        if label_name not in label_distances or dist < label_distances[label_name]:
            label_distances[label_name] = float(dist)

    min_idx = distances.argmin()
    min_distance = distances[min_idx]
    name = known_names[min_idx] if min_distance <= tolerance else "unknown"

    return jsonify({
        "success": True,
        "name": name,
        "distance": float(min_distance),
        "all_distances": label_distances
    })

@app.route("/recognize", methods=["POST"])
def recognize():
    global last_recog_result, last_recog_faces, last_recog_original, last_recog_locations, last_recog_roi, last_recog_names
    data = request.json
    image = data.get("image")
    model = data.get("model", "hog")
    upsample = data.get("upsample", 2)
    tolerance = data.get("tolerance", 0.5)
    roi_index = data.get("roi_index", "")

    path = os.path.join(CAPTURES_DIR, image)
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "画像が見つかりません"})

    if not os.path.exists(ENCODINGS_PATH):
        return jsonify({"success": False, "error": "エンコーディングファイルがありません"})

    try:
        with open(ENCODINGS_PATH, "rb") as f:
            enc_data = pickle.load(f)
        known_names = enc_data.get("names", [])
        known_encodings = enc_data.get("encodings", [])
        if not known_names:
            return jsonify({"success": False, "error": "登録された顔がありません"})
    except:
        return jsonify({"success": False, "error": "エンコーディングの読み込みに失敗しました"})

    img = cv2.imread(path)
    last_recog_original = img.copy()
    roi = get_roi_by_index(roi_index)
    last_recog_roi = roi
    roi_used = roi is not None

    if roi:
        x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        img_roi = img[y:y+rh, x:x+rw]
        roi_offset = (x, y)
    else:
        img_roi = img
        roi_offset = (0, 0)

    rgb = cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB)

    start = time.time()
    face_locations = face_recognition.face_locations(rgb, model=model, number_of_times_to_upsample=upsample)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)
    elapsed = round(time.time() - start, 2)

    faces = []
    last_recog_faces = []
    last_recog_locations = []
    last_recog_names = []

    for i, (enc, (top, right, bottom, left)) in enumerate(zip(face_encodings, face_locations)):
        orig_top = top + roi_offset[1]
        orig_right = right + roi_offset[0]
        orig_bottom = bottom + roi_offset[1]
        orig_left = left + roi_offset[0]

        distances = face_recognition.face_distance(known_encodings, enc)
        if len(distances) == 0:
            name = "unknown"
            min_distance = 1.0
        else:
            min_idx = distances.argmin()
            min_distance = distances[min_idx]
            name = known_names[min_idx] if min_distance <= tolerance else "unknown"

        faces.append({"name": name, "distance": float(min_distance)})
        last_recog_locations.append((orig_top, orig_right, orig_bottom, orig_left))
        last_recog_names.append((name, min_distance))

        # 顔画像を保存
        margin = int((orig_bottom - orig_top) * 0.2)
        crop_top = max(0, orig_top - margin)
        crop_left = max(0, orig_left - margin)
        crop_bottom = min(img.shape[0], orig_bottom + margin)
        crop_right = min(img.shape[1], orig_right + margin)
        face_crop = last_recog_original[crop_top:crop_bottom, crop_left:crop_right]
        last_recog_faces.append(face_crop)

        # 描画
        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(img, (orig_left, orig_top), (orig_right, orig_bottom), color, 2)
        similarity = max(0, (1 - min_distance) * 100)
        cv2.putText(img, f"{name} ({similarity:.0f}%)", (orig_left, orig_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if roi:
        cv2.rectangle(img, (roi["x"], roi["y"]), (roi["x"]+roi["w"], roi["y"]+roi["h"]), (0, 212, 255), 2)

    last_recog_result = img

    return jsonify({"success": True, "faces": faces, "time": elapsed, "image": image, "roi_used": roi_used})

@app.route("/recog_result")
def recog_result():
    if last_recog_result is None:
        return "No result", 404
    _, jpeg = cv2.imencode('.jpg', last_recog_result)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route("/recog_result_render")
def recog_result_render():
    """BBox/ROI表示を切替えて認識結果画像を返す"""
    global last_recog_original, last_recog_locations, last_recog_roi, last_recog_names
    if last_recog_original is None:
        return "No result", 404

    show_bbox = request.args.get('show_bbox', 'true').lower() == 'true'
    show_roi = request.args.get('show_roi', 'true').lower() == 'true'

    img = last_recog_original.copy()

    if show_bbox:
        for (top, right, bottom, left), (name, distance) in zip(last_recog_locations, last_recog_names):
            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(img, (left, top), (right, bottom), color, 2)
            similarity = max(0, (1 - distance) * 100)
            cv2.putText(img, f"{name} ({similarity:.0f}%)", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if show_roi and last_recog_roi:
        roi = last_recog_roi
        cv2.rectangle(img, (roi["x"], roi["y"]), (roi["x"]+roi["w"], roi["y"]+roi["h"]), (0, 212, 255), 2)

    _, jpeg = cv2.imencode('.jpg', img)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route("/recog_face/<int:idx>")
def recog_face(idx):
    if idx >= len(last_recog_faces):
        return "Not found", 404
    _, jpeg = cv2.imencode('.jpg', last_recog_faces[idx])
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

# ダッシュボードAPI
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import subprocess

LOG_PATH = os.path.expanduser("~/tv_watch_log.csv")
DETECTIONS_DIR = os.path.expanduser("~/detections")
os.makedirs(DETECTIONS_DIR, exist_ok=True)

def get_registered_labels():
    """画像が1枚以上登録されているラベルを取得"""
    labels = set()
    if os.path.exists(FACES_DIR):
        for f in os.listdir(FACES_DIR):
            if f.endswith('.jpg'):
                # JSONファイルは .jpg.json の形式
                json_path = os.path.join(FACES_DIR, f + '.json')
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            data = json.load(jf)
                            if data.get('label'):
                                labels.add(data['label'])
                    except:
                        pass
    return list(labels)

def get_first_registered_date():
    """最初の顔登録日を取得"""
    earliest = None
    if os.path.exists(FACES_DIR):
        for f in os.listdir(FACES_DIR):
            if f.endswith('.jpg'):
                # JSONファイルは .jpg.json の形式
                json_path = os.path.join(FACES_DIR, f + '.json')
                if os.path.exists(json_path):
                    try:
                        mtime = os.path.getmtime(json_path)
                        if earliest is None or mtime < earliest:
                            earliest = mtime
                    except:
                        pass
    return datetime.fromtimestamp(earliest) if earliest else None

last_detection_image = None
last_detection_meta = None

@app.route("/api/dashboard")
def api_dashboard():
    global last_detection_image, last_detection_meta
    config = load_config()
    log_path = os.path.expanduser(config.get("log_path", "~/tv_watch_log.csv"))
    # 視聴中断とみなす閾値（秒）- この時間以上空いたら別セッション
    gap_threshold_sec = 120  # 2分

    registered_labels = get_registered_labels()
    first_registered = get_first_registered_date()

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    cutoff = now - timedelta(days=7)
    three_hours_ago = now - timedelta(hours=3)

    daily_minutes = defaultdict(lambda: defaultdict(float))
    recent_grouped = []
    detection_3h = {name: [False] * 180 for name in registered_labels}  # 3時間 = 180分

    current_group = None
    # 各人物・各日の最後の検出時刻を追跡
    last_detection_by_name_date = defaultdict(dict)  # {name: {date: last_timestamp}}

    if os.path.exists(log_path):
        try:
            with open(log_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                        name = row["name"]

                        # 登録前のデータは無視
                        if first_registered and ts < first_registered:
                            continue
                        if ts < cutoff:
                            continue

                        # 登録済みラベルのみ
                        if name not in registered_labels:
                            continue

                        date_str = ts.strftime("%Y-%m-%d")

                        # タイムスタンプ間の時間を計算
                        if date_str in last_detection_by_name_date[name]:
                            last_ts = last_detection_by_name_date[name][date_str]
                            diff_sec = (ts - last_ts).total_seconds()
                            # 閾値以内なら視聴時間としてカウント
                            if 0 < diff_sec <= gap_threshold_sec:
                                daily_minutes[date_str][name] += diff_sec / 60.0
                        last_detection_by_name_date[name][date_str] = ts

                        # 直近3時間のバーコード
                        if ts >= three_hours_ago:
                            minute_idx = int((ts - three_hours_ago).total_seconds() / 60)
                            if 0 <= minute_idx < 180:
                                detection_3h[name][minute_idx] = True

                        # 検出ログのグループ化（同じ秒は1レコード）
                        ts_key = row["timestamp"]
                        # 検出画像ファイル名を生成
                        img_ts = ts.strftime("%Y%m%d_%H%M%S")
                        if current_group and current_group["timestamp"] == ts_key:
                            if name not in current_group["names"]:
                                current_group["names"].append(name)
                                current_group["images"].append(f"detection_{img_ts}_{name}.jpg")
                        else:
                            if current_group:
                                recent_grouped.append(current_group)
                            current_group = {"timestamp": ts_key, "names": [name], "images": [f"detection_{img_ts}_{name}.jpg"]}
                    except:
                        continue
                if current_group:
                    recent_grouped.append(current_group)
        except:
            pass

    recent_grouped = recent_grouped[-50:][::-1]

    # 直近の画像（detectionsフォルダ優先、なければcaptures）
    latest_image = None
    if os.path.exists(DETECTIONS_DIR):
        # latest_frame.jpg を優先
        latest_frame_path = os.path.join(DETECTIONS_DIR, "latest_frame.jpg")
        if os.path.exists(latest_frame_path):
            latest_image = "latest_frame.jpg"
            last_detection_image = latest_frame_path
            # latest_frame専用のメタファイルを使用
            latest_meta_path = os.path.join(DETECTIONS_DIR, "latest_frame_meta.json")
            if os.path.exists(latest_meta_path):
                try:
                    with open(latest_meta_path) as f:
                        last_detection_meta = json.load(f)
                except:
                    last_detection_meta = None
            else:
                last_detection_meta = None
        else:
            # latest_frame.jpgがない場合は従来通り
            all_images = sorted(glob.glob(os.path.join(DETECTIONS_DIR, "*.jpg")), reverse=True)
            if all_images:
                latest_image = os.path.basename(all_images[0])
                last_detection_image = all_images[0]
                last_detection_meta = None

    # detectionsがなければcapturesから
    if not latest_image and os.path.exists(CAPTURES_DIR):
        all_images = sorted(glob.glob(os.path.join(CAPTURES_DIR, "*.jpg")), reverse=True)
        if all_images:
            latest_image = os.path.basename(all_images[0])
            last_detection_image = all_images[0]
            last_detection_meta = None

    # ROI名称を取得
    roi_name = ""
    roi_index = config.get('roi_index')
    if roi_index is not None and roi_index != '':
        try:
            idx = int(roi_index)
            presets = config.get("roi_presets", [])
            if 0 <= idx < len(presets):
                roi_name = presets[idx].get('name', f'ROI {idx+1}')
        except:
            pass

    return jsonify({
        "daily": {k: dict(v) for k, v in daily_minutes.items()},
        "registered_labels": registered_labels,
        "latest_image": latest_image,
        "detection_3h": detection_3h,
        "recent_grouped": recent_grouped,
        "roi_name": roi_name
    })

@app.route("/api/latest_image")
def api_latest_image():
    """直近画像をROI/BBox表示切替で返す"""
    show_roi = request.args.get('roi', 'true').lower() == 'true'
    show_bbox = request.args.get('bbox', 'true').lower() == 'true'

    # クリーンな画像（オーバーレイなし）を優先使用
    clean_path = os.path.join(DETECTIONS_DIR, "latest_frame_clean.jpg")
    if os.path.exists(clean_path):
        img = cv2.imread(clean_path)
    elif last_detection_image and os.path.exists(last_detection_image):
        img = cv2.imread(last_detection_image)
    else:
        return "Not found", 404

    if img is None:
        return "Failed to load", 500

    config = load_config()

    # ROI描画
    if show_roi:
        roi_index = config.get('roi_index')
        roi = get_roi_by_index(roi_index)
        if roi:
            cv2.rectangle(img, (roi['x'], roi['y']), (roi['x']+roi['w'], roi['y']+roi['h']), (0, 212, 255), 2)

    # BBox描画（メタデータがある場合）
    if show_bbox and last_detection_meta:
        faces = last_detection_meta.get('faces', [])
        for face in faces:
            bbox = face.get('bbox', {})
            if bbox:
                # top, right, bottom, left 形式
                top = bbox.get('top', 0)
                right = bbox.get('right', 0)
                bottom = bbox.get('bottom', 0)
                left = bbox.get('left', 0)
                name = face.get('name', 'Unknown')
                similarity = face.get('similarity', 0)
                color = (0, 255, 0) if name != 'unknown' else (0, 0, 255)
                # 顔枠を描画
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                # ラベル表示
                label = f"{name} ({similarity:.0f}%)" if similarity else name
                cv2.putText(img, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, jpeg = cv2.imencode('.jpg', img)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route("/api/service_frame")
def api_service_frame():
    """サービスが撮像中の最新フレームを返す"""
    latest_path = os.path.join(DETECTIONS_DIR, "latest_frame.jpg")
    if os.path.exists(latest_path):
        # ファイルの更新時間をチェック（60秒以内なら有効）
        mtime = os.path.getmtime(latest_path)
        if time.time() - mtime < 60:
            img = cv2.imread(latest_path)
            if img is not None:
                _, jpeg = cv2.imencode('.jpg', img)
                return Response(jpeg.tobytes(), mimetype='image/jpeg')
    return "No recent frame", 404

@app.route("/api/distribution")
def api_distribution():
    """指定日の時間帯別視聴時間"""
    date = request.args.get('date')
    if not date:
        return jsonify({"error": "date required"})

    config = load_config()
    log_path = os.path.expanduser(config.get("log_path", "~/tv_watch_log.csv"))
    gap_threshold_sec = 120  # 2分
    registered_labels = get_registered_labels()

    hourly = defaultdict(lambda: defaultdict(float))
    # 各人物の最後の検出時刻を追跡
    last_detection_by_name = {}

    if os.path.exists(log_path):
        try:
            with open(log_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                        name = row["name"]
                        if name not in registered_labels:
                            continue
                        date_str = ts.strftime("%Y-%m-%d")
                        if date_str == date:
                            hour_str = ts.strftime("%H")
                            # タイムスタンプ間の時間を計算
                            if name in last_detection_by_name:
                                last_ts = last_detection_by_name[name]
                                diff_sec = (ts - last_ts).total_seconds()
                                if 0 < diff_sec <= gap_threshold_sec:
                                    # 時間をまたぐ場合は現在の時間帯に計上
                                    hourly[hour_str][name] += diff_sec / 60.0
                            last_detection_by_name[name] = ts
                    except:
                        continue
        except:
            pass

    return jsonify({
        "date": date,
        "hourly": {k: dict(v) for k, v in hourly.items()},
        "labels": registered_labels
    })

@app.route("/api/trend")
def api_trend():
    """期間指定の日別視聴時間推移"""
    start = request.args.get('start')
    end = request.args.get('end')
    if not start or not end:
        return jsonify({"error": "start and end required"})

    config = load_config()
    log_path = os.path.expanduser(config.get("log_path", "~/tv_watch_log.csv"))
    gap_threshold_sec = 120  # 2分
    registered_labels = get_registered_labels()

    try:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")
    except:
        return jsonify({"error": "invalid date format"})

    daily = defaultdict(lambda: defaultdict(float))
    # 各人物・各日の最後の検出時刻を追跡
    last_detection_by_name_date = defaultdict(dict)

    if os.path.exists(log_path):
        try:
            with open(log_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                        name = row["name"]
                        if name not in registered_labels:
                            continue
                        date_str = ts.strftime("%Y-%m-%d")
                        row_date = datetime.strptime(date_str, "%Y-%m-%d")
                        if start_date <= row_date <= end_date:
                            # タイムスタンプ間の時間を計算
                            if date_str in last_detection_by_name_date[name]:
                                last_ts = last_detection_by_name_date[name][date_str]
                                diff_sec = (ts - last_ts).total_seconds()
                                if 0 < diff_sec <= gap_threshold_sec:
                                    daily[date_str][name] += diff_sec / 60.0
                            last_detection_by_name_date[name][date_str] = ts
                    except:
                        continue
        except:
            pass

    # 日付リストを生成
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return jsonify({
        "start": start,
        "end": end,
        "dates": dates,
        "daily": {k: dict(v) for k, v in daily.items()},
        "labels": registered_labels
    })

@app.route("/api/label_status")
def api_label_status():
    """ラベル管理用：各ラベルの画像数"""
    labels = {}

    # エンコードファイルからラベル一覧を取得
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, 'rb') as f:
                enc_data = pickle.load(f)
                for name in enc_data.get('names', []):
                    if name not in labels:
                        labels[name] = 0
        except:
            pass

    # 画像ファイルからラベルごとの画像数をカウント
    if os.path.exists(FACES_DIR):
        for f in os.listdir(FACES_DIR):
            if f.endswith('.jpg'):
                json_path = os.path.join(FACES_DIR, f + '.json')
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            data = json.load(jf)
                            label = data.get('label')
                            if label:
                                labels[label] = labels.get(label, 0) + 1
                    except:
                        pass

    result = [{"name": name, "count": count} for name, count in sorted(labels.items())]
    return jsonify({"labels": result})

@app.route("/api/delete_label", methods=["POST"])
def api_delete_label():
    """ラベルを削除（画像のラベルも解除）"""
    name = request.json.get('name')
    if not name:
        return jsonify({"success": False, "error": "name required"})

    # 顔画像のJSONからラベルを解除
    cleared_count = 0
    if os.path.exists(FACES_DIR):
        for f in os.listdir(FACES_DIR):
            if f.endswith('.jpg'):
                json_path = os.path.join(FACES_DIR, f + '.json')
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            data = json.load(jf)
                        if data.get('label') == name:
                            data['label'] = ''
                            with open(json_path, 'w') as jf:
                                json.dump(data, jf)
                            cleared_count += 1
                    except:
                        pass

    # エンコードファイルからラベルを削除
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, 'rb') as f:
                enc_data = pickle.load(f)

            new_encodings = []
            new_names = []
            for enc, n in zip(enc_data.get('encodings', []), enc_data.get('names', [])):
                if n != name:
                    new_encodings.append(enc)
                    new_names.append(n)

            with open(ENCODINGS_PATH, 'wb') as f:
                pickle.dump({'encodings': new_encodings, 'names': new_names}, f)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)})

    return jsonify({"success": True, "cleared": cleared_count})

@app.route("/api/rename_label", methods=["POST"])
def api_rename_label():
    """ラベル名を変更"""
    old_name = request.json.get('old_name', '').strip().lower()
    new_name = request.json.get('new_name', '').strip().lower()

    if not old_name or not new_name:
        return jsonify({"success": False, "error": "ラベル名が必要です"})

    if old_name == new_name:
        return jsonify({"success": True})

    # 新しいラベル名が既に存在するかチェック
    existing_labels = set()
    if os.path.exists(FACES_DIR):
        for f in os.listdir(FACES_DIR):
            if f.endswith('.jpg'):
                json_path = os.path.join(FACES_DIR, f + '.json')
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            data = json.load(jf)
                            if data.get('label'):
                                existing_labels.add(data.get('label'))
                    except:
                        pass

    if new_name in existing_labels and new_name != old_name:
        return jsonify({"success": False, "error": f"ラベル '{new_name}' は既に存在します"})

    # 顔画像のJSONファイルを更新
    updated_count = 0
    if os.path.exists(FACES_DIR):
        for f in os.listdir(FACES_DIR):
            if f.endswith('.jpg'):
                json_path = os.path.join(FACES_DIR, f + '.json')
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            data = json.load(jf)
                        if data.get('label') == old_name:
                            data['label'] = new_name
                            with open(json_path, 'w') as jf:
                                json.dump(data, jf)
                            updated_count += 1
                    except:
                        pass

    # エンコードファイルのラベルを更新
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, 'rb') as f:
                enc_data = pickle.load(f)

            new_names = [new_name if n == old_name else n for n in enc_data.get('names', [])]
            enc_data['names'] = new_names

            with open(ENCODINGS_PATH, 'wb') as f:
                pickle.dump(enc_data, f)
        except Exception as e:
            return jsonify({"success": False, "error": f"エンコード更新エラー: {str(e)}"})

    return jsonify({"success": True, "updated": updated_count})

@app.route("/detection_image/<filename>")
def detection_image(filename):
    path = os.path.join(DETECTIONS_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return "Not found", 404

@app.route("/detection_render/<timestamp>")
def detection_render(timestamp):
    """検出画像を動的にレンダリング（BBox/ROI/スコア表示切替対応）"""
    show_bbox = request.args.get('bbox', 'true').lower() == 'true'
    show_roi = request.args.get('roi', 'true').lower() == 'true'
    show_score = request.args.get('score', 'true').lower() == 'true'

    # 元画像とメタデータを読み込む
    orig_path = os.path.join(DETECTIONS_DIR, f"detection_{timestamp}_original.jpg")
    meta_path = os.path.join(DETECTIONS_DIR, f"detection_{timestamp}_meta.json")

    if not os.path.exists(orig_path):
        # 元画像がない場合は旧形式の画像を探す
        import glob
        old_files = glob.glob(os.path.join(DETECTIONS_DIR, f"detection_{timestamp}_*.jpg"))
        old_files = [f for f in old_files if not f.endswith("_original.jpg")]
        if old_files:
            return send_file(old_files[0], mimetype='image/jpeg')
        return "Not found", 404

    img = cv2.imread(orig_path)
    if img is None:
        return "Failed to load image", 500

    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except:
            pass

    # ROI描画
    if show_roi and meta.get("roi"):
        roi = meta["roi"]
        cv2.rectangle(img, (roi["x"], roi["y"]),
                      (roi["x"] + roi["w"], roi["y"] + roi["h"]), (0, 212, 255), 2)

    # BBox描画
    if show_bbox and meta.get("faces"):
        for face in meta["faces"]:
            bbox = face["bbox"]
            name = face["name"]
            similarity = face.get("similarity", 0)
            color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
            cv2.rectangle(img, (bbox["left"], bbox["top"]),
                          (bbox["right"], bbox["bottom"]), color, 2)
            if show_score:
                label = f"{name} ({similarity:.0f}%)"
            else:
                label = name
            cv2.putText(img, label, (bbox["left"], bbox["top"] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    _, jpeg = cv2.imencode('.jpg', img)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route("/api/detection_meta/<timestamp>")
def api_detection_meta(timestamp):
    """検出メタデータを返す"""
    meta_path = os.path.join(DETECTIONS_DIR, f"detection_{timestamp}_meta.json")
    if not os.path.exists(meta_path):
        return jsonify({"error": "Meta not found"}), 404
    try:
        with open(meta_path, 'r') as f:
            return jsonify(json.load(f))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/relabel_detection", methods=["POST"])
def api_relabel_detection():
    """検出のラベルを変更し、顔画像を登録する"""
    data = request.json
    timestamp = data.get('timestamp')
    updates = data.get('updates', [])

    if not timestamp or not updates:
        return jsonify({"success": False, "error": "Invalid request"})

    # メタデータを更新
    meta_path = os.path.join(DETECTIONS_DIR, f"detection_{timestamp}_meta.json")
    orig_path = os.path.join(DETECTIONS_DIR, f"detection_{timestamp}_original.jpg")
    if not os.path.exists(meta_path):
        return jsonify({"success": False, "error": "Meta not found"})

    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # 元画像を読み込み（顔抽出用）
        orig_img = None
        if os.path.exists(orig_path):
            orig_img = cv2.imread(orig_path)

        # 顔ラベルを更新＆顔画像を抽出
        saved_faces = []
        for update in updates:
            idx = update['index']
            old_name = update['old_name']
            new_name = update['new_name']
            if 0 <= idx < len(meta.get('faces', [])):
                meta['faces'][idx]['name'] = new_name

                # 顔画像を抽出して保存（unknownでない場合のみ）
                if new_name != 'unknown' and orig_img is not None:
                    bbox = meta['faces'][idx].get('bbox', {})
                    if bbox:
                        top = bbox.get('top', 0)
                        right = bbox.get('right', 0)
                        bottom = bbox.get('bottom', 0)
                        left = bbox.get('left', 0)
                        # マージンを追加
                        h, w = orig_img.shape[:2]
                        margin = int((bottom - top) * 0.2)
                        top = max(0, top - margin)
                        bottom = min(h, bottom + margin)
                        left = max(0, left - margin)
                        right = min(w, right + margin)
                        face_img = orig_img[top:bottom, left:right]
                        if face_img.size > 0:
                            # 保存ファイル名を生成
                            face_filename = f"relabel_{timestamp}_{idx}_{new_name}.jpg"
                            face_path = os.path.join(FACES_DIR, face_filename)
                            cv2.imwrite(face_path, face_img)
                            # メタデータも保存
                            face_meta = {"label": new_name, "source": f"detection_{timestamp}"}
                            with open(face_path + '.json', 'w') as f:
                                json.dump(face_meta, f)
                            saved_faces.append(face_filename)

        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        # CSVログも更新
        config = load_config()
        log_path = os.path.expanduser(config.get("log_path", "~/tv_watch_log.csv"))

        # タイムスタンプをCSV形式に変換 (YYYYMMDD_HHMMSS -> YYYY-MM-DD HH:MM:SS)
        ts_csv = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"

        if os.path.exists(log_path):
            # CSVを読み込んで更新
            rows = []
            updated = False
            with open(log_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    if row['timestamp'] == ts_csv:
                        for update in updates:
                            if row['name'] == update['old_name']:
                                row['name'] = update['new_name']
                                updated = True
                    rows.append(row)

            if updated:
                with open(log_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

        # 自動エンコード（保存した顔のラベルごとに実行）
        encoded_labels = set()
        for update in updates:
            new_name = update['new_name']
            if new_name != 'unknown' and new_name not in encoded_labels:
                build_encoding_for_label_internal(new_name)
                encoded_labels.add(new_name)

        return jsonify({"success": True, "saved_faces": saved_faces, "encoded_labels": list(encoded_labels)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/delete_detection", methods=["POST"])
def api_delete_detection():
    """検出記録を削除する"""
    data = request.json
    timestamp = data.get('timestamp')

    if not timestamp:
        return jsonify({"success": False, "error": "Invalid request"})

    try:
        # 関連ファイルを削除
        patterns = [
            f"detection_{timestamp}_original.jpg",
            f"detection_{timestamp}_meta.json",
            f"detection_{timestamp}_*.jpg"
        ]
        for pattern in patterns:
            for f in glob.glob(os.path.join(DETECTIONS_DIR, pattern)):
                os.remove(f)

        # CSVログから削除
        config = load_config()
        log_path = os.path.expanduser(config.get("log_path", "~/tv_watch_log.csv"))

        # タイムスタンプをCSV形式に変換
        ts_csv = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"

        if os.path.exists(log_path):
            rows = []
            with open(log_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    if row['timestamp'] != ts_csv:
                        rows.append(row)

            with open(log_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/service_status")
def api_service_status():
    try:
        result = subprocess.run(["systemctl", "is-active", "tv-watch-tracker"], capture_output=True, text=True)
        running = result.stdout.strip() == "active"
    except:
        running = False
    return jsonify({"running": running})

@app.route("/api/applied_config")
def api_applied_config():
    """サービスが実際に使用している設定を返す"""
    since = request.args.get('since', type=float)  # タイムスタンプ（オプション）

    try:
        result = subprocess.run(["systemctl", "is-active", "tv-watch-tracker"], capture_output=True, text=True)
        running = result.stdout.strip() == "active"
    except:
        running = False

    if not running:
        return jsonify({"running": False, "config": None, "mtime": None})

    # サービス起動時に保存された設定を読む
    applied_config_path = os.path.expanduser("~/tv_watch_applied_config.json")
    if os.path.exists(applied_config_path):
        try:
            mtime = os.path.getmtime(applied_config_path)
            # sinceが指定されている場合、ファイルがそれより新しいかチェック
            if since and mtime < since:
                return jsonify({"running": True, "config": None, "mtime": mtime, "waiting": True})
            with open(applied_config_path, 'r') as f:
                config = json.load(f)
            return jsonify({"running": True, "config": config, "mtime": mtime})
        except:
            pass

    # フォールバック: 保存済み設定を返す
    return jsonify({"running": True, "config": load_config(), "mtime": None})

@app.route("/api/service_control", methods=["POST"])
def api_service_control():
    action = request.json.get("action")
    if action not in ["start", "stop", "restart"]:
        return jsonify({"error": "Invalid action"})
    try:
        if action in ["start", "restart"]:
            release_camera()
        result = subprocess.run(["sudo", "systemctl", action, "tv-watch-tracker"], capture_output=True, text=True)
        if result.returncode != 0:
            return jsonify({"error": result.stderr})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/config")
def api_get_config():
    return jsonify(load_config())

@app.route("/api/config", methods=["POST"])
def api_save_config():
    try:
        config = load_config()
        updates = request.json
        for key in ["face_model", "upsample", "interval_sec", "tolerance", "roi_index"]:
            if key in updates:
                config[key] = updates[key]
        save_config(config)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5002, debug=False, threaded=True)
    finally:
        release_camera()
