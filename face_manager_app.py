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
        .service-header {
            background: #16213e;
            padding: 12px 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            border-bottom: 2px solid #0f3460;
            flex-wrap: wrap;
        }
        .service-header .service-label { color: #888; font-size: 0.9em; }
        .service-header #serviceStatus {
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: bold;
            min-width: 100px;
            text-align: center;
        }
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
            display: flex; align-items: center; gap: 10px;
        }
        .roi-preset-item .delete-roi { color: #ff6b6b; cursor: pointer; font-size: 1.2em; }
        .label-group { background: #0f3460; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
        .label-group h4 { color: #ffe66d; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="service-header">
        <span class="service-label">顔認識サービス:</span>
        <span id="serviceStatus" style="background:#666;">確認中...</span>
        <button class="btn btn-success btn-small" onclick="serviceControl('start')">開始</button>
        <button class="btn btn-danger btn-small" onclick="serviceControl('stop')">停止</button>
    </div>
    <div class="tabs">
        <button class="tab active" onclick="showTab('camera')">カメラ</button>
        <button class="tab" onclick="showTab('roi')">ROI設定</button>
        <button class="tab" onclick="showTab('extract')">顔抽出</button>
        <button class="tab" onclick="showTab('register')">顔登録</button>
        <button class="tab" onclick="showTab('recognize')">顔認識テスト</button>
        <button class="tab" onclick="showTab('dashboard')">ダッシュボード</button>
    </div>
    <div class="content">
        <!-- カメラタブ -->
        <div id="camera" class="tab-content active">
            <div class="card">
                <h2>リアルタイムプレビュー</h2>
                <div id="cameraOverlay" style="display:none;background:#0f3460;padding:30px;border-radius:8px;text-align:center;margin-bottom:15px;">
                    <p style="color:#ffe66d;font-size:1.2em;margin-bottom:15px;">📹 顔認識サービス稼働中</p>
                    <p style="color:#888;margin-bottom:20px;">カメラを使用するには顔認識サービスを停止する必要があります</p>
                    <button class="btn btn-primary" onclick="startCamera()">カメラ開始（サービス停止）</button>
                </div>
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
        </div>

        <!-- 顔認識テストタブ -->
        <div id="recognize" class="tab-content">
            <div class="card">
                <h2>顔認識テスト</h2>
                <p style="color:#888;margin-bottom:15px;">既存画像で顔認識（誰か判定）をテスト</p>
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
                        <label>許容度</label>
                        <select id="recogTolerance">
                            <option value="0.4">0.4（厳密）</option>
                            <option value="0.5" selected>0.5（標準）</option>
                            <option value="0.6">0.6（緩め）</option>
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
                    <button class="btn btn-primary" onclick="runRecognition()">顔認識実行</button>
                </div>
                <div id="recogStatus"></div>
                <div class="detection-result" id="recogResult"></div>
            </div>
        </div>

        <!-- ダッシュボードタブ -->
        <div id="dashboard" class="tab-content">
            <div class="card">
                <h2>検出パラメータ設定</h2>
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
                        <label>検出間隔（秒）</label>
                        <select id="cfgInterval">
                            <option value="3">3秒</option>
                            <option value="5">5秒</option>
                            <option value="10">10秒</option>
                            <option value="30">30秒</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>許容度</label>
                        <select id="cfgTolerance">
                            <option value="0.4">0.4（厳密）</option>
                            <option value="0.5">0.5（標準）</option>
                            <option value="0.6">0.6（緩め）</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>ROI選択</label>
                        <select id="cfgRoiSelect">
                            <option value="">使用しない</option>
                        </select>
                    </div>
                </div>
                <div style="text-align:right;">
                    <button class="btn btn-success" onclick="saveConfig()">設定を保存</button>
                    <span id="configStatus" style="margin-left:10px;"></span>
                </div>
            </div>
            <div class="card">
                <h2>今日の視聴時間（ラベル別）</h2>
                <div id="todayByLabel" style="display:flex;flex-wrap:wrap;gap:15px;"></div>
            </div>
            <div class="card">
                <h2>今週の視聴時間（ラベル別）</h2>
                <div id="weekByLabel" style="display:flex;flex-wrap:wrap;gap:15px;"></div>
            </div>
            <div class="card">
                <h2>検出状況</h2>
                <h3 style="color:#ffe66d;margin:10px 0;">直近5枚</h3>
                <div id="recentDetections" style="display:flex;gap:10px;overflow-x:auto;padding:10px 0;"></div>
                <h3 style="color:#ffe66d;margin:15px 0 10px;">ラベル別 最新検出</h3>
                <div id="labelDetections"></div>
            </div>
            <div class="card">
                <h2>本日の視聴状況</h2>
                <div style="height:200px;"><canvas id="todayLineChart"></canvas></div>
            </div>
            <div class="card">
                <h2>本日の時間帯別視聴（1時間単位）</h2>
                <div style="height:200px;"><canvas id="todayBarChart"></canvas></div>
            </div>
            <div class="card">
                <h2>今週の視聴状況</h2>
                <div style="height:200px;"><canvas id="weeklyChart"></canvas></div>
            </div>
            <div class="card">
                <h2>最近の検出ログ</h2>
                <div id="recentActivity" style="max-height:200px;overflow-y:auto;"></div>
            </div>
        </div>
    </div>

    <div class="modal" id="modal">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <img id="modalImage" src="">
        <div class="modal-controls">
            <button class="btn btn-danger" onclick="deleteModalImage()">削除</button>
        </div>
    </div>

    <script>
        let currentRoi = null;
        let roiDrawing = false;
        let roiStart = {x: 0, y: 0};
        let modalImagePath = '';
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
            if (tabId === 'register') { loadUnregisteredFaces(); loadRegisteredFaces(); }
            if (tabId === 'recognize') { populateRoiDropdown('recogRoiSelect'); loadRecogImages(); }
            if (tabId === 'dashboard') { populateRoiDropdown('cfgRoiSelect'); loadDashboard(); loadServiceStatus(); loadConfig(); startDashboardRefresh(); }
            else { stopDashboardRefresh(); }
        }

        // カメラ状態チェック
        function checkCameraStatus() {
            fetch('/camera_status').then(r => r.json()).then(data => {
                const overlay = document.getElementById('cameraOverlay');
                const container = document.getElementById('cameraContainer');
                if (data.service_running) {
                    overlay.style.display = 'block';
                    container.style.display = 'none';
                } else {
                    overlay.style.display = 'none';
                    container.style.display = 'block';
                }
            });
        }

        function startCamera() {
            fetch('/start_camera', { method: 'POST' }).then(r => r.json()).then(data => {
                if (data.success) {
                    document.getElementById('cameraOverlay').style.display = 'none';
                    document.getElementById('cameraContainer').style.display = 'block';
                    document.getElementById('cameraPreview').src = '/stream?' + Date.now();
                }
            });
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

        function renderRoiPresets() {
            const container = document.getElementById('roiPresetList');
            if (roiPresets.length === 0) {
                container.innerHTML = '<p style="color:#888;">保存済みROIなし</p>';
                return;
            }
            container.innerHTML = roiPresets.map((p, i) => `
                <div class="roi-preset-item">
                    <span>${p.name || 'ROI ' + (i+1)}</span>
                    <small style="color:#888;">(${p.x},${p.y} ${p.w}x${p.h})</small>
                    <span class="delete-roi" onclick="deleteRoiPreset(${i})">&times;</span>
                </div>
            `).join('');
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
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            if (currentRoi) {
                const img = document.getElementById('roiImage');
                const scaleX = canvas.width / img.naturalWidth;
                const scaleY = canvas.height / img.naturalHeight;
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
            showStatus('extractStatus', '検出中...', 'info');
            const model = document.getElementById('extractModel').value;
            const upsample = parseInt(document.getElementById('extractUpsample').value);
            const roiIndex = document.getElementById('extractRoiSelect').value;
            const images = Array.from(selectedExtractImages);
            let completed = 0;
            let totalFaces = 0;

            images.forEach(image => {
                fetch('/extract_and_save_faces', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image, model, upsample, roi_index: roiIndex})
                }).then(r => r.json()).then(data => {
                    if (data.success) totalFaces += data.count;
                    completed++;
                    if (completed === images.length) {
                        showStatus('extractStatus', `${totalFaces}個の顔を抽出しました`, 'success');
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
                        <img src="/face_image/${f.filename}">
                        <span class="badge ${f.label ? 'badge-registered' : 'badge-unregistered'}">${f.label || '未登録'}</span>
                    </div>
                `).join('');
            });
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
                container.innerHTML = Object.entries(data).map(([label, files]) => `
                    <div class="label-group">
                        <h4>${label} (${files.length}枚)</h4>
                        <div>${files.map(f => `
                            <div class="face-item">
                                <img src="/face_image/${f}">
                                <button class="delete-btn" style="display:block;top:-5px;right:-5px;width:20px;height:20px;font-size:12px;line-height:20px;" onclick="deleteFace('${f}')">&times;</button>
                            </div>
                        `).join('')}</div>
                    </div>
                `).join('');
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
            });
        }

        // 顔認識テスト
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

        function runRecognition() {
            const image = document.getElementById('recogImage').value;
            const model = document.getElementById('recogModel').value;
            const upsample = document.getElementById('recogUpsample').value;
            const tolerance = document.getElementById('recogTolerance').value;
            const roiIndex = document.getElementById('recogRoiSelect').value;
            if (!image) { alert('画像を選択してください'); return; }
            showStatus('recogStatus', '認識中...', 'info');

            fetch('/recognize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image, model, upsample: parseInt(upsample), tolerance: parseFloat(tolerance), roi_index: roiIndex})
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    const roiText = data.roi_used ? ' [ROI適用]' : '';
                    showStatus('recogStatus', `認識完了: ${data.faces.length}人検出 (${data.time}秒)${roiText}`, 'success');
                    const result = document.getElementById('recogResult');
                    if (data.faces.length === 0) {
                        result.innerHTML = '<p style="color:#ff6b6b;">顔が検出されませんでした</p>';
                    } else {
                        const nameColors = {'mio': '#ff6b6b', 'yu': '#4ecdc4', 'tsubasa': '#ffe66d', 'unknown': '#888'};
                        result.innerHTML = `
                            <img src="/recog_result?${Date.now()}" style="width:100%;border-radius:8px;">
                            <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;">
                                ${data.faces.map((f, i) => `
                                    <div class="face-box" style="border-left:4px solid ${nameColors[f.name] || '#888'};">
                                        <img src="/recog_face/${i}?${Date.now()}">
                                        <div style="color:${nameColors[f.name] || '#888'};font-weight:bold;">${f.name}</div>
                                        <div style="font-size:0.8em;color:#888;">類似度: ${Math.max(0, (1 - f.distance) * 100).toFixed(1)}%</div>
                                    </div>
                                `).join('')}
                            </div>
                        `;
                    }
                } else {
                    showStatus('recogStatus', 'エラー: ' + data.error, 'error');
                }
            });
        }

        // モーダル
        function showModal(src, path) {
            modalImagePath = path;
            document.getElementById('modalImage').src = src;
            document.getElementById('modal').classList.add('active');
        }

        function closeModal() { document.getElementById('modal').classList.remove('active'); }

        function deleteModalImage() {
            if (!confirm('削除しますか？')) return;
            fetch('/delete_capture', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: modalImagePath})
            }).then(() => { closeModal(); loadCaptures(); });
        }

        // ダッシュボード
        let dashboardRefreshInterval = null;
        const nameColors = {'mio': '#ff6b6b', 'yu': '#4ecdc4', 'tsubasa': '#ffe66d', 'unknown': '#888'};

        function startDashboardRefresh() {
            if (dashboardRefreshInterval) clearInterval(dashboardRefreshInterval);
            dashboardRefreshInterval = setInterval(() => {
                if (currentTab === 'dashboard') { loadDashboard(); loadServiceStatus(); }
            }, 10000);
        }

        function stopDashboardRefresh() {
            if (dashboardRefreshInterval) { clearInterval(dashboardRefreshInterval); dashboardRefreshInterval = null; }
        }

        let todayLineChart = null, todayBarChart = null, weeklyChart = null;

        function loadDashboard() {
            fetch('/api/dashboard').then(r => r.json()).then(data => {
                const today = new Date().toISOString().slice(0, 10);
                const names = data.target_names || ['mio', 'yu', 'tsubasa'];

                let todayHtml = '';
                names.forEach(name => {
                    const mins = data.daily[today]?.[name] || 0;
                    const color = nameColors[name] || '#888';
                    todayHtml += `<div style="background:#0f3460;padding:15px 25px;border-radius:8px;text-align:center;border-left:4px solid ${color};">
                        <div style="color:${color};font-weight:bold;margin-bottom:5px;">${name}</div>
                        <div style="font-size:1.8em;font-weight:bold;">${Math.round(mins)}</div>
                        <div style="color:#888;font-size:0.9em;">分</div>
                    </div>`;
                });
                document.getElementById('todayByLabel').innerHTML = todayHtml || '<p style="color:#888;">データなし</p>';

                let weekHtml = '';
                names.forEach(name => {
                    let total = 0;
                    Object.values(data.daily).forEach(day => { total += day[name] || 0; });
                    const color = nameColors[name] || '#888';
                    weekHtml += `<div style="background:#0f3460;padding:15px 25px;border-radius:8px;text-align:center;border-left:4px solid ${color};">
                        <div style="color:${color};font-weight:bold;margin-bottom:5px;">${name}</div>
                        <div style="font-size:1.8em;font-weight:bold;">${Math.round(total)}</div>
                        <div style="color:#888;font-size:0.9em;">分</div>
                    </div>`;
                });
                document.getElementById('weekByLabel').innerHTML = weekHtml || '<p style="color:#888;">データなし</p>';

                if (data.recent_images && data.recent_images.length > 0) {
                    document.getElementById('recentDetections').innerHTML = data.recent_images.slice(0, 5).map(img =>
                        `<img src="/detection_image/${img}" style="height:100px;border-radius:4px;border:2px solid #333;">`
                    ).join('');
                } else {
                    document.getElementById('recentDetections').innerHTML = '<p style="color:#888;">検出画像なし</p>';
                }

                if (data.label_images) {
                    let labelHtml = '';
                    names.forEach(name => {
                        const img = data.label_images[name];
                        const color = nameColors[name] || '#888';
                        const barcode = data.barcode?.[name] || [];
                        const barcodeHtml = barcode.map(v => `<div style="width:2px;height:20px;background:${v ? color : '#333'};"></div>`).join('');
                        labelHtml += `<div style="display:flex;align-items:center;gap:15px;margin-bottom:15px;padding:10px;background:#0f3460;border-radius:8px;">
                            <div style="width:80px;height:80px;background:#1a1a2e;border-radius:4px;overflow:hidden;flex-shrink:0;">
                                ${img ? `<img src="/detection_image/${img}" style="width:100%;height:100%;object-fit:cover;">` : '<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;color:#444;">N/A</div>'}
                            </div>
                            <div style="flex:1;">
                                <div style="color:${color};font-weight:bold;margin-bottom:5px;">${name}</div>
                                <div style="display:flex;gap:1px;align-items:center;">
                                    <span style="color:#888;font-size:0.8em;margin-right:5px;">1h前</span>
                                    ${barcodeHtml}
                                    <span style="color:#888;font-size:0.8em;margin-left:5px;">now</span>
                                </div>
                            </div>
                        </div>`;
                    });
                    document.getElementById('labelDetections').innerHTML = labelHtml;
                }

                if (data.today_hourly) {
                    const hours = Object.keys(data.today_hourly).sort();
                    const datasets = names.map(name => {
                        let cumulative = 0;
                        return {
                            label: name, data: hours.map(h => { cumulative += data.today_hourly[h]?.[name] || 0; return Math.round(cumulative); }),
                            borderColor: nameColors[name] || '#888', backgroundColor: 'transparent', tension: 0.3
                        };
                    });
                    if (todayLineChart) todayLineChart.destroy();
                    todayLineChart = new Chart(document.getElementById('todayLineChart'), {
                        type: 'line', data: { labels: hours.map(h => h + ':00'), datasets },
                        options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { color: '#888' }, grid: { color: '#333' } }, y: { ticks: { color: '#888' }, grid: { color: '#333' } } }, plugins: { legend: { labels: { color: '#eee' } } } }
                    });

                    const barDatasets = names.map(name => ({
                        label: name, data: hours.map(h => Math.round(data.today_hourly[h]?.[name] || 0)), backgroundColor: nameColors[name] || '#888'
                    }));
                    if (todayBarChart) todayBarChart.destroy();
                    todayBarChart = new Chart(document.getElementById('todayBarChart'), {
                        type: 'bar', data: { labels: hours.map(h => h + ':00'), datasets: barDatasets },
                        options: { responsive: true, maintainAspectRatio: false, scales: { x: { stacked: true, ticks: { color: '#888' }, grid: { color: '#333' } }, y: { stacked: true, ticks: { color: '#888' }, grid: { color: '#333' } } }, plugins: { legend: { labels: { color: '#eee' } } } }
                    });
                }

                const dates = Object.keys(data.daily).sort();
                const weekDatasets = names.map(name => ({
                    label: name, data: dates.map(d => Math.round(data.daily[d]?.[name] || 0)),
                    borderColor: nameColors[name] || '#888', backgroundColor: 'transparent', tension: 0.3
                }));
                if (weeklyChart) weeklyChart.destroy();
                weeklyChart = new Chart(document.getElementById('weeklyChart'), {
                    type: 'line', data: { labels: dates.map(d => d.slice(5)), datasets: weekDatasets },
                    options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { color: '#888' }, grid: { color: '#333' } }, y: { ticks: { color: '#888' }, grid: { color: '#333' } } }, plugins: { legend: { labels: { color: '#eee' } } } }
                });

                const recentHtml = data.recent.slice(0, 20).map(e => {
                    const color = nameColors[e.name] || '#888';
                    return `<div style="padding:5px 10px;border-bottom:1px solid #333;display:flex;justify-content:space-between;"><span>${e.timestamp}</span><span style="color:${color};">${e.name}</span></div>`;
                }).join('');
                document.getElementById('recentActivity').innerHTML = recentHtml || '<p style="color:#888;padding:10px;">データなし</p>';
            });
        }

        function loadServiceStatus() {
            fetch('/api/service_status').then(r => r.json()).then(data => {
                const el = document.getElementById('serviceStatus');
                if (data.running) { el.textContent = '稼働中'; el.style.background = '#4ecdc4'; el.style.color = '#000'; }
                else { el.textContent = '停止中'; el.style.background = '#ff6b6b'; el.style.color = '#fff'; }
            });
        }

        function serviceControl(action) {
            fetch('/api/service_control', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action}) })
            .then(r => r.json()).then(data => { setTimeout(loadServiceStatus, 1000); if (data.error) alert(data.error); });
        }

        function loadConfig() {
            fetch('/api/config').then(r => r.json()).then(cfg => {
                document.getElementById('cfgModel').value = cfg.face_model || 'hog';
                document.getElementById('cfgUpsample').value = cfg.upsample || 0;
                document.getElementById('cfgInterval').value = cfg.interval_sec || 5;
                document.getElementById('cfgTolerance').value = cfg.tolerance || 0.5;
                if (cfg.roi_index !== undefined && cfg.roi_index !== null && cfg.roi_index !== '') {
                    setTimeout(() => { document.getElementById('cfgRoiSelect').value = cfg.roi_index; }, 500);
                }
            });
        }

        function saveConfig() {
            const cfg = {
                face_model: document.getElementById('cfgModel').value,
                upsample: parseInt(document.getElementById('cfgUpsample').value),
                interval_sec: parseInt(document.getElementById('cfgInterval').value),
                tolerance: parseFloat(document.getElementById('cfgTolerance').value),
                roi_index: document.getElementById('cfgRoiSelect').value
            };
            fetch('/api/config', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(cfg) })
            .then(r => r.json()).then(data => {
                const st = document.getElementById('configStatus');
                if (data.success) { st.textContent = '保存しました（再起動で反映）'; st.style.color = '#4ecdc4'; }
                else { st.textContent = 'エラー: ' + data.error; st.style.color = '#ff6b6b'; }
                setTimeout(() => st.textContent = '', 3000);
            });
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

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
    return jsonify(result)

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

# 顔認識テスト
last_recog_result = None
last_recog_faces = []

@app.route("/recognize", methods=["POST"])
def recognize():
    global last_recog_result, last_recog_faces
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
    roi = get_roi_by_index(roi_index)
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

        # 顔画像を保存
        margin = int((orig_bottom - orig_top) * 0.2)
        crop_top = max(0, orig_top - margin)
        crop_left = max(0, orig_left - margin)
        crop_bottom = min(img.shape[0], orig_bottom + margin)
        crop_right = min(img.shape[1], orig_right + margin)
        face_crop = img[crop_top:crop_bottom, crop_left:crop_right]
        last_recog_faces.append(face_crop)

        # 描画
        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)
        cv2.rectangle(img, (orig_left, orig_top), (orig_right, orig_bottom), color, 2)
        cv2.putText(img, f"{name} ({min_distance:.2f})", (orig_left, orig_top - 10),
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

@app.route("/api/dashboard")
def api_dashboard():
    config = load_config()
    log_path = os.path.expanduser(config.get("log_path", "~/tv_watch_log.csv"))
    interval_sec = config.get("interval_sec", 5)
    target_names = config.get("target_names", ["mio", "yu", "tsubasa"])

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    cutoff = now - timedelta(days=7)
    one_hour_ago = now - timedelta(hours=1)

    daily_minutes = defaultdict(lambda: defaultdict(float))
    today_hourly = defaultdict(lambda: defaultdict(float))
    recent_entries = []
    barcode = {name: [False] * 60 for name in target_names}

    if os.path.exists(log_path):
        try:
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
                            if date_str == today_str:
                                hour_str = ts.strftime("%H")
                                today_hourly[hour_str][name] += interval_sec / 60.0
                            if ts >= one_hour_ago:
                                minute_idx = int((ts - one_hour_ago).total_seconds() / 60)
                                if 0 <= minute_idx < 60:
                                    barcode[name][minute_idx] = True
                        recent_entries.append({"timestamp": row["timestamp"], "name": name})
                    except:
                        continue
        except:
            pass

    recent_entries = recent_entries[-50:][::-1]

    recent_images = []
    label_images = {name: None for name in target_names}
    if os.path.exists(DETECTIONS_DIR):
        all_images = sorted(glob.glob(os.path.join(DETECTIONS_DIR, "*.jpg")), reverse=True)
        recent_images = [os.path.basename(f) for f in all_images[:5]]
        for img_path in all_images:
            filename = os.path.basename(img_path)
            parts = filename.replace(".jpg", "").split("_")
            if len(parts) >= 3:
                label = parts[-1]
                if label in target_names and label_images[label] is None:
                    label_images[label] = filename

    return jsonify({
        "daily": {k: dict(v) for k, v in daily_minutes.items()},
        "today_hourly": {k: dict(v) for k, v in today_hourly.items()},
        "recent": recent_entries,
        "target_names": target_names,
        "recent_images": recent_images,
        "label_images": label_images,
        "barcode": barcode
    })

@app.route("/detection_image/<filename>")
def detection_image(filename):
    path = os.path.join(DETECTIONS_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return "Not found", 404

@app.route("/api/service_status")
def api_service_status():
    try:
        result = subprocess.run(["systemctl", "is-active", "tv-watch-tracker"], capture_output=True, text=True)
        running = result.stdout.strip() == "active"
    except:
        running = False
    return jsonify({"running": running})

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
