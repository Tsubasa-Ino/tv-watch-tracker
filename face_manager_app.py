#!/usr/bin/env python3
"""
顔管理Web UI
- リアルタイムプレビュー＆撮影
- ROI設定（マウス操作）
- 顔検出テスト
- 顔切出し・ラベル付け・エンコーディング生成
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

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        os.system("sudo systemctl stop tv-watch-tracker 2>/dev/null")
        time.sleep(0.5)
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

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
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            margin: 5px;
            transition: opacity 0.3s;
        }
        .btn:hover { opacity: 0.8; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-primary { background: #00d4ff; color: #1a1a2e; }
        .btn-success { background: #4ecdc4; color: #1a1a2e; }
        .btn-danger { background: #ff6b6b; color: #fff; }
        .btn-secondary { background: #666; color: #fff; }
        .btn-small { padding: 8px 16px; font-size: 0.9em; }
        .status { padding: 10px; border-radius: 8px; margin: 10px 0; text-align: center; }
        .status.success { background: #4ecdc4; color: #1a1a2e; }
        .status.error { background: #ff6b6b; }
        .status.info { background: #0f3460; color: #00d4ff; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 10px; }
        .grid-item { position: relative; aspect-ratio: 1; background: #0f3460; border-radius: 8px; overflow: hidden; cursor: pointer; }
        .grid-item img { width: 100%; height: 100%; object-fit: cover; }
        .grid-item .overlay {
            position: absolute; bottom: 0; left: 0; right: 0;
            background: rgba(0,0,0,0.7); padding: 5px; font-size: 0.8em;
            text-align: center; color: #fff;
        }
        .grid-item .delete-btn {
            position: absolute; top: 5px; right: 5px;
            background: rgba(255,107,107,0.9); color: #fff;
            border: none; border-radius: 50%; width: 24px; height: 24px;
            cursor: pointer; display: none;
        }
        .grid-item:hover .delete-btn { display: block; }
        .grid-item.selected { outline: 3px solid #00d4ff; }
        .face-select { cursor: pointer; padding: 5px; border-radius: 8px; background: #0f3460; text-align: center; transition: all 0.2s; }
        .face-select:hover { background: #1a4a7a; }
        .face-select.selected { outline: 3px solid #4ecdc4; background: #1a5a5a; }
        .face-select img { width: 100px; height: 100px; object-fit: cover; border-radius: 4px; }
        .grid-item .filename { position: absolute; bottom: 0; left: 0; right: 0; background: rgba(0,0,0,0.7); padding: 3px; font-size: 0.7em; text-align: center; color: #fff; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
        .form-group { margin-bottom: 15px; }
        .form-group label { display: block; margin-bottom: 5px; color: #00d4ff; }
        .form-group input, .form-group select {
            width: 100%; padding: 10px; border: none; border-radius: 8px;
            font-size: 1em; background: #0f3460; color: #fff;
        }
        .roi-info { background: #0f3460; padding: 10px; border-radius: 8px; margin-top: 10px; font-family: monospace; }
        .face-list { max-height: 400px; overflow-y: auto; }
        .face-item {
            display: flex; align-items: center; gap: 15px;
            background: #0f3460; padding: 10px; border-radius: 8px; margin-bottom: 10px;
        }
        .face-item img { width: 80px; height: 80px; object-fit: cover; border-radius: 8px; }
        .face-item .info { flex: 1; }
        .face-item input { background: #16213e; }
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
    </style>
</head>
<body>
    <div class="tabs">
        <button class="tab active" onclick="showTab('camera')">カメラ</button>
        <button class="tab" onclick="showTab('roi')">ROI設定</button>
        <button class="tab" onclick="showTab('detect')">顔検出テスト</button>
        <button class="tab" onclick="showTab('register')">顔登録</button>
        <button class="tab" onclick="showTab('faces')">顔管理</button>
        <button class="tab" onclick="showTab('dashboard')">ダッシュボード</button>
    </div>
    <div class="content">
        <!-- カメラタブ -->
        <div id="camera" class="tab-content active">
            <div class="card">
                <h2>リアルタイムプレビュー</h2>
                <div class="preview-container">
                    <img id="cameraPreview" src="/stream">
                </div>
                <div style="margin-top:15px; text-align:center;">
                    <button class="btn btn-success" onclick="capture()">撮影</button>
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
                <p style="color:#888;margin-bottom:15px;">マウスでドラッグして検出領域を指定してください</p>
                <div class="preview-container" id="roiContainer">
                    <img id="roiImage" src="/snapshot">
                    <canvas id="roiCanvas"></canvas>
                </div>
                <div style="margin-top:15px;">
                    <button class="btn btn-primary" onclick="refreshRoiImage()">画像更新</button>
                    <button class="btn btn-success" onclick="saveRoi()">ROI保存</button>
                    <button class="btn btn-danger" onclick="clearRoi()">ROIクリア</button>
                </div>
                <div class="roi-info" id="roiInfo">ROI: 未設定</div>
            </div>
        </div>

        <!-- 顔検出テストタブ -->
        <div id="detect" class="tab-content">
            <div class="card">
                <h2>顔検出テスト</h2>
                <p style="color:#888;margin-bottom:15px;">パラメータを調整して検出精度を確認</p>
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
                        <label>処理サイズ <small style="color:#888;">(メモリ節約)</small></label>
                        <select id="detectResize">
                            <option value="0">元サイズ</option>
                            <option value="480">480px</option>
                            <option value="640" selected>640px</option>
                            <option value="800">800px</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>ROI適用</label>
                        <select id="detectUseRoi" onchange="loadDetectImages()">
                            <option value="1">ON</option>
                            <option value="0">OFF</option>
                        </select>
                    </div>
                </div>
                <h3>テスト画像を選択</h3>
                <div class="grid" id="detectImageGrid"></div>
                <input type="hidden" id="detectImage" value="">
                <div style="margin-top:15px;">
                    <button class="btn btn-primary" onclick="runDetection()">検出実行</button>
                </div>
                <div id="detectStatus"></div>
                <div class="detection-result" id="detectResult"></div>
            </div>
        </div>

        <!-- 顔登録タブ -->
        <div id="register" class="tab-content">
            <div class="card">
                <h2>顔の一括登録</h2>
                <p style="color:#888;margin-bottom:15px;">撮影画像から顔を検出し、まとめてラベル付け・保存</p>
                <div class="form-group">
                    <label>登録する人の名前</label>
                    <input type="text" id="registerLabel" placeholder="例: tsubasa">
                </div>
                <div class="params">
                    <div class="form-group">
                        <label>検出モデル</label>
                        <select id="registerModel">
                            <option value="hog" selected>HOG（軽量）</option>
                            <option value="cnn">CNN（高精度）</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Upsample</label>
                        <select id="registerUpsample">
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2" selected>2</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>処理サイズ</label>
                        <select id="registerResize">
                            <option value="0">元サイズ</option>
                            <option value="480">480px</option>
                            <option value="640" selected>640px</option>
                            <option value="800">800px</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>ROI適用</label>
                        <select id="registerUseRoi" onchange="loadRegisterImages()">
                            <option value="1" selected>ON</option>
                            <option value="0">OFF</option>
                        </select>
                    </div>
                </div>
                <h3>画像を選択（複数可）</h3>
                <div class="grid" id="registerImageGrid"></div>
                <div style="margin-top:15px;">
                    <button class="btn btn-primary" onclick="detectForRegister()">選択画像から顔を検出</button>
                </div>
                <div id="registerDetectStatus"></div>
            </div>
            <div class="card" id="registerResultCard" style="display:none;">
                <h2>検出された顔</h2>
                <p style="color:#888;margin-bottom:10px;">登録する顔をクリックして選択（複数可）→ 一括保存</p>
                <div style="display:flex;flex-wrap:wrap;gap:10px;" id="registerFaces"></div>
                <div style="margin-top:15px;">
                    <button class="btn btn-success" onclick="saveSelectedFaces()">選択した顔を保存</button>
                    <button class="btn btn-secondary" onclick="selectAllFaces()">全選択</button>
                    <button class="btn btn-secondary" onclick="deselectAllFaces()">全解除</button>
                </div>
                <div id="registerSaveStatus"></div>
            </div>
        </div>

        <!-- 顔管理タブ -->
        <div id="faces" class="tab-content">
            <div class="card">
                <h2>ラベル別一覧</h2>
                <p style="color:#888;margin-bottom:15px;">写真クリックで削除可能</p>
                <div id="labeledFaces"></div>
                <div id="encodingStatus"></div>
            </div>
        </div>

        <!-- ダッシュボードタブ -->
        <div id="dashboard" class="tab-content">
            <div class="card">
                <h2>監視サービス制御</h2>
                <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                    <span id="serviceStatus" style="padding:8px 16px;border-radius:8px;background:#666;">状態確認中...</span>
                    <button class="btn btn-success btn-small" onclick="serviceControl('start')">開始</button>
                    <button class="btn btn-danger btn-small" onclick="serviceControl('stop')">停止</button>
                    <button class="btn btn-secondary btn-small" onclick="serviceControl('restart')">再起動</button>
                    <button class="btn btn-primary btn-small" onclick="loadServiceStatus()">更新</button>
                </div>
            </div>
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
                        <label>縮小サイズ</label>
                        <select id="cfgResize">
                            <option value="320">320px</option>
                            <option value="480">480px</option>
                            <option value="640">640px</option>
                            <option value="0">なし</option>
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
                        <label>ROI使用</label>
                        <select id="cfgUseRoi">
                            <option value="true">有効</option>
                            <option value="false">無効</option>
                        </select>
                    </div>
                </div>
                <div style="text-align:right;">
                    <button class="btn btn-success" onclick="saveConfig()">設定を保存</button>
                    <span id="configStatus" style="margin-left:10px;"></span>
                </div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin-bottom:20px;">
                <div class="card" style="text-align:center;">
                    <h3 style="color:#00d4ff;margin-bottom:10px;">今日の合計</h3>
                    <div id="todayTotal" style="font-size:2.5em;font-weight:bold;">--</div>
                    <div style="color:#888;">分</div>
                </div>
                <div class="card" style="text-align:center;">
                    <h3 style="color:#00d4ff;margin-bottom:10px;">今週の合計</h3>
                    <div id="weekTotal" style="font-size:2.5em;font-weight:bold;">--</div>
                    <div style="color:#888;">分</div>
                </div>
                <div class="card" style="text-align:center;">
                    <h3 style="color:#00d4ff;margin-bottom:10px;">最も視聴</h3>
                    <div id="mostActive" style="font-size:2em;font-weight:bold;">--</div>
                    <div style="color:#888;">今週</div>
                </div>
            </div>
            <div class="card">
                <h2>日別視聴時間（過去7日）</h2>
                <div style="height:300px;"><canvas id="dailyChart"></canvas></div>
            </div>
            <div class="card">
                <h2>最近の検出</h2>
                <div id="recentActivity" style="max-height:300px;overflow-y:auto;"></div>
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

        // タブ切り替え
        function showTab(tabId) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`.tab[onclick="showTab('${tabId}')"]`).classList.add('active');
            document.getElementById(tabId).classList.add('active');

            if (tabId === 'camera') loadCaptures();
            if (tabId === 'roi') { refreshRoiImage(); loadRoi(); }
            if (tabId === 'detect') loadDetectImages();
            if (tabId === 'register') loadRegisterImages();
            if (tabId === 'faces') { loadLabeledFaces(); }
            if (tabId === 'dashboard') { loadDashboard(); loadServiceStatus(); loadConfig(); }
        }

        // ステータス表示
        function showStatus(elementId, message, type) {
            const el = document.getElementById(elementId);
            el.className = 'status ' + type;
            el.textContent = message;
            if (type !== 'info') setTimeout(() => { el.className = ''; el.textContent = ''; }, 5000);
        }

        // 撮影
        function capture() {
            fetch('/capture', {method: 'POST'})
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    showStatus('captureStatus', '撮影完了: ' + data.filename, 'success');
                    loadCaptures();
                } else {
                    showStatus('captureStatus', 'エラー: ' + data.error, 'error');
                }
            });
        }

        // 撮影画像一覧
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
        function refreshRoiImage() {
            const img = document.getElementById('roiImage');
            img.src = '/snapshot?' + Date.now();
            img.onload = setupRoiCanvas;
        }

        function setupRoiCanvas() {
            const img = document.getElementById('roiImage');
            const canvas = document.getElementById('roiCanvas');
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
            drawRoi();
        }

        function loadRoi() {
            fetch('/get_roi').then(r => r.json()).then(data => {
                currentRoi = data.roi;
                updateRoiInfo();
                drawRoi();
            });
        }

        function updateRoiInfo() {
            const el = document.getElementById('roiInfo');
            if (currentRoi) {
                el.textContent = `ROI: x=${currentRoi.x}, y=${currentRoi.y}, w=${currentRoi.w}, h=${currentRoi.h}`;
            } else {
                el.textContent = 'ROI: 未設定（全体を検出）';
            }
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
                ctx.strokeRect(
                    currentRoi.x * scaleX,
                    currentRoi.y * scaleY,
                    currentRoi.w * scaleX,
                    currentRoi.h * scaleY
                );

                // 半透明オーバーレイ（ROI外）
                ctx.fillStyle = 'rgba(0,0,0,0.5)';
                ctx.fillRect(0, 0, canvas.width, currentRoi.y * scaleY);
                ctx.fillRect(0, (currentRoi.y + currentRoi.h) * scaleY, canvas.width, canvas.height);
                ctx.fillRect(0, currentRoi.y * scaleY, currentRoi.x * scaleX, currentRoi.h * scaleY);
                ctx.fillRect((currentRoi.x + currentRoi.w) * scaleX, currentRoi.y * scaleY, canvas.width, currentRoi.h * scaleY);
            }
        }

        // ROIマウス操作
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

            // タッチ対応
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
        });

        function saveRoi() {
            fetch('/save_roi', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({roi: currentRoi})
            }).then(r => r.json()).then(data => {
                alert(data.success ? 'ROI保存完了' : 'エラー');
            });
        }

        function clearRoi() {
            currentRoi = null;
            updateRoiInfo();
            drawRoi();
            fetch('/save_roi', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({roi: null})
            });
        }

        // 検出テスト用画像サムネイル読み込み
        function loadDetectImages() {
            const useRoi = document.getElementById('detectUseRoi').value;
            fetch('/captures').then(r => r.json()).then(data => {
                const grid = document.getElementById('detectImageGrid');
                if (data.length === 0) {
                    grid.innerHTML = '<p style="color:#888;">撮影画像なし（カメラタブで撮影してください）</p>';
                    return;
                }
                grid.innerHTML = data.map(f => `
                    <div class="grid-item" onclick="selectDetectImage('${f}', this)">
                        <img src="/thumbnail_roi/${f}?roi=${useRoi}&${Date.now()}">
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

        // 顔登録用画像読み込み
        let selectedRegisterImages = new Set();
        let detectedFacesData = [];

        function loadRegisterImages() {
            const useRoi = document.getElementById('registerUseRoi').value;
            fetch('/captures').then(r => r.json()).then(data => {
                const grid = document.getElementById('registerImageGrid');
                if (data.length === 0) {
                    grid.innerHTML = '<p style="color:#888;">撮影画像なし</p>';
                    return;
                }
                grid.innerHTML = data.map(f => `
                    <div class="grid-item" onclick="toggleRegisterImage('${f}', this)">
                        <img src="/thumbnail_roi/${f}?roi=${useRoi}&${Date.now()}">
                        <div class="filename">${f}</div>
                    </div>
                `).join('');
                selectedRegisterImages.clear();
            });
        }

        function toggleRegisterImage(filename, element) {
            if (selectedRegisterImages.has(filename)) {
                selectedRegisterImages.delete(filename);
                element.classList.remove('selected');
            } else {
                selectedRegisterImages.add(filename);
                element.classList.add('selected');
            }
        }

        let registerParams = {model: 'hog', upsample: 2, resize: 640, useRoi: true};

        function detectForRegister() {
            if (selectedRegisterImages.size === 0) {
                alert('画像を選択してください');
                return;
            }
            showStatus('registerDetectStatus', '検出中...', 'info');
            detectedFacesData = [];

            registerParams.model = document.getElementById('registerModel').value;
            registerParams.upsample = parseInt(document.getElementById('registerUpsample').value);
            registerParams.resize = parseInt(document.getElementById('registerResize').value);
            registerParams.useRoi = document.getElementById('registerUseRoi').value === '1';

            const images = Array.from(selectedRegisterImages);
            let completed = 0;

            images.forEach(image => {
                fetch('/detect_faces_only', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({image, model: registerParams.model, upsample: registerParams.upsample, resize: registerParams.resize, use_roi: registerParams.useRoi})
                }).then(r => r.json()).then(data => {
                    if (data.success && data.faces.length > 0) {
                        data.faces.forEach((f, i) => {
                            detectedFacesData.push({image: image, idx: i});
                        });
                    }
                    completed++;
                    if (completed === images.length) {
                        showRegisterResults();
                    }
                });
            });
        }

        function showRegisterResults() {
            if (detectedFacesData.length === 0) {
                showStatus('registerDetectStatus', '顔が検出されませんでした', 'error');
                document.getElementById('registerResultCard').style.display = 'none';
                return;
            }
            showStatus('registerDetectStatus', `${detectedFacesData.length}個の顔を検出`, 'success');
            document.getElementById('registerResultCard').style.display = 'block';

            const p = registerParams;
            const container = document.getElementById('registerFaces');
            container.innerHTML = detectedFacesData.map((f, i) => `
                <div class="face-select selected" data-index="${i}" onclick="toggleFaceSelect(this)">
                    <img src="/face_crop/${f.image}/${f.idx}?ur=${p.useRoi ? 1 : 0}&model=${p.model}&upsample=${p.upsample}&resize=${p.resize}&${Date.now()}">
                </div>
            `).join('');
        }

        function toggleFaceSelect(element) {
            element.classList.toggle('selected');
        }

        function selectAllFaces() {
            document.querySelectorAll('#registerFaces .face-select').forEach(el => el.classList.add('selected'));
        }

        function deselectAllFaces() {
            document.querySelectorAll('#registerFaces .face-select').forEach(el => el.classList.remove('selected'));
        }

        function saveSelectedFaces() {
            const label = document.getElementById('registerLabel').value.trim().toLowerCase();
            if (!label) {
                alert('名前を入力してください');
                return;
            }

            const selected = document.querySelectorAll('#registerFaces .face-select.selected');
            if (selected.length === 0) {
                alert('保存する顔を選択してください');
                return;
            }

            showStatus('registerSaveStatus', '保存中...', 'info');
            let completed = 0;
            let saved = 0;
            const p = registerParams;

            selected.forEach(el => {
                const idx = parseInt(el.dataset.index);
                const faceData = detectedFacesData[idx];

                fetch('/save_face', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        image: faceData.image,
                        idx: faceData.idx,
                        label: label,
                        use_roi: p.useRoi,
                        model: p.model,
                        upsample: p.upsample,
                        resize: p.resize
                    })
                }).then(r => r.json()).then(data => {
                    if (data.success) saved++;
                    completed++;
                    if (completed === selected.length) {
                        showStatus('registerSaveStatus', `${saved}件保存しました`, 'success');
                        loadLabeledFaces();
                    }
                });
            });
        }

        // 顔検出テスト
        function runDetection() {
            const image = document.getElementById('detectImage').value;
            const model = document.getElementById('detectModel').value;
            const upsample = document.getElementById('detectUpsample').value;
            const resize = document.getElementById('detectResize').value;
            const useRoi = document.getElementById('detectUseRoi').value;

            if (!image) { alert('画像を選択してください'); return; }

            showStatus('detectStatus', '検出中...', 'info');

            fetch('/detect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image, model, upsample: parseInt(upsample), resize: parseInt(resize), use_roi: useRoi === '1'})
            }).then(r => r.json()).then(data => {
                if (data.success) {
                    const roiText = data.roi_used ? ' [ROI適用]' : '';
                    showStatus('detectStatus', `検出完了: ${data.faces.length}人 (${data.time}秒)${roiText}`, 'success');
                    const result = document.getElementById('detectResult');
                    if (data.faces.length === 0) {
                        result.innerHTML = '<p style="color:#ff6b6b;">顔が検出されませんでした</p>';
                    } else {
                        result.innerHTML = `
                            <img src="/detect_result?${Date.now()}" style="width:100%;border-radius:8px;">
                            <div style="display:flex;flex-wrap:wrap;gap:10px;margin-top:10px;">
                                ${data.faces.map((f, i) => `
                                    <div class="face-box">
                                        <img src="/face_crop/${data.image}/${i}?ur=${useRoi}&${Date.now()}">
                                        <div>顔 ${i+1}</div>
                                    </div>
                                `).join('')}
                            </div>
                            <p style="color:#888;margin-top:10px;">※ 顔の登録は「顔登録」タブで行えます</p>
                        `;
                    }
                } else {
                    showStatus('detectStatus', 'エラー: ' + data.error, 'error');
                }
            });
        }

        // ラベル別一覧
        function loadLabeledFaces() {
            fetch('/labeled_faces_status').then(r => r.json()).then(data => {
                const container = document.getElementById('labeledFaces');
                if (Object.keys(data).length === 0) {
                    container.innerHTML = '<p style="color:#888;">登録された顔なし（顔登録タブで登録してください）</p>';
                    return;
                }
                container.innerHTML = Object.entries(data).map(([label, info]) => {
                    let statusBadge = '';
                    if (info.encoded && info.newPhotos === 0) {
                        statusBadge = '<span style="background:#4ecdc4;color:#000;padding:2px 8px;border-radius:4px;font-size:0.8em;">エンコード済</span>';
                    } else if (info.encoded && info.newPhotos > 0) {
                        statusBadge = `<span style="background:#ffe66d;color:#000;padding:2px 8px;border-radius:4px;font-size:0.8em;">+${info.newPhotos}枚 未反映</span>`;
                    } else {
                        statusBadge = '<span style="background:#ff6b6b;color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8em;">未エンコード</span>';
                    }
                    const encodedSet = new Set(info.encodedFiles || []);
                    return `
                        <div style="background:#0f3460;padding:15px;border-radius:8px;margin-bottom:15px;">
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                                <h3 style="margin:0;">${label} (${info.photos.length}枚) ${statusBadge}</h3>
                                <button class="btn btn-success btn-small" onclick="buildEncodingForLabel('${label}')">エンコード</button>
                            </div>
                            <div class="grid">
                                ${info.photos.map(f => {
                                    const isEncoded = info.encodedFiles && info.encodedFiles.includes(f);
                                    const style = isEncoded
                                        ? 'outline:3px solid #4ecdc4;'
                                        : 'outline:3px solid #ff6b6b;';
                                    const badge = isEncoded
                                        ? '<span style="position:absolute;top:3px;left:3px;background:#4ecdc4;color:#000;font-size:0.6em;padding:1px 4px;border-radius:3px;">✓</span>'
                                        : '<span style="position:absolute;top:3px;left:3px;background:#ff6b6b;color:#fff;font-size:0.6em;padding:1px 4px;border-radius:3px;">未</span>';
                                    return `
                                        <div class="grid-item" style="${style}">
                                            ${badge}
                                            <img src="/face_image/${f}">
                                            <button class="delete-btn" onclick="deleteFace('${f}')">&times;</button>
                                        </div>
                                    `;
                                }).join('')}
                            </div>
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
            }).then(() => loadLabeledFaces());
        }

        // エンコーディング生成（ラベル別）
        function buildEncodingForLabel(label) {
            showStatus('encodingStatus', `${label} のエンコーディング生成中...`, 'info');
            fetch('/build_encoding_for_label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({label})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    showStatus('encodingStatus', `${label}: ${data.count}件のエンコーディングを生成`, 'success');
                    loadLabeledFaces();
                } else {
                    showStatus('encodingStatus', 'エラー: ' + data.error, 'error');
                }
            });
        }

        // モーダル
        function showModal(src, path) {
            modalImagePath = path;
            document.getElementById('modalImage').src = src;
            document.getElementById('modal').classList.add('active');
        }

        function closeModal() {
            document.getElementById('modal').classList.remove('active');
        }

        function deleteModalImage() {
            if (!confirm('削除しますか？')) return;
            fetch('/delete_capture', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: modalImagePath})
            }).then(() => { closeModal(); loadCaptures(); });
        }

        // ダッシュボード
        let dashboardChart = null;
        const nameColors = {'mio': '#ff6b6b', 'yu': '#4ecdc4', 'tsubasa': '#ffe66d', 'unknown': '#888', 'none': '#444'};

        function loadDashboard() {
            fetch('/api/dashboard').then(r => r.json()).then(data => {
                // 今日の合計
                const today = new Date().toISOString().slice(0, 10);
                let todayTotal = 0;
                if (data.daily[today]) {
                    todayTotal = Object.values(data.daily[today]).reduce((a, b) => a + b, 0);
                }
                document.getElementById('todayTotal').textContent = Math.round(todayTotal);

                // 週間合計
                let weekTotal = 0;
                let personTotals = {};
                Object.values(data.daily).forEach(day => {
                    Object.entries(day).forEach(([name, mins]) => {
                        weekTotal += mins;
                        personTotals[name] = (personTotals[name] || 0) + mins;
                    });
                });
                document.getElementById('weekTotal').textContent = Math.round(weekTotal);

                // 最も視聴した人
                let mostActive = '--';
                let maxMins = 0;
                Object.entries(personTotals).forEach(([name, mins]) => {
                    if (mins > maxMins) { maxMins = mins; mostActive = name; }
                });
                document.getElementById('mostActive').textContent = mostActive;

                // チャート
                const dates = Object.keys(data.daily).sort();
                const names = data.target_names || ['mio', 'yu', 'tsubasa'];
                const datasets = names.map(name => ({
                    label: name,
                    data: dates.map(d => Math.round(data.daily[d]?.[name] || 0)),
                    backgroundColor: nameColors[name] || '#888',
                    borderColor: nameColors[name] || '#888',
                    borderWidth: 1
                }));

                if (dashboardChart) dashboardChart.destroy();
                dashboardChart = new Chart(document.getElementById('dailyChart'), {
                    type: 'bar',
                    data: { labels: dates.map(d => d.slice(5)), datasets: datasets },
                    options: {
                        responsive: true, maintainAspectRatio: false,
                        scales: {
                            x: { stacked: true, ticks: { color: '#888' }, grid: { color: '#333' } },
                            y: { stacked: true, ticks: { color: '#888' }, grid: { color: '#333' } }
                        },
                        plugins: { legend: { labels: { color: '#eee' } } }
                    }
                });

                // 最近の検出
                const recentHtml = data.recent.slice(0, 30).map(e => {
                    const color = nameColors[e.name] || '#888';
                    return `<div style="padding:5px 10px;border-bottom:1px solid #333;display:flex;justify-content:space-between;">
                        <span>${e.timestamp}</span><span style="color:${color};">${e.name}</span>
                    </div>`;
                }).join('');
                document.getElementById('recentActivity').innerHTML = recentHtml || '<p style="color:#888;padding:10px;">データなし</p>';
            });
        }

        function loadServiceStatus() {
            fetch('/api/service_status').then(r => r.json()).then(data => {
                const el = document.getElementById('serviceStatus');
                if (data.running) {
                    el.textContent = '稼働中';
                    el.style.background = '#4ecdc4';
                    el.style.color = '#000';
                } else {
                    el.textContent = '停止中';
                    el.style.background = '#ff6b6b';
                    el.style.color = '#fff';
                }
            });
        }

        function serviceControl(action) {
            fetch('/api/service_control', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action: action})
            }).then(r => r.json()).then(data => {
                setTimeout(loadServiceStatus, 1000);
                if (data.error) alert(data.error);
            });
        }

        function loadConfig() {
            fetch('/api/config').then(r => r.json()).then(cfg => {
                document.getElementById('cfgModel').value = cfg.face_model || 'hog';
                document.getElementById('cfgUpsample').value = cfg.upsample || 0;
                document.getElementById('cfgResize').value = cfg.resize_width || 640;
                document.getElementById('cfgInterval').value = cfg.interval_sec || 5;
                document.getElementById('cfgTolerance').value = cfg.tolerance || 0.5;
                document.getElementById('cfgUseRoi').value = cfg.use_roi ? 'true' : 'false';
            });
        }

        function saveConfig() {
            const cfg = {
                face_model: document.getElementById('cfgModel').value,
                upsample: parseInt(document.getElementById('cfgUpsample').value),
                resize_width: parseInt(document.getElementById('cfgResize').value),
                interval_sec: parseInt(document.getElementById('cfgInterval').value),
                tolerance: parseFloat(document.getElementById('cfgTolerance').value),
                use_roi: document.getElementById('cfgUseRoi').value === 'true'
            };
            fetch('/api/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(cfg)
            }).then(r => r.json()).then(data => {
                const st = document.getElementById('configStatus');
                if (data.success) {
                    st.textContent = '保存しました（再起動で反映）';
                    st.style.color = '#4ecdc4';
                } else {
                    st.textContent = 'エラー: ' + data.error;
                    st.style.color = '#ff6b6b';
                }
                setTimeout(() => st.textContent = '', 3000);
            });
        }

        // 初期化
        loadCaptures();
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    get_camera()
    return render_template_string(HTML_TEMPLATE)

# ストリーミング
def gen_frames():
    cam = get_camera()
    while True:
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.1)
            continue
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/stream")
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/snapshot")
def snapshot():
    cam = get_camera()
    ret, frame = cam.read()
    if not ret:
        return "Camera error", 500
    _, jpeg = cv2.imencode('.jpg', frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

# 撮影
@app.route("/capture", methods=["POST"])
def capture():
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

    use_roi = request.args.get("roi", "1") == "1"

    img = cv2.imread(path)
    h, w = img.shape[:2]

    # サムネイルサイズに縮小
    thumb_size = 200
    scale = thumb_size / max(h, w)
    thumb = cv2.resize(img, (int(w * scale), int(h * scale)))

    # ROI描画
    if use_roi:
        config = load_config()
        roi = config.get("roi")
        if roi:
            x = int(roi["x"] * scale)
            y = int(roi["y"] * scale)
            rw = int(roi["w"] * scale)
            rh = int(roi["h"] * scale)
            # 半透明オーバーレイ（ROI外を暗く）
            overlay = thumb.copy()
            cv2.rectangle(overlay, (0, 0), (thumb.shape[1], thumb.shape[0]), (0, 0, 0), -1)
            cv2.rectangle(overlay, (x, y), (x + rw, y + rh), (0, 0, 0), -1)
            mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
            mask[y:y+rh, x:x+rw] = 0
            mask[mask > 0] = 128
            thumb = cv2.addWeighted(thumb, 1, overlay, 0, 0)
            # ROI外を暗く
            dark = thumb.copy()
            dark[mask > 0] = (dark[mask > 0] * 0.4).astype('uint8')
            thumb = dark
            # ROI枠
            cv2.rectangle(thumb, (x, y), (x + rw, y + rh), (0, 212, 255), 2)

    _, jpeg = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

# ROI
@app.route("/get_roi")
def get_roi():
    config = load_config()
    return jsonify({"roi": config.get("roi")})

@app.route("/save_roi", methods=["POST"])
def save_roi():
    config = load_config()
    config["roi"] = request.json.get("roi")
    save_config(config)
    return jsonify({"success": True})

# 顔検出テスト
last_detect_result = None
last_detect_image = None

last_detect_use_roi = True

@app.route("/detect", methods=["POST"])
def detect():
    global last_detect_result, last_detect_image, last_detect_use_roi
    data = request.json
    image = data.get("image")
    model = data.get("model", "hog")
    upsample = data.get("upsample", 2)
    resize = data.get("resize", 640)
    use_roi = data.get("use_roi", True)

    path = os.path.join(CAPTURES_DIR, image)
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "画像が見つかりません"})

    img = cv2.imread(path)
    h, w = img.shape[:2]

    # ROI適用（use_roiがTrueの場合のみ）
    config = load_config()
    roi = config.get("roi") if use_roi else None
    roi_used = roi is not None

    if roi:
        x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        img_roi = img[y:y+rh, x:x+rw]
        roi_offset = (x, y)
    else:
        img_roi = img
        roi_offset = (0, 0)

    # リサイズ
    h_roi, w_roi = img_roi.shape[:2]
    if resize > 0 and max(h_roi, w_roi) > resize:
        scale = resize / max(h_roi, w_roi)
        small = cv2.resize(img_roi, (int(w_roi * scale), int(h_roi * scale)))
    else:
        scale = 1.0
        small = img_roi

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    start = time.time()
    face_locations = face_recognition.face_locations(rgb, model=model, number_of_times_to_upsample=upsample)
    elapsed = round(time.time() - start, 2)

    # 座標を元画像に変換
    faces = []
    for (top, right, bottom, left) in face_locations:
        top = int(top / scale) + roi_offset[1]
        right = int(right / scale) + roi_offset[0]
        bottom = int(bottom / scale) + roi_offset[1]
        left = int(left / scale) + roi_offset[0]
        faces.append({"top": top, "right": right, "bottom": bottom, "left": left})
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    # ROI枠描画
    if roi:
        cv2.rectangle(img, (roi["x"], roi["y"]), (roi["x"]+roi["w"], roi["y"]+roi["h"]), (0, 212, 255), 2)

    cv2.putText(img, f"Faces: {len(faces)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    last_detect_result = img
    last_detect_image = image
    last_detect_use_roi = use_roi

    return jsonify({"success": True, "faces": faces, "time": elapsed, "image": image, "roi_used": roi_used})

@app.route("/detect_faces_only", methods=["POST"])
def detect_faces_only():
    """顔登録用の検出エンドポイント"""
    data = request.json
    image = data.get("image")
    model = data.get("model", "hog")
    upsample = data.get("upsample", 2)
    resize = data.get("resize", 640)
    use_roi = data.get("use_roi", True)

    path = os.path.join(CAPTURES_DIR, image)
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "画像が見つかりません"})

    img = cv2.imread(path)
    config = load_config()
    roi = config.get("roi") if use_roi else None

    if roi:
        x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        img_roi = img[y:y+rh, x:x+rw]
        roi_offset = (x, y)
    else:
        img_roi = img
        roi_offset = (0, 0)

    h_roi, w_roi = img_roi.shape[:2]
    if resize > 0 and max(h_roi, w_roi) > resize:
        scale = resize / max(h_roi, w_roi)
        small = cv2.resize(img_roi, (int(w_roi * scale), int(h_roi * scale)))
    else:
        scale = 1.0
        small = img_roi

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, model=model, number_of_times_to_upsample=upsample)

    faces = []
    for (top, right, bottom, left) in face_locations:
        faces.append({"top": top, "right": right, "bottom": bottom, "left": left})

    return jsonify({"success": True, "faces": faces, "image": image})

@app.route("/detect_result")
def detect_result():
    if last_detect_result is None:
        return "No result", 404
    _, jpeg = cv2.imencode('.jpg', last_detect_result)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route("/face_crop/<image>/<int:idx>")
def face_crop(image, idx):
    path = os.path.join(CAPTURES_DIR, image)
    if not os.path.exists(path):
        return "Not found", 404

    use_roi = request.args.get("ur", "1") == "1"
    model = request.args.get("model", "hog")
    upsample = int(request.args.get("upsample", "2"))
    resize = int(request.args.get("resize", "640"))

    img = cv2.imread(path)
    config = load_config()
    roi = config.get("roi") if use_roi else None

    # 再検出
    if roi:
        x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        img_roi = img[y:y+rh, x:x+rw]
        roi_offset = (x, y)
    else:
        img_roi = img
        roi_offset = (0, 0)

    h_roi, w_roi = img_roi.shape[:2]
    if resize > 0 and max(h_roi, w_roi) > resize:
        scale = resize / max(h_roi, w_roi)
        small = cv2.resize(img_roi, (int(w_roi * scale), int(h_roi * scale)))
    else:
        scale = 1.0
        small = img_roi

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, model=model, number_of_times_to_upsample=upsample)

    if idx >= len(face_locations):
        return "Not found", 404

    top, right, bottom, left = face_locations[idx]
    top = int(top / scale) + roi_offset[1]
    right = int(right / scale) + roi_offset[0]
    bottom = int(bottom / scale) + roi_offset[1]
    left = int(left / scale) + roi_offset[0]

    # マージン追加
    margin = int((bottom - top) * 0.3)
    top = max(0, top - margin)
    left = max(0, left - margin)
    bottom = min(img.shape[0], bottom + margin)
    right = min(img.shape[1], right + margin)

    face_img = img[top:bottom, left:right]
    _, jpeg = cv2.imencode('.jpg', face_img)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route("/save_face", methods=["POST"])
def save_face():
    data = request.json
    image = data.get("image")
    idx = data.get("idx")
    label = data.get("label", "").strip().lower()
    use_roi = data.get("use_roi", True)
    model = data.get("model", "hog")
    upsample = data.get("upsample", 2)
    resize = data.get("resize", 640)

    if not label:
        return jsonify({"success": False, "error": "ラベルが必要です"})

    path = os.path.join(CAPTURES_DIR, image)
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "画像が見つかりません"})

    img = cv2.imread(path)
    config = load_config()
    roi = config.get("roi") if use_roi else None

    if roi:
        x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        img_roi = img[y:y+rh, x:x+rw]
        roi_offset = (x, y)
    else:
        img_roi = img
        roi_offset = (0, 0)

    h_roi, w_roi = img_roi.shape[:2]
    if resize > 0 and max(h_roi, w_roi) > resize:
        scale = resize / max(h_roi, w_roi)
        small = cv2.resize(img_roi, (int(w_roi * scale), int(h_roi * scale)))
    else:
        scale = 1.0
        small = img_roi

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, model=model, number_of_times_to_upsample=upsample)

    if idx >= len(face_locations):
        return jsonify({"success": False, "error": "顔が見つかりません"})

    top, right, bottom, left = face_locations[idx]
    top = int(top / scale) + roi_offset[1]
    right = int(right / scale) + roi_offset[0]
    bottom = int(bottom / scale) + roi_offset[1]
    left = int(left / scale) + roi_offset[0]

    margin = int((bottom - top) * 0.3)
    top = max(0, top - margin)
    left = max(0, left - margin)
    bottom = min(img.shape[0], bottom + margin)
    right = min(img.shape[1], right + margin)

    face_img = img[top:bottom, left:right]
    import uuid
    filename = f"face_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
    cv2.imwrite(os.path.join(FACES_DIR, filename), face_img)

    meta_path = os.path.join(FACES_DIR, filename + ".json")
    with open(meta_path, "w") as f:
        json.dump({"source": image, "label": label}, f)

    return jsonify({"success": True, "filename": filename})

# 顔切り出し
@app.route("/extract_faces", methods=["POST"])
def extract_faces():
    image = request.json.get("image")
    path = os.path.join(CAPTURES_DIR, image)
    if not os.path.exists(path):
        return jsonify({"success": False, "error": "画像が見つかりません"})

    img = cv2.imread(path)
    config = load_config()
    roi = config.get("roi")

    if roi:
        x, y, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        img_roi = img[y:y+rh, x:x+rw]
        roi_offset = (x, y)
    else:
        img_roi = img
        roi_offset = (0, 0)

    h_roi, w_roi = img_roi.shape[:2]
    resize = 640
    if max(h_roi, w_roi) > resize:
        scale = resize / max(h_roi, w_roi)
        small = cv2.resize(img_roi, (int(w_roi * scale), int(h_roi * scale)))
    else:
        scale = 1.0
        small = img_roi

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=2)

    count = 0
    for i, (top, right, bottom, left) in enumerate(face_locations):
        top = int(top / scale) + roi_offset[1]
        right = int(right / scale) + roi_offset[0]
        bottom = int(bottom / scale) + roi_offset[1]
        left = int(left / scale) + roi_offset[0]

        margin = int((bottom - top) * 0.3)
        top = max(0, top - margin)
        left = max(0, left - margin)
        bottom = min(img.shape[0], bottom + margin)
        right = min(img.shape[1], right + margin)

        face_img = img[top:bottom, left:right]
        filename = f"face_{int(time.time())}_{i}.jpg"
        cv2.imwrite(os.path.join(FACES_DIR, filename), face_img)

        # メタデータ保存
        meta_path = os.path.join(FACES_DIR, filename + ".json")
        with open(meta_path, "w") as f:
            json.dump({"source": image, "label": ""}, f)
        count += 1

    return jsonify({"success": True, "count": count})

@app.route("/faces")
def faces():
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

@app.route("/face_image/<filename>")
def face_image(filename):
    path = os.path.join(FACES_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype='image/jpeg')
    return "Not found", 404

@app.route("/update_label", methods=["POST"])
def update_label():
    data = request.json
    filename = data.get("filename")
    label = data.get("label", "").strip().lower()
    meta_path = os.path.join(FACES_DIR, filename + ".json")

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta["label"] = label
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return jsonify({"success": True})

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

@app.route("/labeled_faces_status")
def labeled_faces_status():
    """ラベル別の写真一覧とエンコーディング状態を返す"""
    files = glob.glob(os.path.join(FACES_DIR, "*.jpg"))

    # 現在のエンコーディング情報を読み込み
    encoded_files = {}
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                enc_data = pickle.load(f)
                # エンコード済みファイルのリストを取得
                if "files" in enc_data:
                    for label, filelist in enc_data["files"].items():
                        encoded_files[label] = set(filelist)
        except:
            pass

    result = {}
    for f in files:
        filename = os.path.basename(f)
        meta_path = f + ".json"
        if os.path.exists(meta_path):
            with open(meta_path) as mf:
                label = json.load(mf).get("label", "")
                if label:
                    if label not in result:
                        result[label] = {"photos": [], "encoded": False, "newPhotos": 0, "encodedFiles": []}
                    result[label]["photos"].append(filename)

    # エンコーディング状態を確認
    for label in result:
        if label in encoded_files:
            result[label]["encoded"] = True
            result[label]["encodedFiles"] = list(encoded_files[label])
            # 新規写真の数をカウント
            current_photos = set(result[label]["photos"])
            encoded_set = encoded_files[label]
            result[label]["newPhotos"] = len(current_photos - encoded_set)
        else:
            result[label]["encoded"] = False
            result[label]["newPhotos"] = len(result[label]["photos"])

    return jsonify(result)

@app.route("/build_encoding_for_label", methods=["POST"])
def build_encoding_for_label():
    """特定ラベルのエンコーディングを生成"""
    target_label = request.json.get("label")
    if not target_label:
        return jsonify({"success": False, "error": "ラベルが必要です"})

    # 既存のエンコーディングを読み込み
    existing_data = {"names": [], "encodings": [], "files": {}}
    if os.path.exists(ENCODINGS_PATH):
        try:
            with open(ENCODINGS_PATH, "rb") as f:
                existing_data = pickle.load(f)
                if "files" not in existing_data:
                    existing_data["files"] = {}
        except:
            pass

    # 対象ラベル以外のエンコーディングを保持
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

    # 対象ラベルの写真からエンコーディングを生成
    files = glob.glob(os.path.join(FACES_DIR, "*.jpg"))
    encoded_files_list = []
    count = 0

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

        # 切り出し画像は既に顔領域なので、画像全体を顔として扱う
        # まず通常の検出を試み、失敗したら画像全体を使用
        face_locations = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=1)

        if len(face_locations) == 0:
            # 顔が検出できない場合、画像全体を顔として扱う
            h, w = rgb.shape[:2]
            face_locations = [(0, w, h, 0)]
        elif len(face_locations) > 1:
            # 複数検出された場合は最大の顔を使用
            face_locations = [max(face_locations, key=lambda x: (x[2]-x[0]) * (x[1]-x[3]))]

        try:
            enc = face_recognition.face_encodings(rgb, face_locations)[0]
            new_names.append(target_label)
            new_encodings.append(enc)
            encoded_files_list.append(filename)
            count += 1
        except:
            continue

    if encoded_files_list:
        new_files[target_label] = encoded_files_list

    # 保存
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"names": new_names, "encodings": new_encodings, "files": new_files}, f)

    return jsonify({"success": True, "count": count})

# ダッシュボードAPI
import csv
from datetime import datetime, timedelta
from collections import defaultdict
import subprocess

LOG_PATH = os.path.expanduser("~/tv_watch_log.csv")

@app.route("/api/dashboard")
def api_dashboard():
    """ダッシュボードデータを返す"""
    config = load_config()
    log_path = os.path.expanduser(config.get("log_path", "~/tv_watch_log.csv"))
    interval_sec = config.get("interval_sec", 5)
    target_names = config.get("target_names", ["mio", "yu", "tsubasa"])

    cutoff = datetime.now() - timedelta(days=7)
    daily_minutes = defaultdict(lambda: defaultdict(float))
    recent_entries = []

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
                        recent_entries.append({"timestamp": row["timestamp"], "name": name})
                    except (ValueError, KeyError):
                        continue
        except:
            pass

    recent_entries = recent_entries[-50:][::-1]

    return jsonify({
        "daily": {k: dict(v) for k, v in daily_minutes.items()},
        "recent": recent_entries,
        "target_names": target_names
    })

@app.route("/api/service_status")
def api_service_status():
    """監視サービスの状態を返す"""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "tv-watch-tracker"],
            capture_output=True, text=True
        )
        running = result.stdout.strip() == "active"
    except:
        running = False
    return jsonify({"running": running})

@app.route("/api/service_control", methods=["POST"])
def api_service_control():
    """監視サービスを制御"""
    action = request.json.get("action")
    if action not in ["start", "stop", "restart"]:
        return jsonify({"error": "Invalid action"})

    try:
        # カメラを解放してからサービスを操作
        if action in ["start", "restart"]:
            release_camera()

        result = subprocess.run(
            ["sudo", "systemctl", action, "tv-watch-tracker"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return jsonify({"error": result.stderr})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/config")
def api_get_config():
    """設定を取得"""
    return jsonify(load_config())

@app.route("/api/config", methods=["POST"])
def api_save_config():
    """設定を保存"""
    try:
        config = load_config()
        updates = request.json
        for key in ["face_model", "upsample", "resize_width", "interval_sec", "tolerance", "use_roi"]:
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
