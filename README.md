# TV Watch Tracker

Raspberry Pi上で動作する、顔認識によるテレビ視聴時間トラッカー。

## 機能

- カメラで家族の顔を認識
- 誰がいつテレビを見ていたかをCSVに記録
- 日別・人別の視聴時間を集計

## セットアップ

### 1. 依存関係のインストール

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python face_recognition
```

### 2. 設定ファイル

```bash
cp config.json.example config.json
# 必要に応じて編集
```

設定項目：
| 項目 | 説明 | デフォルト |
|------|------|-----------|
| `camera_device` | カメラデバイス番号 | 0 |
| `interval_sec` | 検出間隔（秒） | 5 |
| `tolerance` | 顔認識の閾値（小さいほど厳格） | 0.5 |
| `face_model` | 顔検出モデル（cnn/hog） | cnn |
| `target_names` | 集計対象の名前リスト | ["mio", "yu", "tsubasa"] |

### 3. 顔の登録

```bash
# 顔写真を撮影
python capture_faces.py --name <名前> --count 15

# new_faces/ に画像を配置後、エンコーディング生成
python build_encodings.py
```

### 4. 監視の開始

```bash
python watch_faces.py
```

## 自動起動（systemd）

```bash
# サービスファイルをコピー
sudo cp tv-watch-tracker.service /etc/systemd/system/

# 有効化
sudo systemctl daemon-reload
sudo systemctl enable tv-watch-tracker
sudo systemctl start tv-watch-tracker

# ステータス確認
sudo systemctl status tv-watch-tracker

# ログ確認
journalctl -u tv-watch-tracker -f
```

## Webダッシュボード

ブラウザで視聴データをグラフ表示できます。

```bash
# Flask インストール
pip install flask

# 起動
python dashboard.py
```

ブラウザで `http://<ラズパイIP>:5000` にアクセス。

### ダッシュボードの自動起動

```bash
sudo cp tv-watch-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tv-watch-dashboard
sudo systemctl start tv-watch-dashboard
```

## ログローテーション

月別にログをアーカイブし、メインログファイルを軽量に保ちます。

```bash
# 手動実行
python rotate_logs.py

# cronで毎月1日に自動実行
crontab -e
# 以下を追加:
# 0 0 1 * * /home/pi/venv/bin/python /home/pi/rotate_logs.py
```

アーカイブは `~/tv_watch_archives/` に gzip 圧縮で保存されます。

## ファイル構成

| ファイル | 説明 |
|---------|------|
| `watch_faces.py` | メイン監視スクリプト |
| `build_encodings.py` | 顔エンコーディング生成 |
| `capture_faces.py` | 顔写真撮影ユーティリティ |
| `summarize_tv.py` | 視聴時間集計 |
| `rotate_logs.py` | ログローテーション |
| `dashboard.py` | Webダッシュボード |
| `tv-watch-tracker.service` | 監視用systemdサービス |
| `tv-watch-dashboard.service` | ダッシュボード用systemdサービス |
