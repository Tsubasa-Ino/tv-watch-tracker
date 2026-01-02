# TV Watch Tracker

Raspberry Pi + 顔認識で、家族のテレビ視聴時間を自動トラッキングするシステム。

**[📖 オンラインマニュアル](https://tsubasa-ino.github.io/tv-watch-tracker/manual.html)**

## 特徴

- USB カメラで顔を定期検出し、誰がいつテレビを見ていたかを記録
- **Web UI** でブラウザから簡単セットアップ（顔登録・ROI設定・パラメータ調整）
- スマホ対応レスポンシブデザイン
- ダッシュボードで視聴時間をグラフ表示
- 誤検出の再ラベリング機能で継続的に精度向上

## スクリーンショット

### ダッシュボード
視聴時間の集計、検出状況のバーコード表示、時間帯別分布グラフ

### 顔登録
撮影→顔抽出→ラベリングの一気通貫フロー

## 必要なもの

- Raspberry Pi 4（4GB推奨）
- USB カメラ
- Python 3.9+

## クイックスタート

### 1. 依存関係のインストール

```bash
# 仮想環境作成
python3 -m venv venv
source venv/bin/activate

# ライブラリインストール
pip install opencv-python face_recognition flask
```

### 2. 設定ファイル作成

```bash
cp config.json.example config.json
```

### 3. Web UI 起動

```bash
python face_manager_app.py
```

ブラウザで `http://<ラズパイIP>:5002` にアクセス。

### 4. 初期設定（Web UIで実行）

1. **撮影タブ**: 顔写真を撮影
2. **ROI設定タブ**: 検出範囲を指定（任意）
3. **顔抽出タブ**: 画像から顔を切り出し
4. **顔登録タブ**: 顔に名前を付ける
5. **テストタブ**: 認識精度を確認
6. **顔認識タブ**: パラメータ調整・サービス起動

## 設定項目

| 項目 | 説明 | 選択肢 |
|------|------|--------|
| 検出モデル | 顔検出アルゴリズム | HOG（高速）/ CNN（高精度） |
| upsample | 小さい顔の検出感度 | 0〜2 |
| 撮影間隔 | 検出間隔 | 3秒〜5分 |
| 類似度閾値 | 認識の厳しさ | 40%〜60% |
| ROI | 検出範囲の限定 | プリセット選択 |

## systemd サービス化

```bash
# 顔認識サービス
sudo cp tv-watch-tracker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tv-watch-tracker
sudo systemctl start tv-watch-tracker

# Web UI（任意）
sudo cp tv-watch-dashboard.service /etc/systemd/system/
sudo systemctl enable tv-watch-dashboard
sudo systemctl start tv-watch-dashboard
```

## 外出先からのアクセス（Tailscale）

```bash
# Tailscale インストール
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# スマホにも Tailscale アプリをインストールして同じアカウントでログイン
# Tailscale IP でアクセス: http://<Tailscale IP>:5002
```

## ファイル構成

| ファイル | 説明 |
|---------|------|
| `face_manager_app.py` | Web UI（Flask） |
| `watch_faces.py` | 顔認識サービス |
| `summarize_tv.py` | 視聴時間集計CLI |
| `rotate_logs.py` | ログローテーション |
| `config.json.example` | 設定ファイルテンプレート |
| `tv-watch-tracker.service` | 顔認識サービス定義 |
| `tv-watch-dashboard.service` | Web UIサービス定義 |

## 出力データ

### tv_watch_log.csv

```csv
timestamp,name
2025-01-02 10:00:00,mio
2025-01-02 10:00:00,yu
2025-01-02 10:00:10,mio
```

### 視聴時間計算

連続検出間の時間を合計（2分以上空いたら別セッション）

## Raspberry Pi 4 でのメモリ対策

CNN + upsample=2 はメモリ不足になることがあります。

**対策（効果順）：**
1. ROI設定で検出範囲を限定
2. upsample を 1 に下げる
3. HOG モデルに切り替える

## ライセンス

MIT

## 関連記事

- [ラズパイ×顔認識でテレビ視聴時間の見える化](https://zenn.dev/tbs_noguchi/articles/3748b3ca842a8c)
