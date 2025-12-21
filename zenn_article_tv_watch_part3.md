---
title: "ラズパイ×顔認識で視聴時間管理【完成編】Webアプリで顔登録からダッシュボードまで全部できるようにした"
emoji: "📺"
type: "tech"
topics: ["raspberrypi", "python", "flask", "opencv", "顔認識"]
published: false
---

## はじめに

前回の記事で、Raspberry Pi + USBカメラ + 顔認識で「誰がどれくらいテレビを見ていたか」を記録するシステムの基礎を作りました。

https://zenn.dev/tsubatech/articles/3748b3ca842a8c

https://zenn.dev/tsubatech/articles/5d4c285717de8f

あれから実際に運用してみて、いくつか課題が見えてきました。

**運用してみて困ったこと：**
- 顔の登録がコマンドライン操作で面倒
- カメラの向きを調整したいのにプレビューがない
- 視聴時間を確認するのにCSVを開く必要がある
- サービスの開始/停止にSSH接続が必要

要するに「**家族が使えない**」という致命的な問題です。

ということで、**スマホからでも操作できるWebアプリ**を作りました！

![ダッシュボード画面](/images/tv-watch-dashboard.png)

## 完成したもの

### 機能一覧

| 機能 | 説明 |
|------|------|
| 📷 撮影タブ | カメラプレビュー、画像撮影 |
| 🎯 ROI設定 | 検出領域の指定（画面の一部だけ監視） |
| 👤 顔抽出 | 撮影画像から顔を自動検出・切り出し |
| ✅ 顔登録 | 抽出した顔に名前をつけて登録 |
| 🧪 テスト | 登録した顔で認識テスト |
| ⚙️ 顔認識設定 | 検出パラメータ調整、サービス制御 |
| 📊 ダッシュボード | 視聴時間グラフ、検出ログ |

全部ブラウザから操作できます。SSHもコマンドも不要！

## システム構成

```
┌─────────────────────────────────────────────────────┐
│                   Raspberry Pi 4                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Flask App   │  │ 顔認識      │  │ systemd     │ │
│  │ (Port 5002) │  │ サービス    │  │ 管理        │ │
│  │             │←→│             │←→│             │ │
│  │ - 顔登録    │  │ - 定期撮影  │  │ - 自動起動  │ │
│  │ - 設定変更  │  │ - 顔認識    │  │ - 再起動    │ │
│  │ - ダッシュ  │  │ - ログ記録  │  │             │ │
│  └─────────────┘  └─────────────┘  └─────────────┘ │
│         ↑                ↑                          │
│         │                │                          │
│    ┌────┴────┐     ┌────┴────┐                     │
│    │ config  │     │ USBカメラ│                     │
│    │ .json   │     └─────────┘                     │
│    └─────────┘                                      │
└─────────────────────────────────────────────────────┘
         ↑
    ┌────┴────┐
    │ スマホ   │
    │ ブラウザ │
    └─────────┘
```

## 実装のポイント

### 1. カメラの排他制御

Raspberry Piのカメラは同時に1つのプロセスしかアクセスできません。
Webアプリと顔認識サービスが同時にカメラを使おうとすると競合します。

**解決策：サービス稼働中はWebアプリからカメラを使わない**

```python
def is_service_running():
    """顔認識サービスが動いているか確認"""
    result = subprocess.run(
        ["systemctl", "is-active", "tv-watch-tracker"],
        capture_output=True, text=True
    )
    return result.stdout.strip() == "active"

@app.route("/capture", methods=["POST"])
def capture():
    if is_service_running():
        return jsonify({"success": False, "error": "顔認識サービス稼働中"})
    # カメラから撮影...
```

サービス稼働中は、サービスが保存した最新の検出画像を表示します。

### 2. ROI（Region of Interest）設定

テレビの前にいる人だけを検出したいので、画面の一部だけを監視対象にできます。

![ROI設定画面](/images/tv-watch-roi.png)

```python
# ROI適用
if USE_ROI and ROI:
    x1, y1 = ROI["x"], ROI["y"]
    x2, y2 = x1 + ROI["w"], y1 + ROI["h"]
    frame = frame[y1:y2, x1:x2]  # 画像を切り出し
```

ROIを設定すると：
- 処理する画像サイズが小さくなる → **高速化**
- 関係ない場所の誤検出を防げる → **精度向上**

### 3. 視聴時間の計算方法

最初は「検出回数 × 撮影間隔」で計算していましたが、問題がありました。

**問題：撮影間隔を変えると過去のデータの計算結果も変わってしまう**

例えば、30秒間隔で100回検出 → 50分のはずが、
設定を60秒に変えると → 100分と表示されてしまう。

**解決策：タイムスタンプ間の実際の時間を計算**

```python
gap_threshold_sec = 120  # 2分以上空いたら別セッション

for row in reader:
    ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
    name = row["name"]

    if name in last_detection:
        diff_sec = (ts - last_detection[name]).total_seconds()
        # 2分以内の連続検出なら視聴時間としてカウント
        if 0 < diff_sec <= gap_threshold_sec:
            viewing_time[name] += diff_sec / 60.0

    last_detection[name] = ts
```

これで撮影間隔の設定に関係なく、正確な視聴時間が計算できます。

### 4. 検出画像の保存とメタデータ

ダッシュボードで検出ログをクリックすると、その時の画像を確認できます。
BBox（顔の枠）やROI、類似度スコアの表示/非表示も切り替え可能。

```python
# 元画像とメタデータを別々に保存
cv2.imwrite(f"detection_{timestamp}_original.jpg", full_frame)

meta = {
    "timestamp": timestamp,
    "roi": {"x": roi_x, "y": roi_y, "w": roi_w, "h": roi_h},
    "faces": [
        {
            "name": "mio",
            "bbox": {"top": 100, "right": 200, "bottom": 250, "left": 50},
            "similarity": 85.5
        }
    ]
}
with open(f"detection_{timestamp}_meta.json", 'w') as f:
    json.dump(meta, f)
```

表示時に動的にオーバーレイを描画：

```python
@app.route("/detection_render/<timestamp>")
def detection_render(timestamp):
    show_bbox = request.args.get('bbox', 'true') == 'true'
    show_roi = request.args.get('roi', 'true') == 'true'

    img = cv2.imread(f"detection_{timestamp}_original.jpg")
    meta = json.load(open(f"detection_{timestamp}_meta.json"))

    if show_roi and meta.get("roi"):
        roi = meta["roi"]
        cv2.rectangle(img, (roi["x"], roi["y"]),
                      (roi["x"]+roi["w"], roi["y"]+roi["h"]),
                      (0, 165, 255), 2)  # オレンジ色

    if show_bbox:
        for face in meta["faces"]:
            bbox = face["bbox"]
            cv2.rectangle(img, (bbox["left"], bbox["top"]),
                          (bbox["right"], bbox["bottom"]),
                          (0, 255, 0), 2)  # 緑色

    return Response(cv2.imencode('.jpg', img)[1].tobytes(),
                    mimetype='image/jpeg')
```

## ダッシュボードの機能

### 視聴時間表示

- **本日/今週** の人物別視聴時間
- **時間帯別分布** グラフ（Chart.js使用）
- **日別推移** グラフ

### 検出状況バーコード

直近3時間の検出状況を1分単位でバーコード表示。
「さっきまで見てたな」がひと目でわかります。

```javascript
// 180分 = 3時間分のバーコード生成
for (let i = 0; i < 180; i++) {
    const bar = document.createElement('div');
    bar.style.width = '2px';
    bar.style.height = detection[i] ? '20px' : '5px';
    bar.style.backgroundColor = detection[i] ? color : '#333';
    container.appendChild(bar);
}
```

### 検出ログ

最新50件の検出ログを表示。クリックでその時の画像を確認。

## Raspberry Pi 4でのCNNモデル

前回の記事ではHOGモデルを使っていましたが、今回はCNNモデルも試しました。

| モデル | 精度 | 速度 | メモリ |
|--------|------|------|--------|
| HOG | △ | ◎ | ◎ |
| CNN | ◎ | △ | × |

**CNN + upsample=2 の罠**

最初、CNNモデルでupsample=2（画像を2倍に拡大して検出）に設定したら、
Raspberry Pi 4のメモリ（4GB）では足りずにOOM Killerに殺されました。

```bash
# dmesgでこんなログが...
Out of memory: Killed process 66320 (python3)
total-vm:5702600kB, anon-rss:2568336kB
```

**対策：**
- upsample=1 にする
- ROIで処理領域を限定する
- resize_width で画像を縮小してから処理

現在は CNN + upsample=2 + ROI設定 で安定動作しています。

## セットアップ手順

### 1. 必要なパッケージ

```bash
# 仮想環境作成
python3 -m venv ~/venv
source ~/venv/bin/activate

# パッケージインストール
pip install flask opencv-python face_recognition
```

### 2. サービス登録

```bash
sudo nano /etc/systemd/system/tv-watch-tracker.service
```

```ini
[Unit]
Description=TV Watch Face Recognition Tracker
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi
ExecStart=/home/pi/venv/bin/python3 /home/pi/watch_faces.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable tv-watch-tracker
sudo systemctl start tv-watch-tracker
```

### 3. Webアプリ起動

```bash
# 手動起動
python3 face_manager_app.py

# または自動起動設定
sudo nano /etc/systemd/system/face-manager.service
```

## 運用してみて

### 良かったこと

- 家族でも使える（特に妻が「これなら私でも見れる」と言ってくれた）
- スマホからリビングのカメラ映像が見れる
- 子供たちが「今日何分見た？」と自分で確認するようになった

### 課題

- 暗い部屋だと検出精度が落ちる
- テレビの光で顔が白飛びすることがある
- たまに他の家族を誤認識する（学習データが少ない）

### 今後やりたいこと

- 照明条件の自動調整
- 視聴時間の上限アラート
- LINEへの通知連携
- 複数カメラ対応

## おわりに

「子供のYouTube依存をなんとかしたい」という動機で始めたこのプロジェクト、
気づけばWebアプリまで作ってしまいました。

正直、子供たちの視聴時間が劇的に減ったわけではありません。
でも「見える化」されることで、親子で「今日は見すぎだね」という会話が生まれるようになりました。

技術で問題を完全に解決するのは難しいけど、
**技術で会話のきっかけを作る**ことはできる。
そんな学びがありました。

次回は、このシステムをさらに発展させて、
視聴時間の上限を超えたらテレビの電源を切る（！）機能を実装してみたいと思います。

---

**シリーズ記事：**
1. [子供のYouTube依存を技術で解決したい](https://zenn.dev/tsubatech/articles/3748b3ca842a8c)
2. [Raspberry Pi で「家にAIを住ませる」プロジェクト](https://zenn.dev/tsubatech/articles/5d4c285717de8f)
3. 本記事（Webアプリ完成編）

ソースコードはGitHubで公開予定です。
