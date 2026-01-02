---
title: "ラズパイ×顔認識でテレビ視聴ログ - Web UIで誰でも簡単セットアップ編"
emoji: "📺"
type: "tech"
topics: ["RaspberryPi", "Python", "顔認識", "Flask", "IoT"]
published: false
---

## はじめに

[前回の記事](https://zenn.dev/tbs_noguchi/articles/3748b3ca842a8c)では、Raspberry Pi と `face_recognition` ライブラリを使って、家族のテレビ視聴時間を自動ログする仕組みを作りました。

しかし、実際に運用してみると課題が見えてきました。

- カメラの設置位置を変えたら検出精度が落ちた
- 子供が成長して顔が変わり、再登録が必要になった
- 家族に「ちょっと設定変えて」と頼まれても、CLIを叩くのは敷居が高い

**「技術に詳しくない家族でも、ブラウザから設定・調整できるようにしたい」**

そんな思いから、Web UIを開発しました。本記事ではその設計コンセプトと実装を紹介します。

---

## 設計コンセプト：なぜタブ分けUIなのか

### 環境によって最適解が異なる問題

顔認識システムは、設置環境によって最適な設定が大きく異なります。

| 環境要因 | 影響 | 対策 |
|---------|------|------|
| カメラとの距離 | 遠いと顔が小さく検出困難 | upsample値を上げる / ROIで領域限定 |
| 照明条件 | 逆光・暗所で精度低下 | 閾値調整 / 撮影位置変更 |
| デバイス性能 | CNN方式はメモリ2GB以上必要 | HOG方式に切り替え |
| 人数・動き | 複数人・動きが速いと取りこぼし | 撮影間隔の調整 |

さらに、顔認識には「ラベリング」という準備作業が必須です。

```
1. 顔画像を撮影する
2. 撮影画像から顔を切り出す
3. 切り出した顔に名前（ラベル）を付ける
4. ラベル付き顔画像をエンコード（特徴量化）する
5. 実際に認識テストする
6. 精度が悪ければパラメータ調整して再テスト
```

この一連の流れを**コマンドラインでやるのは、正直しんどい**。

### 解決策：ステップバイステップのタブUI

そこで、作業フローに沿った7つのタブを用意しました。

```
[撮影] → [ROI設定] → [顔抽出] → [顔登録] → [テスト] → [顔認識] → [ダッシュボード]
```

各タブの役割：

| タブ | 目的 | 初心者向けポイント |
|-----|------|------------------|
| 撮影 | カメラで顔画像を撮る | ボタン一つで撮影、サービス稼働中も検出画像を表示 |
| ROI設定 | 検出範囲を限定する | マウスドラッグで直感的に範囲指定 |
| 顔抽出 | 画像から顔を切り出す | パラメータ変えながらプレビュー確認 |
| 顔登録 | 顔に名前を付ける | ドラッグ&ドロップでラベリング |
| テスト | 認識精度を確認する | 撮影→即結果表示のリアルタイムテスト |
| 顔認識 | サービス設定を変更 | パラメータ変更→即反映 |
| ダッシュボード | 視聴状況を確認 | グラフと検出ログで一目瞭然 |

**「左のタブから順番にやっていけば、設定が完了する」** という設計です。

---

## システム構成

```
┌─────────────────────────────────────────────────────┐
│                  Raspberry Pi 4                      │
│  ┌─────────────┐     ┌─────────────────────────┐   │
│  │ watch_faces │────▶│ tv_watch_log.csv        │   │
│  │   .py       │     │ detections/             │   │
│  │ (systemd)   │     │   ├─ latest_frame.jpg   │   │
│  └─────────────┘     │   ├─ latest_frame_meta  │   │
│         │            │   └─ detection_*.jpg    │   │
│         ▼            └─────────────────────────┘   │
│  ┌─────────────┐              │                    │
│  │  USB Camera │              ▼                    │
│  └─────────────┘     ┌─────────────────────────┐   │
│                      │ face_manager_app.py     │   │
│                      │ (Flask :5002)           │   │
│                      └─────────────────────────┘   │
│                               │                    │
└───────────────────────────────┼────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │    Browser (スマホ/PC) │
                    └───────────────────────┘
```

- **watch_faces.py**: 常駐サービス。定期的にカメラ撮影→顔認識→ログ記録
- **face_manager_app.py**: Flask製Web UI。設定変更・顔登録・ダッシュボード表示

両者はカメラを排他利用するため、Web UIでリアルタイムプレビューを使う時はサービスを一時停止します。

---

## 主要機能の実装

### 1. ROI（検出領域）設定

カメラの画角全体で顔検出すると：
- 処理負荷が高い（特にRaspberry Piでは深刻）
- 関係ない領域の誤検出が増える

**ROI（Region of Interest）** で検出範囲を限定することで、両方の問題を解決できます。

```python
# ROI適用（ピクセル値）
if USE_ROI and ROI:
    x1, y1 = ROI["x"], ROI["y"]
    x2, y2 = x1 + ROI["w"], y1 + ROI["h"]
    roi_offset_x = ROI["x"]
    roi_offset_y = ROI["y"]
    frame = frame[y1:y2, x1:x2]  # ROI領域のみ切り出し
```

Web UIでは、撮影画像上でマウスドラッグするだけでROIを設定できます。複数のプリセットを保存して切り替えることも可能です。

### 2. 検出パラメータの動的変更

顔検出の精度とパフォーマンスは、以下のパラメータで調整します。

| パラメータ | 選択肢 | トレードオフ |
|-----------|--------|-------------|
| 検出モデル | HOG / CNN | HOG:高速・低精度、CNN:低速・高精度 |
| upsample | 0〜2 | 高いほど小さい顔を検出可能、処理重い |
| 撮影間隔 | 3秒〜5分 | 短いほど精度向上、負荷増大 |
| 類似度閾値 | 40〜60% | 低いほど誤検出減、見逃し増 |

```python
@app.route("/api/config", methods=["POST"])
def api_save_config():
    config = load_config()
    updates = request.json
    for key in ["face_model", "upsample", "interval_sec", "tolerance", "roi_index"]:
        if key in updates:
            config[key] = updates[key]
    save_config(config)
    return jsonify({"success": True})
```

設定変更後、サービスを再起動すれば即座に反映されます。

### 3. 視聴時間の計算ロジック

前回記事では「撮影間隔 × 検出回数」で視聴時間を算出していましたが、撮影間隔を変更すると過去のデータと整合性が取れなくなる問題がありました。

**改善版：タイムスタンプ間隔ベースの計算**

```python
gap_threshold_sec = 120  # 2分以上空いたら別セッション

for row in reader:
    ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
    name = row["name"]

    if date_str in last_detection_by_name_date[name]:
        last_ts = last_detection_by_name_date[name][date_str]
        diff_sec = (ts - last_ts).total_seconds()

        # 閾値以内なら視聴時間としてカウント
        if 0 < diff_sec <= gap_threshold_sec:
            daily_minutes[date_str][name] += diff_sec / 60.0

    last_detection_by_name_date[name][date_str] = ts
```

これにより、撮影間隔を途中で変更しても、視聴時間の計算結果は変わりません。

### 4. 検出ログの再ラベリング

誤認識があった場合、後から修正できる機能を実装しました。

```python
@app.route("/api/relabel_detection", methods=["POST"])
def api_relabel_detection():
    # 1. メタデータのラベルを更新
    meta['faces'][idx]['name'] = new_name

    # 2. 顔画像を切り出して保存
    face_img = orig_img[top:bottom, left:right]
    cv2.imwrite(face_path, face_img)

    # 3. CSVログも更新
    if row['name'] == old_name:
        row['name'] = new_name

    # 4. 自動エンコード（認識用特徴量を更新）
    build_encoding_for_label_internal(new_name)
```

再ラベリングすると：
1. 検出ログの名前が修正される
2. 顔画像が自動抽出・登録される
3. エンコーディングが更新され、次回から正しく認識される

**誤検出を直すほど、システムが賢くなる** 仕組みです。

### 5. スマホ対応レスポンシブUI

外出先からでも視聴状況を確認できるよう、レスポンシブデザインを実装しました。

```css
@media (max-width: 768px) {
    .tabs {
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
    }
    .tab {
        padding: 12px 14px;
        font-size: 0.85em;
        white-space: nowrap;
    }
    .card { padding: 15px; }
    .grid { grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); }
}
```

Tailscaleを使えば、外出先からもセキュアにアクセス可能です。

---

## Raspberry Pi 4での実運用Tips

### メモリ不足対策

CNN方式 + upsample=2 は精度が高いですが、Raspberry Pi 4（4GB）でもメモリ不足で落ちることがあります。

```bash
# メモリ使用状況の確認
free -h

# OOMキラーのログ確認
dmesg | grep -i "out of memory"
```

**対策の優先順位：**
1. ROIを設定して処理領域を限定（効果大）
2. upsampleを1に下げる（効果中）
3. HOG方式に切り替える（効果大、精度低下あり）

### systemdサービス化

```ini
# /etc/systemd/system/tv-watch-tracker.service
[Unit]
Description=TV Watch Face Tracker
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
sudo systemctl enable tv-watch-tracker
sudo systemctl start tv-watch-tracker
```

---

## ダッシュボードで見える化

最終的なアウトプットは、ダッシュボードタブで確認できます。

### 表示項目

- **直近の検出画像**: ROI/BBox表示切替可能
- **本日/今週の視聴時間**: 人物別に集計
- **検出状況（直近3時間）**: バーコード表示で視聴パターンを可視化
- **視聴時間分布**: 時間帯別の視聴傾向をグラフ化
- **視聴時間推移**: 日別の推移をグラフ化
- **検出ログ**: 誤検出があればここから再ラベリング

---

## まとめ

### 作ったもの

- 7タブ構成のWeb UI（Flask製）
- 撮影→顔抽出→ラベリング→テスト→運用の一気通貫フロー
- スマホからも使えるレスポンシブデザイン
- 誤検出を直すと賢くなる再ラベリング機能

### 得られた知見

1. **UIの順序が重要**: 作業フローに沿ったタブ配置で、迷わず設定できる
2. **パラメータは環境依存**: 固定値ではなく、調整可能にしておくべき
3. **誤検出は学習機会**: 再ラベリング機能で継続的に精度向上

### 今後の展望

- 視聴時間の上限アラート機能
- 複数カメラ対応
- グラフのエクスポート機能

---

## 参考リンク

- [前回記事: ラズパイ×顔認識で視聴時間の見える化](https://zenn.dev/tbs_noguchi/articles/3748b3ca842a8c)
- [face_recognition ライブラリ](https://github.com/ageitgey/face_recognition)
- [Flask ドキュメント](https://flask.palletsprojects.com/)

---

家族の「テレビ見すぎ問題」、技術で解決してみませんか？
