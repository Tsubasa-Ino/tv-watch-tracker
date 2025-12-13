#!/usr/bin/env python3
import os
import time
import cv2

OUT_DIR = os.path.expanduser("~/new_faces/papa")
os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("カメラが開けません（/dev/video0？）")

print("camera opened.")
print("これから 10 枚撮影します。")
print("カメラの前に座って、少しずつ向きや表情を変えてください。")
time.sleep(2)

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得失敗、スキップ:", i)
        time.sleep(1)
        continue

    filename = os.path.join(OUT_DIR, f"papa_{i:02d}.jpg")
    cv2.imwrite(filename, frame)
    print("saved:", filename)

    time.sleep(1.5)

cap.release()
print("done:", OUT_DIR)
