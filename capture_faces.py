import cv2, os, time, argparse
p=argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--count", type=int, default=15)
p.add_argument("--device", type=int, default=0)
args=p.parse_args()

out_dir=os.path.expanduser(f"~/faces/{args.name}")
os.makedirs(out_dir, exist_ok=True)

cap=cv2.VideoCapture(args.device)
if not cap.isOpened(): raise SystemExit("❌ カメラを開けません")
print(f"📸 {args.name} を {args.count}枚 撮影します。3秒後に開始…")
time.sleep(3)

i=0
while i<args.count:
    ok, frame=cap.read()
    if not ok: continue
    path=os.path.join(out_dir, f"{args.name}_{i:02}.jpg")
    cv2.imwrite(path, frame)
    print("保存:", path)
    i+=1
    time.sleep(0.4)
cap.release()
print("✅ 撮影完了")
