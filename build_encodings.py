import os, glob, pickle
import cv2
import face_recognition

base = os.path.expanduser("~/new_faces")
names, encs = [], []

print("📂 base:", base)

for person in sorted(os.listdir(base)):
    folder = os.path.join(base, person)
    if not os.path.isdir(folder):
        continue

    print(f"\n👤 Person: {person}")
    for path in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
        img = cv2.imread(path)
        if img is None:
            print("  ❌ 読み込み失敗:", path)
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")

        if len(locs) != 1:
            print(f"  ⚠ スキップ ({len(locs)} faces):", path)
            continue

        enc = face_recognition.face_encodings(rgb, locs)[0]
        names.append(person)
        encs.append(enc)
        print("  ✅ 登録:", path)

out_path = os.path.expanduser("~/encodings.pkl")
with open(out_path, "wb") as f:
    pickle.dump({"names": names, "encodings": encs}, f)

print("\n🎉 完了: 登録枚数 =", len(names), " →", out_path)
