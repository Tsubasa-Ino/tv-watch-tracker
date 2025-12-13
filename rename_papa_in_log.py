#!/usr/bin/env python3
import os, csv

LOG_PATH = os.path.expanduser("~/tv_watch_log.csv")
TMP_PATH = os.path.expanduser("~/tv_watch_log.tmp")

with open(LOG_PATH, newline="", encoding="utf-8") as fin, \
     open(TMP_PATH, "w", newline="", encoding="utf-8") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)

    header = next(reader)
    writer.writerow(header)
    name_idx = header.index("name")

    for row in reader:
        if row[name_idx] == "papa":
            row[name_idx] = "tsubasa"
        writer.writerow(row)

os.replace(TMP_PATH, LOG_PATH)
print("置換完了:", LOG_PATH)
