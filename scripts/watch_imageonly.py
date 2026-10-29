"""Stops the UNETR image-only run at epoch 50 if it is still weak (best validation Dice below the threshold)."""
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

RUN_PID, THRESHOLD, AT_EPOCH = int(sys.argv[1]), 0.20, 50
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS = os.path.join(REPO, "experiments", "exp_04_pipeline", "chunk_150_50_e100_vit_imageonly", "metrics.csv")
LOG = os.path.join(REPO, "logs", "imageonly_watch.log")


def say(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def alive(pid):
    return str(pid) in subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout


say(f"watching run PID {RUN_PID}: stop at epoch {AT_EPOCH} if best val Dice < {THRESHOLD}")
while alive(RUN_PID):
    rows = [r for r in csv.DictReader(open(METRICS, encoding="utf-8")) if r.get("val_dice")] if os.path.exists(METRICS) else []
    if rows and int(rows[-1]["epoch"]) >= AT_EPOCH:
        best = max(float(r["val_dice"]) for r in rows)
        if best < THRESHOLD:
            say(f"epoch {rows[-1]['epoch']}: best val Dice {best:.4f} < {THRESHOLD}; stopping the run")
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(RUN_PID)], capture_output=True)
        else:
            say(f"epoch {rows[-1]['epoch']}: best val Dice {best:.4f} >= {THRESHOLD}; the run continues")
        break
    time.sleep(120)
say("watcher done")
