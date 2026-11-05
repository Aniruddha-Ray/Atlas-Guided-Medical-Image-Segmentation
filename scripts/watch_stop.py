"""
Stop a run if it has not improved on its starting model after a given number of epochs.

Waits until RUN has logged --check-epoch epochs. If the best validation Dice among them is above --baseline (the
validation Dice of the weights the run started from), the run is left alone. Otherwise its run.py and trainer (with their
data-loader workers) are stopped; checkpoints and logs stay where they are, and rerunning the run's own command would
resume it from checkpoint_last.pth.

    python scripts/watch_stop.py --run chunk_150_50_e100_swin_prior_ext50 --check-epoch 10 --baseline 0.6963
    python scripts/watch_stop.py --run <run> --check-epoch 5 --baseline 0.6963 --dry-run   # decide, kill nothing
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def val_dice(exp, run):
    path = os.path.join(REPO, "experiments", exp, run, "metrics.csv")
    if not os.path.exists(path):
        return []
    return [float(r["val_dice"]) for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("val_dice")]


def matching_pids(needle):
    ps = "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        pid, _, cmd = line.partition("|")
        if pid.strip().isdigit() and needle in cmd and "watch_stop" not in cmd:
            pids.append(int(pid))
    return pids


def say(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [watch_stop] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)
    with open(os.path.join(REPO, "logs", "queue_ablation.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--experiment", default="exp_04_pipeline")
    p.add_argument("--check-epoch", type=int, default=10)
    p.add_argument("--baseline", type=float, required=True, help="validation Dice of the starting weights")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    say(f"waiting for {a.check_epoch} epochs of {a.run}; stop it unless some epoch beats {a.baseline:.4f}")
    while len(val_dice(a.experiment, a.run)) < a.check_epoch:
        time.sleep(60)
    vals = val_dice(a.experiment, a.run)[: a.check_epoch]
    best = max(vals)
    say(f"epochs 1-{a.check_epoch}: best {best:.4f} (epoch {vals.index(best) + 1}), last {vals[-1]:.4f}, baseline {a.baseline:.4f}")
    if best > a.baseline:
        say("improved on the starting model: the run continues")
        return
    say("no improvement on the starting model: stopping the run" + (" (dry run, nothing killed)" if a.dry_run else ""))
    if not a.dry_run:
        for pid in matching_pids(a.run):
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        say("stopped; checkpoints and logs kept")


if __name__ == "__main__":
    main()
