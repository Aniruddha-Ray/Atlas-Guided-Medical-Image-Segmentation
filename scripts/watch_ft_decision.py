"""
Decision point of the Swin physics fine-tune (chunk_150_50_e100_swin_prior_huin_physics25).

Waits until that run has logged --check-epoch epochs. If its best validation Dice so far is at least --threshold, the run
is left alone (it continues to epoch 25 and the queue goes on as before). Otherwise it is stopped and a 12-epoch run
with ONLY the physics losses (HU-consistency + edge-aware; no HU input channels, no beta) is started from the same
initial weights, the best epoch (96) of chunk_150_50_e100_swin_prior.

Stopping follows the pause protocol: the queue wrapper first (otherwise it restarts the run), then run.py and the trainer
of that run, with their worker processes. The queue's later runs (from-scratch physics, image-only) are then NOT started.

    python scripts/watch_ft_decision.py                # the real thing
    python scripts/watch_ft_decision.py --dry-run --check-epoch 3   # print the decision from the metrics so far, kill nothing
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scripts.queue_ablation import (CHUNK, EDGE_WEIGHT, EXP, FT_LR, FT_RAMP, HU_WEIGHT, lr_args, run, say,  # noqa: E402
                                    swin_args)

BASE = "chunk_150_50_e100"
WATCHED = f"{BASE}_swin_prior_huin_physics25"
NEW_RUN = f"{BASE}_swin_prior_phys12"


def val_dice(run_name):
    path = os.path.join(REPO, "experiments", EXP, run_name, "metrics.csv")
    if not os.path.exists(path):
        return []
    return [float(r["val_dice"]) for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("val_dice")]


def matching_pids(*needles):
    """PIDs of python processes whose command line contains every needle."""
    ps = "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout
    pids = []
    for line in out.splitlines():
        pid, _, cmd = line.partition("|")
        if pid.strip().isdigit() and all(n in cmd for n in needles) and int(pid) != os.getpid():
            pids.append(int(pid))
    return pids


def kill(pids):
    for pid in pids:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check-epoch", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.705, help="best val Dice by the check epoch that counts as a real gain")
    p.add_argument("--epochs", type=int, default=12, help="length of the physics-losses-only run")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    say(f"[watch] waiting for {a.check_epoch} epochs of {WATCHED}; keep it if best val Dice >= {a.threshold}")
    while len(val_dice(WATCHED)) < a.check_epoch:
        time.sleep(60)
    vals = val_dice(WATCHED)[: a.check_epoch]
    best = max(vals)
    say(f"[watch] epochs 1-{a.check_epoch}: best val Dice {best:.4f} (epoch {vals.index(best) + 1}), last {vals[-1]:.4f}")
    if best >= a.threshold:
        say("[watch] gain reached the threshold: the run continues to epoch 25 and the queue is left as it is")
        return
    say(f"[watch] no gain to {a.threshold}: stopping {WATCHED}, starting {NEW_RUN} ({a.epochs} epochs, physics losses only)")
    if a.dry_run:
        say("[watch] dry run: nothing killed, nothing started")
        return
    kill(matching_pids("queue_ablation"))  # first, or it restarts the run
    kill(matching_pids(WATCHED))           # run.py and the trainer (their workers go with /T)
    time.sleep(10)
    best_ck = os.path.join("experiments", EXP, f"{BASE}_swin_prior", "checkpoint_best.pth").replace(os.sep, "/")
    sw = swin_args(48, False)
    args = lr_args(FT_LR) + [f"--init-from={best_ck}", f"--hu-weight={HU_WEIGHT:g}", f"--edge-weight={EDGE_WEIGHT:g}",
                             f"--ramp-epochs={FT_RAMP}"]
    if run(NEW_RUN, sw + args, a.epochs) != 0:
        say("[watch] failed; retrying once with activation checkpointing")
        run(NEW_RUN + "_ckpt", sw + ["--swin-checkpoint"] + args, a.epochs)
    say("[watch] done")


if __name__ == "__main__":
    main()
