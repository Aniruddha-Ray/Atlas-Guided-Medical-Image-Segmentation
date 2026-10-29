"""
Backbone ablation queue: waits for the running UNETR+prior run, then trains the other cells of the
{UNETR, Swin UNETR} x {image-only, with prior} grid on the same 150/50 chunk, epochs, seed and schedule, both
backbones at the default learning rate 2e-4. Per-backbone learning-rate tuning is built in but OFF by default
(--tune); it is planned for the server on the full data.

  0. Tuning (only with --tune): for each backbone, LR in {5e-5, 1e-4, 2e-4, 4e-4} (min LR = LR / 20), 25 epochs, trained on 100 of the
     150 training cases and scored on the other 50 (--tune-holdout). The real 50-case validation split is never used
     to choose settings. Score = mean validation Dice over the last 3 epochs.
  1. Swin UNETR + prior
  2. UNETR + prior again, only if its best LR is not 2e-4 (the run being waited for used 2e-4, untuned)
  3. Swin UNETR + prior + HU-consistency and edge-aware losses (--hu-weight 0.03 --edge-weight 50)
  4. UNETR, image only
  5. Swin UNETR, image only

Before any Swin run it benchmarks Swin on the free GPU (4 GB laptop card) and takes the first configuration that
fits: feature size 48, then 48 with activation checkpointing, then 24 with checkpointing. A failed run does not stop
the queue. Everything is resumable: rerun this script and finished runs are skipped.

    python scripts/queue_ablation.py --wait-pid 21192
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "exp_04_pipeline"
CHUNK = os.path.join("experiments", EXP, "chunk_150_50_e100_chunk.json")
TUNE_LRS = [5e-5, 1e-4, 2e-4, 4e-4]
TUNE_EPOCHS, TUNE_HOLDOUT, DEFAULT_LR = 25, 50, 2e-4
HU_WEIGHT, EDGE_WEIGHT = 0.03, 50.0  # new_model_plan.md steps 8 and 9
SWIN_CANDIDATES = [(48, False), (48, True), (24, True)]
LOG = os.path.join(REPO, "logs", "queue_ablation.log")


def say(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def pid_alive(pid):
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
    return str(pid) in out


def swin_args(fs, ckpt):
    return ["--backbone", "swin", f"--swin-feature-size={fs}"] + (["--swin-checkpoint"] if ckpt else [])


def bench_swin():
    """First (feature size, checkpointing) that trains a step without running out of memory, or None."""
    for fs, ckpt in SWIN_CANDIDATES:
        name = f"bench_swin_fs{fs}{'_ckpt' if ckpt else ''}"
        cmd = [sys.executable, "scripts/train_prototype_v2.py", "--experiment", EXP, "--run-name", name, "--bench",
               "--amp", "--num-workers", "2", "--seed", "0", "--chunk", CHUNK, "--gamma-init=1.0"] + swin_args(fs, ckpt)
        say(f"[bench] Swin feature_size={fs} checkpointing={ckpt}")
        code = subprocess.run(cmd, cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        res = os.path.join(REPO, "experiments", EXP, name, "bench.json")
        if code == 0 and os.path.exists(res):
            r = json.load(open(res, encoding="utf-8"))
            epoch_h = (150 * r["step_s_per_micro_batch"] + 50 * r["val_s_per_case"] * 1.2) / 3600
            say(f"[bench] fits: step {r['step_s_per_micro_batch']:.2f} s, peak {r['peak_gpu_gib']:.2f} GiB, "
                f"about {epoch_h * 3600:.0f} s/epoch, {epoch_h * 100:.1f} h for 100 epochs")
            return fs, ckpt
        say(f"[bench] failed (exit {code}); trying the next configuration")
    return None


def run(name, train_args, epochs):
    cmd = [sys.executable, "run.py", "--mode", "chunk", "--epochs", str(epochs), "--experiment", EXP, "--run-name", name,
           "--chunk-file", CHUNK, "--stages", "library,train,report", f"--train-args={' '.join(train_args)}"]
    say(f"[run] {name}: {' '.join(train_args)}")
    code = subprocess.run(cmd, cwd=REPO).returncode
    say(f"[run] {name} exited with code {code}")
    return code


def lr_args(lr):
    return [f"--lr={lr:g}", f"--min-lr={lr / 20:g}"]


def tune_score(name):
    """Mean validation Dice over the last 3 epochs of a tuning run, or None if it did not finish."""
    path = os.path.join(REPO, "experiments", EXP, name, "metrics.csv")
    if not os.path.exists(path):
        return None
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("val_dice")]
    if len(rows) < TUNE_EPOCHS:
        return None
    return sum(float(r["val_dice"]) for r in rows[-3:]) / 3


def tune(backbone, extra):
    """Best learning rate of one backbone on the tuning split (None if no run finished)."""
    scores = {}
    for lr in TUNE_LRS:
        name = f"tune_{backbone}_lr{lr:g}"
        if tune_score(name) is None:
            run(name, extra + lr_args(lr) + ["--gamma-init=1.0", f"--tune-holdout={TUNE_HOLDOUT}"], TUNE_EPOCHS)
        scores[lr] = tune_score(name)
        say(f"[tune] {backbone} lr {lr:g}: {'failed' if scores[lr] is None else f'{scores[lr]:.4f}'}")
    ok = {k: v for k, v in scores.items() if v is not None}
    if not ok:
        return None
    best = max(ok, key=ok.get)
    edge = " (at the edge of the grid: a better value may lie outside it)" if best in (min(TUNE_LRS), max(TUNE_LRS)) else ""
    say(f"[tune] {backbone}: best lr {best:g}{edge}; scores " + ", ".join(f"{k:g}={v:.4f}" for k, v in sorted(ok.items())))
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wait-pid", type=int, default=None, help="start once this process has exited")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--swin-config", default=None, metavar="FS,CKPT",
                   help="skip the Swin benchmark and use this configuration, e.g. 48,0 (feature size 48, no checkpointing); "
                        "needed to resume a Swin run with exactly the model it started with")
    p.add_argument("--tune", action="store_true",
                   help="tune each backbone's learning rate first (off by default: tuning is planned for the server, on the full data)")
    a = p.parse_args()
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    if a.wait_pid is not None:
        say(f"waiting for PID {a.wait_pid}")
        while pid_alive(a.wait_pid):
            time.sleep(60)
        say("the UNETR + prior run has exited")
    base = "chunk_150_50_e100"
    if a.swin_config:
        fs, ck = a.swin_config.split(",")
        cfg = (int(fs), bool(int(ck)))
        say(f"[bench] skipped: Swin configuration fixed to feature_size={cfg[0]} checkpointing={cfg[1]}")
    else:
        cfg = bench_swin()
    sw = swin_args(*cfg) if cfg is not None else None
    lr_vit = tune("vit", []) if a.tune else DEFAULT_LR
    lr_swin = (tune("swin", sw) if a.tune else DEFAULT_LR) if sw is not None else None
    if not a.tune:
        say(f"[tune] skipped: both backbones use the default learning rate {DEFAULT_LR:g}")
    if lr_vit is None:
        say("[tune] no UNETR tuning run finished; using the default learning rate")
        lr_vit = DEFAULT_LR
    if sw is not None and lr_swin is None:
        say("[tune] no Swin tuning run finished; using the default learning rate")
        lr_swin = DEFAULT_LR
    if sw is not None:
        run(f"{base}_swin_prior", sw + lr_args(lr_swin) + ["--gamma-init=1.0"], a.epochs)
        run(f"{base}_swin_prior_physics", sw + lr_args(lr_swin) + ["--gamma-init=1.0", f"--hu-weight={HU_WEIGHT:g}",
                                                                   f"--edge-weight={EDGE_WEIGHT:g}"], a.epochs)
    if lr_vit != DEFAULT_LR:
        run(f"{base}_vit_prior_lr{lr_vit:g}", lr_args(lr_vit) + ["--gamma-init=1.0"], a.epochs)
    run(f"{base}_vit_imageonly", lr_args(lr_vit) + ["--image-only"], a.epochs)
    if sw is not None:
        run(f"{base}_swin_imageonly", sw + lr_args(lr_swin) + ["--image-only"], a.epochs)
    say("queue finished")


if __name__ == "__main__":
    main()
