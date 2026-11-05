"""
Unattended 5-fold cross-validation of the atlas-prior Swin UNETR on the whole data (server, one worker per GPU).

Stages, all idempotent: rerunning the same command continues where it stopped (finished folds are skipped, a stopped fold
resumes from its checkpoint_last.pth).

  1. register   run.py --mode full --stages library,register   (every non-atlas case without a cached registration;
                --reg-shards parallel processes; cases that still fail are dropped from the folds and listed)
  2. folds      scripts/make_folds.py: 5 folds over the usable cases, stratified by source, fixed seed, never regenerated
  3. train      one worker per GPU takes the next unfinished fold: run.py --mode chunk --chunk-file fold_k --epochs N
                with the recipe of the laptop runs (Swin 48, lr 2e-4 -> 1e-5, gamma-init 1, prior dropout, KL 0.1).
                A crashed fold is retried (up to --retries), resuming from its checkpoint.
  4. summarize  <exp>/cv_summary.md and cv_folds.csv: per-fold and mean +- sd of the validation Dice, per organ.
                Written after every finished fold, so partial results are always there.

    nohup python server/run_cv.py --epochs 100 > ~/atlas/cv5.out 2>&1 &      # start and leave
    python server/run_cv.py --summarize                                      # results so far, any time
    python server/run_cv.py --epochs 100 --dry-run                           # plan only

Primary number: the validation Dice at the LAST epoch (nothing is selected on the validation fold). The best-epoch number
is also given but is optimistic, because the epoch is picked on the same cases.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
NAMES = ["spleen", "r_kidney", "l_kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "ivc", "pancreas",
         "r_adrenal", "l_adrenal"]
RECIPE = ["--backbone", "swin", "--swin-feature-size=48", "--lr=0.0002", "--min-lr=1e-05", "--gamma-init=1.0"]
lock = threading.Lock()


def say(log_path, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with lock:
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def detect_gpus():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        return [str(i) for i, ln in enumerate(out.splitlines()) if ln.startswith("GPU")]
    except Exception:
        return []


def run_dir(exp, fold, epochs):
    return os.path.join(REPO, "experiments", exp, f"cv_fold{fold}_e{epochs}")


def finished(exp, fold, epochs):
    return os.path.exists(os.path.join(run_dir(exp, fold, epochs), "summary.json"))


def read_metrics(path):
    return [r for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("val_dice")] if os.path.exists(path) else []


def summarize(exp, epochs, folds, log_path):
    rows, lines = [], []
    for k in folds:
        m = read_metrics(os.path.join(run_dir(exp, k, epochs), "metrics.csv"))
        if not m:
            continue
        val = [float(r["val_dice"]) for r in m]
        bi = max(range(len(val)), key=val.__getitem__)
        organ = lambda r: [float(r[f"o{j}"]) if r.get(f"o{j}") not in (None, "") else float("nan") for j in range(1, 13)]
        rows.append({"fold": k, "epochs_done": len(m), "done": finished(exp, k, epochs), "last": val[-1], "best": val[bi],
                     "best_epoch": int(m[bi]["epoch"]), "organs_last": organ(m[-1])})
    csv_path = os.path.join(REPO, "experiments", exp, "cv_folds.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["fold", "epochs_done", "finished", "val_dice_last", "val_dice_best", "best_epoch"] + NAMES)
        for r in rows:
            w.writerow([r["fold"], r["epochs_done"], r["done"], f"{r['last']:.4f}", f"{r['best']:.4f}", r["best_epoch"]]
                       + [f"{x:.4f}" for x in r["organs_last"]])

    def ms(xs):
        xs = [x for x in xs if not math.isnan(x)]
        if not xs:
            return float("nan"), float("nan")
        mu = sum(xs) / len(xs)
        return mu, (math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float("nan"))

    done = [r for r in rows if r["done"]]
    lines += [f"# 5-fold cross-validation, Swin UNETR + atlas prior ({epochs} epochs)", "",
              f"Folds finished: {len(done)} of {len(folds)}; folds with any metrics: {len(rows)}. "
              "Primary: validation Dice at the last epoch (no epoch selection). Best-epoch is optimistic.", "",
              "| fold | epochs done | last-epoch Dice | best-epoch Dice (epoch) |", "|---|---|---|---|"]
    lines += [f"| {r['fold']} | {r['epochs_done']}{'' if r['done'] else ' (running/incomplete)'} | {r['last']:.4f} | "
              f"{r['best']:.4f} ({r['best_epoch']}) |" for r in rows]
    if done:
        mu, sd = ms([r["last"] for r in done])
        mub, sdb = ms([r["best"] for r in done])
        lines += ["", f"**Mean over {len(done)} finished fold(s): last-epoch {mu:.4f} +- {sd:.4f} (sd over folds); "
                      f"best-epoch {mub:.4f} +- {sdb:.4f}.**", "",
                  "| organ | last-epoch Dice, mean +- sd over folds |", "|---|---|"]
        for j, n in enumerate(NAMES):
            m_, s_ = ms([r["organs_last"][j] for r in done])
            lines.append(f"| {n} | {m_:.3f} +- {s_:.3f} |")
    path = os.path.join(REPO, "experiments", exp, "cv_summary.md")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    say(log_path, f"[summary] {len(done)}/{len(folds)} folds finished -> {path}")


def train_fold(args, gpu, fold, log_path):
    exp, epochs = args.experiment, args.epochs
    name = f"cv_fold{fold}_e{epochs}"
    chunk = os.path.join("experiments", exp, f"fold_{fold}_chunk.json")
    out = os.path.join(REPO, "experiments", exp, "logs")
    os.makedirs(out, exist_ok=True)
    for attempt in range(1, args.retries + 2):
        if finished(exp, fold, epochs):
            return True
        d = run_dir(exp, fold, epochs)
        if os.path.exists(d) and not os.path.exists(os.path.join(d, "checkpoint_last.pth")):
            os.replace(d, d + f"_failed_{time.strftime('%Y%m%d_%H%M%S')}")  # keep it for inspection; run.py refuses such a dir
        cmd = [sys.executable, "run.py", "--mode", "chunk", "--chunk-file", chunk, "--epochs", str(epochs), "--experiment", exp,
               "--run-name", name, "--stages", "library,train,report", "--num-workers", str(args.num_workers),
               f"--train-args={' '.join(RECIPE + args.train_extra.split())}"]
        say(log_path, f"[fold {fold}] GPU {gpu}: attempt {attempt}: {name}")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONIOENCODING": "utf-8"}
        with open(os.path.join(out, f"fold{fold}.out"), "a", encoding="utf-8") as f:
            code = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
        say(log_path, f"[fold {fold}] exited with code {code}")
        if finished(exp, fold, epochs):
            return True
        time.sleep(60)
    say(log_path, f"[fold {fold}] GAVE UP after {args.retries + 1} attempts; see experiments/{exp}/logs/fold{fold}.out")
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--experiment", default="exp_cv5")
    p.add_argument("--gpus", default=None, help="comma-separated GPU ids; default: all that nvidia-smi lists")
    p.add_argument("--reg-shards", type=int, default=16, help="parallel registration processes")
    p.add_argument("--num-workers", type=int, default=8, help="data-loader workers per training process")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--train-extra", default="", help="extra trainer flags, one string, e.g. '--micro-batch 2 --accum 1'")
    p.add_argument("--skip-register", action="store_true")
    p.add_argument("--summarize", action="store_true", help="only write the summary from the folds' metrics so far")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    exp_dir = os.path.join(REPO, "experiments", args.experiment)
    os.makedirs(exp_dir, exist_ok=True)
    log_path = os.path.join(exp_dir, "cv_queue.log")
    folds = list(range(args.k))
    if args.summarize:
        return summarize(args.experiment, args.epochs, folds, log_path)
    gpus = args.gpus.split(",") if args.gpus else detect_gpus()
    say(log_path, f"=== run_cv: {args.k} folds x {args.epochs} epochs, experiment {args.experiment}, GPUs {gpus} ===")
    if not gpus:
        raise SystemExit("no GPU found (nvidia-smi -L lists none); pass --gpus to override")
    if args.dry_run:
        say(log_path, "dry run: register (if needed), make folds, then train folds on the GPUs above")
        return

    if not args.skip_register:
        say(log_path, "[register] run.py --mode full --stages library,register (resumable; cached cases are skipped)")
        code = subprocess.run([sys.executable, "run.py", "--mode", "full", "--stages", "library,register", "--epochs", "1",
                               "--experiment", args.experiment, "--run-name", "cv_registration",
                               "--reg-shards", str(args.reg_shards)], cwd=REPO,
                              stdout=open(os.path.join(exp_dir, "cv_register.out"), "a"), stderr=subprocess.STDOUT).returncode
        say(log_path, f"[register] exited with code {code}")
    code = subprocess.run([sys.executable, "scripts/make_folds.py", "--k", str(args.k), "--out", exp_dir, "--require-cache"],
                          cwd=REPO, capture_output=True, text=True)
    say(log_path, "[folds] " + (code.stdout.strip().replace("\n", " | ") or code.stderr.strip()[-300:]))
    if code.returncode != 0:
        raise SystemExit("could not make the folds")

    todo = [k for k in folds if not finished(args.experiment, k, args.epochs)]
    q, results = list(todo), {}

    def worker(gpu):
        while True:
            with lock:
                if not q:
                    return
                fold = q.pop(0)
            try:
                results[fold] = train_fold(args, gpu, fold, log_path)
            except Exception as e:  # a bug in one worker must not silently stop it
                results[fold] = False
                say(log_path, f"[fold {fold}] ERROR in the runner: {e!r}")
            summarize(args.experiment, args.epochs, folds, log_path)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    [t.start() for t in threads]
    [t.join() for t in threads]
    summarize(args.experiment, args.epochs, folds, log_path)
    bad = [k for k, ok in results.items() if not ok]
    say(log_path, f"=== run_cv finished; failed folds: {bad or 'none'} ===")


if __name__ == "__main__":
    main()
