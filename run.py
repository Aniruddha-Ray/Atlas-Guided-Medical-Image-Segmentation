"""
One entry point for the whole pipeline: choose cases, register the atlases onto them, validate the registration and
the priors, train, and report.

    select    pick a chunk of cases (scripts/select_chunk.py) or use --chunk-file; --mode full uses every case
    library   build the atlas library if it is missing or does not match atlas/atlas_selection.json
    register  register the atlases onto every case that has no cached registration (resumable)
    validate  scripts/validate_priors.py: registration quality and prior-insertion checks; stops the run if an
              invariant fails (unless --ignore-validation-failures)
    train     scripts/train_prototype_v2.py for --epochs epochs (resumes from checkpoint_last.pth after an interruption)
    report    experiments/<experiment>/<run>_report.md: model against the atlas vote, per organ

Every stage is idempotent, so a stopped run is continued by running the same command again.

    python run.py --mode chunk --train-size 48 --val-size 24 --epochs 200      # laptop-sized run
    python run.py --mode full --epochs 200 --reg-shards 8                      # whole dataset, e.g. on a server
    python run.py --mode chunk --dry-run                                       # show the plan and time estimate only
    python run.py --stages validate,report --run-name my_run                   # just some stages

Extra trainer flags go through --train-args, for example --train-args "--hu-weight 0.03 --edge-weight 50".
Time estimates use the constants below, measured or calibrated on the laptop RTX 3050 with a 48 + 24 case run;
they scale roughly linearly, so expect larger chunks to be less exact.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STAGES = ["select", "library", "register", "validate", "train", "report"]

REG_S_PER_TARGET = 180.0  # measured on the laptop RTX 3050: 175 s (60 targets) and 193 s (12 targets), 10 atlases
TRAIN_S_PER_CASE = 1.5  # calibrated on one run (48 train + 24 val: 91 s median per epoch)
VAL_S_PER_CASE = 0.8  # same calibration
CACHE_S_PER_CASE = 5.0  # one-time sample cache, from the first epoch of that run (435 s against 91 s)
VALIDATE_S_PER_CASE = 25.0  # measured: 72 cases in 30 min
PHYSICS_TRAIN_FACTOR = 1.56  # measured: training step 0.88 -> 1.37 s with both physics losses on

TRAIN_SHARE = {"amos": 0.40, "img": 0.04}  # of the non-atlas training pool (amos 39 %, img 4 %, s 56 %)
VAL_SHARE = {"amos": 0.43, "img": 0.05}  # of the validation split (amos 58, s 71, img 7 of 136)


def proportional_quota(n: int, shares: dict) -> dict:
    """Per-source case counts for a chunk of n cases: amos and img from `shares`, s the rest; at least 1 img."""
    amos = int(round(n * shares["amos"]))
    img = max(1, int(round(n * shares["img"])))
    s = n - amos - img
    if s < 1:
        raise ValueError(f"chunk of {n} cases is too small for the source quotas")
    return {"amos": amos, "img": img, "s": s}


def quota_string(q: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in q.items())


def estimate(n_train: int, n_val: int, to_register: int, epochs: int, physics: bool = False, workers_note: bool = False) -> dict:
    """Hours per phase. All inputs are counts; see the module docstring for how reliable each constant is."""
    train_factor = PHYSICS_TRAIN_FACTOR if physics else 1.0
    per_epoch = n_train * TRAIN_S_PER_CASE * train_factor + n_val * VAL_S_PER_CASE * 1.2  # no-prior pass every 5th epoch
    phases = {
        "registration": to_register * REG_S_PER_TARGET / 3600,
        "validation": (n_train + n_val) * VALIDATE_S_PER_CASE / 3600,
        "sample cache": (n_train + n_val) * CACHE_S_PER_CASE / 3600,
        "training": epochs * per_epoch / 3600,
    }
    phases["total"] = sum(phases.values())
    phases["seconds_per_epoch"] = per_epoch
    return phases


def fmt_hours(h: float) -> str:
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


class Pipeline:
    def __init__(self, args):
        self.a = args
        self.exp_dir = os.path.join(REPO, "experiments", args.experiment)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.run_dir = os.path.join(self.exp_dir, args.run_name)
        self.chunk_path = args.chunk_file or os.path.join(self.exp_dir, f"{args.run_name}_chunk.json")
        self.log_path = os.path.join(self.exp_dir, f"{args.run_name}_pipeline.log")
        self.cache_dir = os.path.join(REPO, "cache")
        self.logf = open(self.log_path, "a", encoding="utf-8")

    def log(self, msg=""):
        print(msg, flush=True)
        self.logf.write(msg + "\n")
        self.logf.flush()

    def run(self, cmd: list, check: bool = True) -> int:
        self.log("$ " + " ".join(shlex.quote(c) for c in cmd))
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            if "Warning" in line or "warnings.warn(" in line or "warn_deprecated(" in line:
                continue  # library deprecation noise; real errors still come through
            self.log(line.rstrip())
        code = proc.wait()
        if check and code != 0:
            raise SystemExit(f"stage failed (exit code {code}): {' '.join(cmd[:3])} ...")
        return code

    def stacks(self, names):
        return [n for n in names if os.path.exists(os.path.join(self.cache_dir, "warped_labels", "tps", f"{n}.nii.gz"))]

    def chunk(self) -> dict:
        return json.load(open(self.chunk_path, encoding="utf-8"))

    # ---- stages -------------------------------------------------------------------------------------------------
    def select(self):
        a = self.a
        if a.chunk_file:
            self.log(f"[select] using the given chunk file {a.chunk_file}")
            return
        if a.mode == "full":
            self.log("[select] full mode: every non-atlas training target and the whole validation split")
            return
        tq, vq = proportional_quota(a.train_size, TRAIN_SHARE), proportional_quota(a.val_size, VAL_SHARE)
        self.log(f"[select] train {a.train_size} ({quota_string(tq)}), val {a.val_size} ({quota_string(vq)}), seed {a.seed}")
        tmp_exp = f"_chunk_{a.run_name}"
        self.run([PY, "scripts/select_chunk.py", "--experiment", tmp_exp, "--train-quota", quota_string(tq),
                  "--val-quota", quota_string(vq), "--seed", str(a.seed)])
        src = os.path.join(REPO, "experiments", tmp_exp, "chunk.json")
        os.replace(src, self.chunk_path)
        os.rmdir(os.path.dirname(src))
        self.log(f"[select] wrote {self.chunk_path}")

    def library(self):
        from atlas.reliable_features import load_atlas_library
        from data.splits import atlas_names, case_name

        path = os.path.join(self.cache_dir, "atlas_library", "atlas_library.npz")
        want = sorted(atlas_names())
        have = sorted(case_name(e["image"]) for e in load_atlas_library(path)) if os.path.exists(path) else None
        if have == want:
            self.log(f"[library] up to date ({len(want)} atlases)")
            return
        self.log(f"[library] {'missing' if have is None else 'stale'}; building for {want}")
        self.run([PY, "scripts/build_atlas_library.py"])

    def register(self):
        a = self.a
        common = ["--eval-gt", "--continue-on-error"]
        jobs = []
        if a.mode == "full" and not a.chunk_file:
            for split in ("train", "val"):
                jobs.append((split, None))
        else:
            ch = self.chunk()
            for split in ("train", "val"):
                todo = [n for n in ch[split] if n not in set(self.stacks(ch[split]))]
                self.log(f"[register] {split}: {len(ch[split]) - len(todo)} cached, {len(todo)} to register")
                if todo:
                    jobs.append((split, todo))
        for split, names in jobs:
            base = [PY, "scripts/register_targets.py", "--split", split] + common
            if names:
                base += ["--cases"] + names
            if a.reg_shards > 1:
                procs = []
                for i in range(a.reg_shards):
                    cmd = base + ["--num-shards", str(a.reg_shards), "--shard-index", str(i)]
                    self.log("$ " + " ".join(shlex.quote(c) for c in cmd) + f"   (shard {i}, log logs/reg_{split}_{i}.log)")
                    os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)
                    procs.append(subprocess.Popen(cmd, cwd=REPO, stdout=open(os.path.join(REPO, "logs", f"reg_{split}_{i}.log"), "w"),
                                                  stderr=subprocess.STDOUT))
                codes = [p.wait() for p in procs]
                if any(codes):
                    raise SystemExit(f"registration shard failed: exit codes {codes}")
            else:
                self.run(base)

    def validate(self):
        a = self.a
        prefix = os.path.join(self.exp_dir, f"{a.run_name}_validation")
        cmd = [PY, "scripts/validate_priors.py", "--out-prefix", prefix]
        if a.mode == "full" and not a.chunk_file:
            from data.splits import case_name, split_with_atlases_removed
            tr, va = split_with_atlases_removed()
            names = [case_name(c["image"]) for c in tr + va]
            cmd += ["--cases"] + names
        else:
            cmd += ["--chunk", self.chunk_path]
        code = self.run(cmd, check=False)
        if code == 2 and not a.ignore_validation_failures:
            raise SystemExit(f"prior validation failed (see {prefix}.md); fix it or pass --ignore-validation-failures")
        if code not in (0, 2):
            raise SystemExit(f"validation script crashed (exit code {code})")

    def train(self):
        a = self.a
        summary = os.path.join(self.run_dir, "summary.json")
        if os.path.exists(summary):
            self.log(f"[train] already finished ({summary}); skipping")
            return
        cmd = [PY, "scripts/train_prototype_v2.py", "--experiment", a.experiment, "--run-name", a.run_name,
               "--epochs", str(a.epochs), "--num-workers", str(a.num_workers), "--seed", str(a.seed)]
        if not a.no_amp:
            cmd.append("--amp")
        cmd += ["--full-data"] if (a.mode == "full" and not a.chunk_file) else ["--chunk", self.chunk_path]
        last = os.path.join(self.run_dir, "checkpoint_last.pth")
        if os.path.exists(last):
            self.log(f"[train] resuming from {last}")
            cmd += ["--resume", last]
        elif os.path.exists(self.run_dir):
            raise SystemExit(f"run directory exists without a checkpoint: {self.run_dir}; choose another --run-name")
        cmd += shlex.split(a.train_args)
        self.run(cmd)

    def report(self):
        a = self.a
        metrics, summary_p = os.path.join(self.run_dir, "metrics.csv"), os.path.join(self.run_dir, "summary.json")
        if not os.path.exists(metrics):
            self.log("[report] no metrics.csv yet")
            return
        rows = list(csv.DictReader(open(metrics, encoding="utf-8")))
        val = [float(r["val_dice"]) for r in rows]
        best_i = max(range(len(val)), key=val.__getitem__)
        organ_cols = [f"o{k}" for k in range(1, 13)]
        best_per = [float(rows[best_i][c]) if rows[best_i][c] != "" else float("nan") for c in organ_cols]
        no_prior = [(int(r["epoch"]), float(r["val_dice_no_prior"])) for r in rows if r.get("val_dice_no_prior")]
        ref = None
        if os.path.exists(summary_p):
            ref = json.load(open(summary_p, encoding="utf-8")).get("atlas_only_reference")
        vote_org = None
        vp = os.path.join(self.exp_dir, f"{a.run_name}_validation.json")
        if os.path.exists(vp):
            s = json.load(open(vp, encoding="utf-8"))["summary"]
            vote_org = (s.get("val") or s["all"])["per_organ_vote_dice"]
        L = [f"# Run report: {a.experiment}/{a.run_name}", "",
             f"Epochs completed: {len(rows)} of {a.epochs}"
             + ("" if os.path.exists(summary_p) else " (still running or interrupted)"), ""]
        L += ["| Measure | Value |", "|---|---|",
              f"| Best validation Dice | **{val[best_i]:.4f}** (epoch {rows[best_i]['epoch']}) |",
              f"| Last validation Dice | {val[-1]:.4f} |"]
        if ref is not None:
            L.append(f"| Atlas vote alone, same validation cases (96^3) | {ref:.4f} |")
            L.append(f"| Model minus atlas vote (best epoch) | {val[best_i] - ref:+.4f} |")
        if no_prior:
            L.append(f"| Model with the prior removed at evaluation (epoch {no_prior[-1][0]}) | {no_prior[-1][1]:.4f} |")
        L += ["", "Per organ at the best epoch (1 to 12):", "", "| | " + " | ".join(str(i) for i in range(1, 13)) + " |",
              "|---" * 13 + "|", "| model | " + " | ".join("" if math.isnan(v) else f"{v:.2f}" for v in best_per) + " |"]
        if vote_org:
            L.append("| atlas vote | " + " | ".join(f"{v:.2f}" for v in vote_org) + " |")
        L += ["", "Training curve (validation Dice): " + ", ".join(f"e{rows[i]['epoch']} {val[i]:.3f}"
                                                                  for i in sorted({0, len(rows) // 4, len(rows) // 2,
                                                                                   3 * len(rows) // 4, len(rows) - 1}))]
        L.append(f"\nRegistration and prior validation: `{a.run_name}_validation.md`.")
        path = os.path.join(self.exp_dir, f"{a.run_name}_report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        self.log("\n".join(L))
        self.log(f"[report] wrote {path}")

    # ---- plan ---------------------------------------------------------------------------------------------------
    def plan(self):
        a = self.a
        physics = "--hu-weight" in a.train_args or "--edge-weight" in a.train_args
        if a.mode == "full" and not a.chunk_file:
            from data.splits import case_name, split_with_atlases_removed
            tr, va = split_with_atlases_removed()
            names_t, names_v = [case_name(c["image"]) for c in tr], [case_name(c["image"]) for c in va]
        else:
            ch = self.chunk()
            names_t, names_v = ch["train"], ch["val"]
        todo = len(names_t) + len(names_v) - len(self.stacks(names_t)) - len(self.stacks(names_v))
        est = estimate(len(names_t), len(names_v), todo, a.epochs, physics)
        shards = max(a.reg_shards, 1)
        self.log(f"\nPlan: {a.mode} mode, {len(names_t)} train + {len(names_v)} val cases, {todo} still to register, "
                 f"{a.epochs} epochs{' (physics losses on)' if physics else ''}")
        for k in ("registration", "validation", "sample cache", "training"):
            note = f" (split over {shards} shards, if they run in parallel)" if k == "registration" and shards > 1 else ""
            self.log(f"  {k:13s} {fmt_hours(est[k] / (shards if k == 'registration' else 1)):>8s}{note}")
        self.log(f"  {'total':13s} {fmt_hours(est['total']):>8s}   ({est['seconds_per_epoch']:.0f} s per epoch; training constants are "
                 f"calibrated on a 48 + 24 case laptop run)")
        return est


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["chunk", "full"], default="chunk")
    p.add_argument("--train-size", type=int, default=48)
    p.add_argument("--val-size", type=int, default=24)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--experiment", default="exp_04_pipeline")
    p.add_argument("--run-name", default=None, help="default: chunk_<train>_<val>_e<epochs> or full_e<epochs>")
    p.add_argument("--chunk-file", default=None, help="use this chunk.json instead of selecting one")
    p.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {STAGES}")
    p.add_argument("--reg-shards", type=int, default=1, help="parallel registration processes (one per GPU/CPU slot; 1 on a laptop)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-args", default="", help="extra flags for scripts/train_prototype_v2.py, as one string")
    p.add_argument("--ignore-validation-failures", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="select cases and print the plan and time estimate, run nothing else")
    a = p.parse_args(argv)
    if a.run_name is None:
        a.run_name = f"chunk_{a.train_size}_{a.val_size}_e{a.epochs}" if a.mode == "chunk" else f"full_e{a.epochs}"
    bad = [s for s in a.stages.split(",") if s not in STAGES]
    if bad:
        p.error(f"unknown stage(s) {bad}; choose from {STAGES}")
    return a


def main(argv=None):
    a = parse_args(argv)
    pipe = Pipeline(a)
    stages = a.stages.split(",")
    pipe.log(f"=== run.py {time.strftime('%Y-%m-%d %H:%M:%S')} | {a.experiment}/{a.run_name} | stages {stages} ===")
    if "select" in stages or a.dry_run:
        pipe.select()
    pipe.plan()
    if a.dry_run:
        return
    t0 = time.time()
    for stage in STAGES:
        if stage in stages and stage != "select":
            pipe.log(f"\n=== stage: {stage} ({time.strftime('%H:%M:%S')}) ===")
            getattr(pipe, stage)()
    pipe.log(f"\n=== done in {fmt_hours((time.time() - t0) / 3600)} ===")


if __name__ == "__main__":
    main()
