"""
The cascade study, unattended: train on AMOS, test on BTCV, Dice and HD95.

  wait       until the running AMOS prior run (Dataset073 fold 2) has finished and no nnU-Net training is running
  split      scripts/cascade_split.py: 30 AMOS atlases, 42 stage-1 cases, 200 stage-2 cases, 30 BTCV test cases
  btcv_reg   CT-to-CT registration of the 30 BTCV cases to the 10 first-study atlases (cache_btcv/), as for AMOS
  s1_build   scripts/cascade_make_datasets.py stage1 -> Dataset080_CascadeS1
  s1_prep    Dataset073's plans moved to it, preprocessing
  s1_train   stage 1: CT + CT-to-CT prior, 42 cases, 50 epochs (nnUNetTrainerAtlasPrior_50epochs)
  s1_pred    stage-1 prediction of the 200 stage-2 cases and the 30 BTCV cases (no test-time mirroring, for all alike)
  s1_eval    stage-1 Dice / HD95 on BTCV
  priors     scripts/cascade_label_priors.py: arm B and arm C prior channels (label-to-label registration of 30 atlases)
  s2_build   scripts/cascade_make_datasets.py stage2 -> Dataset081 (A), 082 (B), 083 (C)
  s2_prep    Dataset073's plans moved to each, preprocessing
  train      fold 0 (200 AMOS / 30 BTCV), 100 epochs each: C atlas cascade, A image-only, B plain cascade
  eval       Dice / HD95 of A, B, C on BTCV -> experiments/exp_cascade/results_btcv.{csv,md}

Every step is skipped when done (marker files in <nnunet root>/cascade/), training resumes from its last epoch, so the
script can simply be rerun. Pause: python scripts/amos_pause.py pause (a run stops after its current epoch; between
steps the queue stops before the next one), resume: python scripts/amos_pause.py resume.

    python scripts/cascade_queue.py            # the real thing (in the background)
    python scripts/cascade_queue.py --dry-run  # what is done and what is next
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from nnunet_ext.nn import DEFAULT_ROOT, executable, nnunet_env  # noqa: E402

LOG = os.path.join(REPO, "logs", "cascade_queue.log")
SPLIT = os.path.join(REPO, "experiments", "exp_cascade", "cascade_split.json")
BTCV_DIR = os.path.join(os.path.dirname(REPO), "Dataset062_Synapse")
PLANS, CONF, SOURCE_ID = "nnUNetPlans_4G", "3d_fullres", 73
PAUSE_EXIT_CODE = 75  # nnunet_ext/nnUNetTrainerAtlasPrior.py
S1 = (80, "Dataset080_CascadeS1", "nnUNetTrainerAtlasPrior_50epochs")
ARMS = [("C", 83, "Dataset083_CascadeAtlas", "nnUNetTrainerAtlasPrior_100epochs"),
        ("A", 81, "Dataset081_CascadeImage", "nnUNetTrainerResumable_100epochs"),
        ("B", 82, "Dataset082_CascadePlain", "nnUNetTrainerAtlasPrior_100epochs")]
CURRENT_RUN = ("Dataset073_AMOSPrior", "nnUNetTrainerAtlasPrior_100epochs", 2)


def say(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [cascade] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


class Queue:
    def __init__(self, root: str, da_workers: int, prep_workers: int, reg_workers: int):
        self.root, self.prep_workers, self.reg_workers = root, prep_workers, reg_workers
        self.env = nnunet_env(root)
        self.env["nnUNet_n_proc_DA"] = str(da_workers)
        self.py = sys.executable
        self.state = os.path.join(root, "cascade")
        os.makedirs(self.state, exist_ok=True)

    # ---- helpers
    def done(self, step):
        return os.path.exists(os.path.join(self.state, f"_done_{step}"))

    def mark(self, step):
        open(os.path.join(self.state, f"_done_{step}"), "w").close()
        say(f"step {step}: done")

    def check_pause(self):
        if os.path.exists(os.path.join(self.root, "PAUSE")):
            say("PAUSED between steps; python scripts/amos_pause.py resume continues from here")
            sys.exit(0)

    def run(self, cmd, log_name, wait=True):
        say("$ " + " ".join(str(c) for c in cmd))
        f = open(os.path.join(REPO, "logs", log_name), "a", encoding="utf-8")
        proc = subprocess.Popen([str(c) for c in cmd], cwd=REPO, env=self.env, stdout=f, stderr=subprocess.STDOUT)
        return proc.wait() if wait else proc

    def must(self, code, what):
        if code != 0:
            say(f"STOP: {what} failed (exit {code}); see its log under logs/. Rerun this script to resume.")
            sys.exit(1)

    def nn(self, *args, log_name="cascade_build.out"):
        return self.run([executable(args[0]), *args[1:]], log_name)

    def results(self, ds, tr, fold=0):
        return os.path.join(self.root, "nnUNet_results", ds, f"{tr}__{PLANS}__{CONF}", f"fold_{fold}")

    def raw(self, ds):
        return os.path.join(self.root, "nnUNet_raw", ds)

    @staticmethod
    def trainer_running() -> bool:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -notlike 'powershell*' -and $_.CommandLine -match 'nnUNetv2_train' }).Count"],
                             capture_output=True, text=True).stdout.strip()
        return out not in ("", "0")

    # ---- steps
    def wait_current(self):
        ds, tr, fold = CURRENT_RUN
        summary = os.path.join(self.results(ds, tr, fold), "validation", "summary.json")
        if not os.path.exists(summary) or self.trainer_running():
            say(f"waiting for the running {ds} fold {fold} run to finish")
        while not os.path.exists(summary) or self.trainer_running():
            time.sleep(300)

    def split(self):
        self.must(self.run([self.py, "-W", "ignore", "scripts/cascade_split.py"], "cascade_build.out"), "cascade_split.py")

    def btcv_reg(self):
        base = [self.py, "-W", "ignore", "-u", "scripts/register_targets.py", "--data-dir", BTCV_DIR,
                "--atlas-library", os.path.join("cache_amos", "atlas_library", "atlas_library.npz"),
                "--selection", os.path.join("atlas", "atlas_selection_amos.json"), "--out-dir", "cache_btcv",
                "--eval-gt", "--continue-on-error"]
        n = 3  # each registration process holds 10 atlas CTs; more than 4 at once strains 16 GB of RAM
        procs = [self.run(base + ["--split", "train", "--num-shards", n, "--shard-index", i], f"cascade_btcv_reg{i}.out", wait=False)
                 for i in range(n)]
        procs.append(self.run(base + ["--split", "val"], "cascade_btcv_reg_val.out", wait=False))
        codes = [p.wait() for p in procs]
        split = json.load(open(SPLIT, encoding="utf-8"))
        miss = [c for c in split["test"] if not os.path.exists(os.path.join(REPO, "cache_btcv", "warped_labels", "tps", f"{c}.nii.gz"))]
        if miss:
            say(f"STOP: {len(miss)} BTCV cases not registered ({miss[:5]}), exit codes {codes}; see logs/cascade_btcv_reg*.out")
            sys.exit(1)

    def prep(self, did, ds):
        for cmd in (("extract_fingerprint", "-d", did, "-np", self.prep_workers),
                    ("move_plans_between_datasets", "-s", SOURCE_ID, "-t", did, "-sp", PLANS, "-tp", PLANS)):
            self.must(self.nn(*map(str, cmd)), f"nnUNetv2_{cmd[0]} for {ds}")
        shutil.copyfile(os.path.join(self.raw(ds), "dataset.json"),
                        os.path.join(self.root, "nnUNet_preprocessed", ds, "dataset.json"))
        self.must(self.nn("preprocess", "-d", str(did), "-plans_name", PLANS, "-c", CONF, "-np", str(self.prep_workers)),
                  f"preprocessing {ds}")

    def train(self, did, ds, tr, label, fold=0):
        rd = self.results(ds, tr, fold)
        if os.path.exists(os.path.join(rd, "checkpoint_final.pth")):
            return
        cmd = ["train", str(did), CONF, str(fold), "-tr", tr, "-p", PLANS]
        if os.path.exists(os.path.join(rd, "checkpoint_latest.pth")):
            cmd.append("--c")
            say(f"{label}: resuming")
        t0 = time.time()
        code = self.nn(*cmd, log_name=f"cascade_train_{label}.out")
        say(f"{label}: exit {code} after {(time.time() - t0) / 3600:.1f} h")
        if code == PAUSE_EXIT_CODE:
            say(f"PAUSED after the current epoch of {label}; python scripts/amos_pause.py resume continues from there")
            sys.exit(0)
        self.must(code, f"training {label}")

    def s1_pred(self):
        did, ds, tr = S1
        out = os.path.join(self.state, "stage1_pred")
        self.must(self.nn("predict", "-i", os.path.join(self.raw(ds), "imagesPred"), "-o", out, "-d", str(did), "-c", CONF,
                          "-f", "0", "-tr", tr, "-p", PLANS, "-chk", "checkpoint_final.pth", "--disable_tta",
                          "-npp", "2", "-nps", "2", "--continue_prediction", log_name="cascade_stage1_predict.out"),
                  "stage-1 prediction")
        split = json.load(open(SPLIT, encoding="utf-8"))
        miss = [c for c in split["stage2"] + split["test"] if not os.path.exists(os.path.join(out, f"{c}.nii.gz"))]
        if miss:
            say(f"STOP: {len(miss)} stage-1 predictions missing, e.g. {miss[:3]}")
            sys.exit(1)

    def evaluate(self, arms, out_name):
        split = json.load(open(SPLIT, encoding="utf-8"))
        cmd = [self.py, "-W", "ignore", "scripts/eval_seg.py", "--gt-dir", os.path.join(self.raw(S1[1]), "labelsTr"),
               "--out", os.path.join("experiments", "exp_cascade", out_name), "--cases", *split["test"]]
        for name, folder in arms:
            cmd += ["--arm", f"{name}={folder}"]
        self.must(self.run(cmd, "cascade_eval.out"), f"evaluation {out_name}")

    def go(self):
        say(f"=== cascade queue, root {self.root} ===")
        pause = os.path.join(self.root, "PAUSE")
        if os.path.exists(pause):
            os.remove(pause)
            say("removed the PAUSE file: this start is the resume")
        self.wait_current()
        run_py = lambda script, *args, log="cascade_build.out": self.must(  # noqa: E731
            self.run([self.py, "-W", "ignore", "-u", script, *args], log), script)
        steps = [
            ("split", lambda: self.split() if not os.path.exists(SPLIT) else None),
            ("btcv_reg", self.btcv_reg),
            ("s1_build", lambda: run_py("scripts/cascade_make_datasets.py", "stage1")),
            ("s1_prep", lambda: self.prep(S1[0], S1[1])),
            ("s1_train", lambda: self.train(*S1, label="stage1")),
            ("s1_pred", self.s1_pred),
            ("s1_eval", lambda: self.evaluate([("stage1", os.path.join(self.state, "stage1_pred"))], "results_stage1_btcv")),
            ("priors", lambda: run_py("scripts/cascade_label_priors.py", "--workers", str(self.reg_workers),
                                      log="cascade_label_priors.out")),
            ("s2_build", lambda: run_py("scripts/cascade_make_datasets.py", "stage2")),
            *[(f"s2_prep_{k}", (lambda d=d, s=s: self.prep(d, s))) for k, d, s, _ in ARMS],
            *[(f"train_{k}", (lambda d=d, s=s, t=t, k=k: self.train(d, s, t, label=f"arm{k}"))) for k, d, s, t in ARMS],
            ("eval", lambda: self.evaluate([(k, os.path.join(self.results(s, t), "validation")) for k, _, s, t in ARMS],
                                           "results_btcv")),
        ]
        for name, fn in steps:
            if self.done(name):
                continue
            self.check_pause()
            say(f"step {name}")
            fn()
            self.mark(name)
        say("=== cascade queue finished: experiments/exp_cascade/results_btcv.md ===")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--da-workers", type=int, default=6, help="nnU-Net augmentation workers (12 runs the 4 GB GPU out of memory)")
    p.add_argument("--prep-workers", type=int, default=3)
    p.add_argument("--reg-workers", type=int, default=8, help="parallel processes for the CPU registration steps")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    q = Queue(a.root, a.da_workers, a.prep_workers, a.reg_workers)
    if a.dry_run:
        markers = sorted(os.path.basename(m)[6:] for m in glob.glob(os.path.join(q.state, "_done_*")))
        say(f"done steps: {markers or 'none'}; split file: {os.path.exists(SPLIT)}; "
            f"current AMOS run finished: {os.path.exists(os.path.join(q.results(*CURRENT_RUN[:2], CURRENT_RUN[2]), 'validation', 'summary.json'))}")
        return
    q.go()


if __name__ == "__main__":
    main()
