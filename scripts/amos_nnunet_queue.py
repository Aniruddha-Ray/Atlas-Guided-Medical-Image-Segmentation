"""
AMOS-only atlas-prior nnU-Net study, unattended after the registration of every non-atlas AMOS case has finished.

  1. wait   until scripts/amos_register_all.sh has finished ("ALL SHARDS DONE" in logs/amos_reg_shard0.out);
            stops (does not drop cases silently) if any training or validation case has no cached registration
  2. build  scripts/nnunet_make_datasets.py  -> Dataset073_AMOSPrior (CT + 25 prior channels), Dataset074_AMOSImage (CT only)
            nnU-Net planning for the prior dataset (3d_fullres, 4 GB GPU target), the same plans moved to the image-only
            dataset, preprocessing of both. Both arms therefore train with identical plans, patches and folds.
  3. train  in this order, each resumable (nnUNetv2_train --c) and skipped when finished:
              image-only  fold 2 (25 training cases)     prior fold 2     <- the decision experiment
              image-only  fold 0 (all training cases)    prior fold 0
            --epochs epochs each (nnU-Net epoch = 250 iterations, whatever the number of cases); every run ends with
            nnU-Net's own validation on the same 50 cases and writes summary.json (foreground mean Dice).

    python scripts/amos_nnunet_queue.py            # the real thing (run in the background; safe to rerun after a stop)
    python scripts/amos_nnunet_queue.py --dry-run  # print the plan and what is already done
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from nnunet_ext.nn import DEFAULT_ROOT, executable, nnunet_env  # noqa: E402

CHUNK = os.path.join(REPO, "experiments", "exp_amos", "amos_chunk.json")
AMOS_DIR = os.path.join(os.path.dirname(REPO), "Dataset061_AMOS")
CACHE = os.path.join(REPO, "cache_amos")
LOG = os.path.join(REPO, "logs", "amos_nnunet_queue.log")
PAUSE_EXIT_CODE = 75  # nnunet_ext/nnUNetTrainerAtlasPrior.py
PRIOR_ID, IMAGE_ID, PLANS, CONF = 73, 74, "nnUNetPlans_4G", "3d_fullres"
PRIOR_DS, IMAGE_DS = f"Dataset{PRIOR_ID:03d}_AMOSPrior", f"Dataset{IMAGE_ID:03d}_AMOSImage"
SUBSETS = "10,25,50,100,150"  # fold 1.. = nested subsets of the training cases: 10, 25, 50, 100, 150


def say(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [amos_queue] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def missing_registrations() -> list:
    chunk = json.load(open(CHUNK, encoding="utf-8"))
    return [n for n in chunk["train"] + chunk["val"]
            if not os.path.exists(os.path.join(CACHE, "warped_labels", "tps", f"{n}.nii.gz"))]


def registration_finished() -> bool:
    path = os.path.join(REPO, "logs", "amos_reg_shard0.out")
    return os.path.exists(path) and "ALL SHARDS DONE" in open(path, encoding="utf-8", errors="replace").read()


def run(cmd: list, env: dict, log_name: str) -> int:
    say("$ " + " ".join(cmd))
    with open(os.path.join(REPO, "logs", log_name), "a", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def nn(env, command, *args, log_name="amos_nnunet_build.out") -> int:
    return run([executable(command), *args], env, log_name)


def results_dir(ds, trainer, fold, root):
    return os.path.join(root, "nnUNet_results", ds, f"{trainer}__{PLANS}__{CONF}", f"fold_{fold}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--epochs", type=int, choices=(50, 100, 250), default=100)
    p.add_argument("--da-workers", type=int, default=6, help="nnU-Net augmentation workers (12 runs the 4 GB GPU out of memory)")
    p.add_argument("--prep-workers", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    env = nnunet_env(a.root)
    env["nnUNet_n_proc_DA"] = str(a.da_workers)
    py = sys.executable
    # image fold 2 was trained (and finished) with the stock trainer; the same network and schedule, only without per-epoch checkpoints
    plan = [("image", IMAGE_ID, IMAGE_DS, f"nnUNetTrainer_{a.epochs}epochs", 2), ("prior", PRIOR_ID, PRIOR_DS, f"nnUNetTrainerAtlasPrior_{a.epochs}epochs", 2),
            ("image", IMAGE_ID, IMAGE_DS, f"nnUNetTrainerResumable_{a.epochs}epochs", 0), ("prior", PRIOR_ID, PRIOR_DS, f"nnUNetTrainerAtlasPrior_{a.epochs}epochs", 0)]
    if a.dry_run:
        say(f"registration finished: {registration_finished()}; missing registrations: {len(missing_registrations())}")
        for arm, did, ds, tr, fold in plan:
            done = os.path.exists(os.path.join(results_dir(ds, tr, fold, a.root), "checkpoint_final.pth"))
            say(f"  train {arm:5s} fold {fold} trainer {tr}: {'done' if done else 'to do'}")
        return

    say(f"=== AMOS nnU-Net queue: {a.epochs} epochs, root {a.root} ===")
    pause = os.path.join(a.root, "PAUSE")
    if os.path.exists(pause):
        os.remove(pause)
        say("removed the PAUSE file: this start is the resume")
    while not registration_finished():
        time.sleep(300)
    miss = missing_registrations()
    if miss:
        say(f"STOP: {len(miss)} cases have no registration (e.g. {miss[:5]}); see logs/amos_reg_shard*.out and "
            f"cache_amos/registrations/tps/_failures_*.json. Fix or drop them deliberately, then rerun.")
        sys.exit(1)
    say("all registrations present")
    run([py, "nnunet_ext/install.py"], env, "amos_nnunet_build.out")  # the trainer must sit inside the nnunetv2 package

    if not os.path.exists(os.path.join(a.root, "nnUNet_preprocessed", IMAGE_DS, "_preprocess_done")):
        steps = [
            ([py, "-W", "ignore", "scripts/nnunet_make_datasets.py", "--nnunet-root", a.root, "--chunk-file", CHUNK, "--name", "AMOS",
              "--prior-id", str(PRIOR_ID), "--image-id", str(IMAGE_ID), "--data-dir", AMOS_DIR, "--cache-dir", CACHE,
              "--subsets", SUBSETS, "--workers", str(a.prep_workers)], "make datasets"),
        ]
        for cmd, what in steps:
            if run(cmd, env, "amos_nnunet_build.out") != 0:
                say(f"STOP: {what} failed; see logs/amos_nnunet_build.out"); sys.exit(1)
        build = [
            ("plan_and_preprocess", "-d", str(PRIOR_ID), "-c", CONF, "-gpu_memory_target", "4", "-overwrite_plans_name", PLANS,
             "-np", str(a.prep_workers), "-npfp", str(a.prep_workers), "--verify_dataset_integrity"),
            ("extract_fingerprint", "-d", str(IMAGE_ID), "-np", str(a.prep_workers)),
            ("move_plans_between_datasets", "-s", str(PRIOR_ID), "-t", str(IMAGE_ID), "-sp", PLANS, "-tp", PLANS),
        ]
        for cmd in build:
            if nn(env, *cmd) != 0:
                say(f"STOP: nnUNetv2_{cmd[0]} failed; see logs/amos_nnunet_build.out"); sys.exit(1)
        # nnU-Net looks for dataset.json in the preprocessed folder of the image-only dataset
        import shutil
        shutil.copyfile(os.path.join(a.root, "nnUNet_raw", IMAGE_DS, "dataset.json"),
                        os.path.join(a.root, "nnUNet_preprocessed", IMAGE_DS, "dataset.json"))
        if nn(env, "preprocess", "-d", str(IMAGE_ID), "-plans_name", PLANS, "-c", CONF, "-np", str(a.prep_workers)) != 0:
            say("STOP: preprocessing the image-only dataset failed"); sys.exit(1)
        open(os.path.join(a.root, "nnUNet_preprocessed", IMAGE_DS, "_preprocess_done"), "w").close()
        say("datasets built and preprocessed")

    for arm, did, ds, tr, fold in plan:
        rd = results_dir(ds, tr, fold, a.root)
        if os.path.exists(os.path.join(rd, "checkpoint_final.pth")):
            say(f"{arm} fold {fold}: already finished, skipping")
            continue
        cmd = ["train", str(did), CONF, str(fold), "-tr", tr, "-p", PLANS]
        if os.path.exists(os.path.join(rd, "checkpoint_latest.pth")):
            cmd.append("--c")
            say(f"{arm} fold {fold}: resuming")
        t0 = time.time()
        code = nn(env, *cmd, log_name=f"amos_nnunet_train_{arm}_fold{fold}.out")
        say(f"{arm} fold {fold}: exit {code} after {(time.time() - t0) / 3600:.1f} h")
        if code == PAUSE_EXIT_CODE:
            say(f"PAUSED after the current epoch ({arm} fold {fold}); rerun this script to resume from that epoch"); sys.exit(0)
        if code != 0:
            say("STOP: a training run failed; see its log under logs/. Rerun this script to resume.")
            sys.exit(1)
    say("=== queue finished ===")


if __name__ == "__main__":
    main()
