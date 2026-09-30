"""
Trains / fine-tunes AtlasGuidedViTUNETR with a chosen anatomical-prior source.

Experiment 1 (default): start from the epoch-101 weights (best_model.pth)
and change ONLY the prior source -- atlas-derived P(v), D(v) from the cached
warped atlas labels instead of the case's own ground truth. Loss, optimizer,
learning rate, augmentation and metric are the notebook's, so any change in
Dice is attributable to the prior.

Leakage rules enforced here:
  - atlas cases are removed from the training targets (data/splits.py)
  - with --prior-source atlas, neither training nor validation priors touch
    the target's own label; the label is only the supervised-loss target and
    the evaluation reference

Every run gets its own directory, experiments/<experiment>/<run_name>/,
holding config.json, command.txt, git.txt (+ git_diff.patch), metrics.csv,
summary.json, training.log, checkpoint_best.pth and checkpoint_last.pth. An
existing run directory is never overwritten.

Examples:
    # local smoke test (a handful of cached cases, a few steps)
    python scripts/train.py --run-name smoke --limit-train 4 --limit-val 2 --epochs 1 --max-steps-per-epoch 4 --allow-missing --amp

    # full Experiment 1 (after register_targets.py has cached train + val)
    python scripts/train.py --epochs 50 --amp
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceCELoss
from monai.utils import set_determinism

from data.splits import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, attach_atlas_priors, split_with_atlases_removed
from data.transforms import PRIOR_KEYS, build_transforms
from models.segmentation_model import AtlasGuidedViTUNETR
from training.engine import NUM_CLASSES, ORGAN_NAMES, train_one_epoch, validate

IMG_SIZE = (96, 96, 96)


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        print(msg, flush=True)
        self.f.write(msg + "\n")
        self.f.flush()


def git_info():
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as e:  # git missing or not a repo
            return f"<unavailable: {e}>"

    return run("git", "rev-parse", "HEAD"), run("git", "status", "--porcelain"), run("git", "diff")


def build_model(device):
    return AtlasGuidedViTUNETR(
        in_channels=1,
        out_channels=NUM_CLASSES,
        img_size=IMG_SIZE,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
    ).to(device)


def load_weights(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default="exp_01_atlas_prior")
    p.add_argument("--run-name", default=None, help="default: timestamp")
    p.add_argument("--prior-source", choices=sorted(PRIOR_KEYS), default="atlas")
    p.add_argument("--variant", default="tps", help="registration cache variant (cache/warped_labels/<variant>)")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--init-checkpoint", default=os.path.join(_REPO_ROOT, "best_model.pth"), help="'none' to train from scratch")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="fp16 autocast on CUDA (needed to fit a 4 GB GPU)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dist-clip-mm", type=float, default=50.0)
    p.add_argument("--rotate-range-deg", type=float, default=None, help="default: the notebook's (-15, 15) RADIANS, for comparability")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--limit-val", type=int, default=None)
    p.add_argument("--max-steps-per-epoch", type=int, default=None)
    p.add_argument("--max-val-steps", type=int, default=None)
    p.add_argument("--allow-missing", action="store_true", help="train on the cached subset if some cases have no registration cache")
    p.add_argument("--skip-initial-eval", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    run_name = args.run_name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(_REPO_ROOT, "experiments", args.experiment, run_name)
    if os.path.exists(out_dir):
        raise SystemExit(f"run directory already exists, refusing to overwrite: {out_dir}")
    os.makedirs(out_dir)
    log = Tee(os.path.join(out_dir, "training.log"))

    random.seed(args.seed)
    np.random.seed(args.seed)
    set_determinism(seed=args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_cases, val_cases = split_with_atlases_removed(args.data_dir)
    if args.prior_source == "atlas":
        train_cases, miss_t = attach_atlas_priors(train_cases, args.variant, args.cache_dir)
        val_cases, miss_v = attach_atlas_priors(val_cases, args.variant, args.cache_dir)
        if (miss_t or miss_v) and not args.allow_missing:
            raise SystemExit(
                f"{len(miss_t)} train / {len(miss_v)} val cases have no registration cache under "
                f"{args.cache_dir}/warped_labels/{args.variant}. Run scripts/register_targets.py first, "
                f"or pass --allow-missing for a smoke test."
            )
    else:
        miss_t, miss_v = [], []
    if args.limit_train is not None:
        train_cases = train_cases[: args.limit_train]
    if args.limit_val is not None:
        val_cases = val_cases[: args.limit_val]

    rotate = (-15, 15) if args.rotate_range_deg is None else tuple(np.deg2rad([-args.rotate_range_deg, args.rotate_range_deg]))
    tf_kwargs = dict(prior_source=args.prior_source, rotate_range=rotate, dist_clip_mm=args.dist_clip_mm)
    train_loader = DataLoader(
        Dataset(train_cases, build_transforms(IMG_SIZE, NUM_CLASSES, train=True, **tf_kwargs)),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Dataset(val_cases, build_transforms(IMG_SIZE, NUM_CLASSES, train=False, **tf_kwargs)),
        batch_size=1, shuffle=False, num_workers=args.num_workers,
    )
    prior_keys = PRIOR_KEYS[args.prior_source]

    model = build_model(device)
    if args.init_checkpoint.lower() != "none":
        load_weights(model, args.init_checkpoint, device)
    criterion = DiceCELoss(to_onehot_y=False, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    commit, status, diff = git_info()
    config = {
        **vars(args),
        "run_dir": out_dir,
        "prior_keys": prior_keys,
        "rotate_range_used": list(map(float, rotate)),
        "num_train": len(train_cases),
        "num_val": len(val_cases),
        "missing_cache_train": miss_t,
        "missing_cache_val": miss_v,
        "img_size": IMG_SIZE,
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
    }
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    with open(os.path.join(out_dir, "command.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join([sys.executable] + sys.argv) + "\n")
    with open(os.path.join(out_dir, "git.txt"), "w", encoding="utf-8") as f:
        f.write(f"commit: {commit}\nuncommitted changes:\n{status}\n")
    with open(os.path.join(out_dir, "git_diff.patch"), "w", encoding="utf-8") as f:
        f.write(diff)

    log(f"Run: {out_dir}")
    log(f"Prior source: {args.prior_source} (keys {prior_keys}), variant={args.variant}")
    log(f"Train targets: {len(train_cases)}  Val: {len(val_cases)}  (missing cache: {len(miss_t)} train / {len(miss_v)} val)")
    log(f"Init: {args.init_checkpoint}  Device: {device}  AMP: {args.amp}")

    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "mean_dice"] + ORGAN_NAMES + ["epoch_time_s"])

    def record(epoch, train_loss, mean_dice, per_organ, secs):
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, mean_dice] + per_organ + [round(secs, 1)])
        log(f"Epoch {epoch}: train_loss={train_loss if train_loss is None else round(train_loss, 4)}  val Dice={mean_dice:.4f}")
        log("   " + "  ".join(f"{i + 1}:{'N/A' if s is None else f'{s:.3f}'}" for i, s in enumerate(per_organ)))

    best = {"epoch": None, "mean_dice": -1.0}
    if not args.skip_initial_eval:
        t0 = time.time()
        mean, per = validate(model, val_loader, device, prior_keys, amp=args.amp, max_steps=args.max_val_steps)
        record(0, None, mean, per, time.time() - t0)
        log("   ^ epoch 0 = initial weights evaluated with this prior source, before any fine-tuning")
        best = {"epoch": 0, "mean_dice": mean, "per_organ": per}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, prior_keys,
            scaler=scaler, amp=args.amp, max_steps=args.max_steps_per_epoch, log_every=50, log=log,
        )
        mean, per = validate(model, val_loader, device, prior_keys, amp=args.amp, max_steps=args.max_val_steps)
        record(epoch, loss, mean, per, time.time() - t0)

        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "best": best, "config": config},
            os.path.join(out_dir, "checkpoint_last.pth"),
        )
        if mean > best["mean_dice"]:
            best = {"epoch": epoch, "mean_dice": mean, "per_organ": per}
            torch.save(model.state_dict(), os.path.join(out_dir, "checkpoint_best.pth"))
            log(f"   new best (epoch {epoch})")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best, "config": config}, f, indent=2, default=str)
    log(f"Done. Best val Dice {best['mean_dice']:.4f} at epoch {best['epoch']}")


if __name__ == "__main__":
    main()
