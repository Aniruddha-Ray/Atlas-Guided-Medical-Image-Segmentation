"""
Validation-only reproduction of a trained checkpoint's Dice score, without
resuming/continuing training.

This fills the gap noted in research_audit.md Section E / F: the notebook's
training loop (cell 27) always resumes from latest_checkpoint.pth and keeps
training: there was no way to just re-run validation. best_model.pth is the
epoch-101 weights (state_dict only, saved whenever validation Dice improved;
Dice peaked at epoch 101 with 0.6505 and never improved again in epochs
102-103), so it is loaded here.

IMPORTANT: this script builds validation data with legacy_buggy_distance=True
by default, i.e. the ORIGINAL (buggy) distance-map semantics the epoch-101
checkpoint was actually trained on (see data/transforms.py docstring and
research_audit.md Section G). That is required to reproduce the historical
Dice=0.6505 number. Do not use --legacy-buggy-distance for evaluating any
checkpoint trained after the bugfix.

Usage:
    python scripts/evaluate_checkpoint.py \\
        --checkpoint best_model.pth \\
        --data-dir "../Dataset060_Merged_Def" \\
        --legacy-buggy-distance

Use --limit N to run a smoke test on just the first N validation cases
(deterministic subset -- same split, just truncated) instead of the full
136-case set. That's for confirming the pipeline runs end-to-end and
produces sane per-case Dice on a slow/local machine; it is NOT a substitute
for the full run when reporting metrics -- run the full set (no --limit) on
a machine that can finish it in reasonable time, e.g. a remote GPU.

--prior-source atlas evaluates with atlas-derived priors (P, D from the
cached warped atlas labels, see scripts/register_targets.py) instead of the
case's own ground truth. Cases without a registration cache are skipped and
reported. --legacy-buggy-distance only applies to --prior-source gt.
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import sys

# Running this file directly (`python scripts/evaluate_checkpoint.py`) puts
# scripts/ itself on sys.path, not the repo root -- so `import data`/`import
# models` fail unless the repo root is added explicitly first.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
from monai.data import CacheDataset, DataLoader
from monai.metrics import DiceMetric

from data.splits import attach_atlas_priors
from data.transforms import PRIOR_KEYS, build_transforms
from models.segmentation_model import AtlasGuidedViTUNETR

IMG_SIZE = (96, 96, 96)
NUM_CLASSES = 13
ORGAN_NAMES = [f"Organ {i}" for i in range(1, 13)]


def build_val_files(data_dir: str, val_fraction: float = 0.2, seed: int = 42):
    """Mirrors the notebook's data-prep cell exactly (same seed, same split)."""
    image_files = sorted(glob.glob(os.path.join(data_dir, "*_0000.nii.gz")))
    label_files = sorted(glob.glob(os.path.join(data_dir, "*.nii.gz")))

    paired_label_files = []
    filtered_image_files = []
    for img_path in image_files:
        base_name = os.path.basename(img_path).replace("_0000.nii.gz", ".nii.gz")
        lbl_path = os.path.join(data_dir, base_name)
        if lbl_path in label_files:
            filtered_image_files.append(img_path)
            paired_label_files.append(lbl_path)

    data_dicts = [{"image": img, "label": lbl} for img, lbl in zip(filtered_image_files, paired_label_files)]
    random.seed(seed)
    random.shuffle(data_dicts)

    train_split = int(len(data_dicts) * (1 - val_fraction))
    return data_dicts[train_split:]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="best_model.pth", help="state_dict-only checkpoint (best_model.pth) or full checkpoint dict (latest_checkpoint.pth)")
    parser.add_argument("--data-dir", default=r"C:\\Users\\Asus\\OneDrive\\Desktop\\Atlas Based Medical Segmentation\\Dataset060_Merged_Def")
    parser.add_argument("--legacy-buggy-distance", action="store_true", help="Reproduce the original (buggy) distance-map semantics the epoch-101/103 checkpoints were trained on.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N validation cases (smoke test). Omit for the full set.")
    parser.add_argument("--prior-source", choices=sorted(PRIOR_KEYS), default="gt")
    parser.add_argument("--variant", default="tps", help="registration cache variant for --prior-source atlas")
    args = parser.parse_args()
    if args.prior_source == "atlas" and args.legacy_buggy_distance:
        parser.error("--legacy-buggy-distance only applies to --prior-source gt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_files = build_val_files(args.data_dir)
    if args.prior_source == "atlas":
        val_files, missing = attach_atlas_priors(val_files, args.variant)
        if missing:
            print(f"Skipping {len(missing)} validation case(s) without a registration cache ({args.variant}).")
    full_val_count = len(val_files)
    if args.limit is not None:
        val_files = val_files[: args.limit]
        print(
            f"PARTIAL RUN: evaluating {len(val_files)}/{full_val_count} validation cases "
            f"(--limit {args.limit}). This is a smoke test, NOT the reported baseline metric."
        )
    else:
        print(f"Validation samples: {len(val_files)}")

    val_transforms = build_transforms(
        IMG_SIZE, NUM_CLASSES, train=False, legacy_buggy_distance=args.legacy_buggy_distance, prior_source=args.prior_source
    )
    prob_key, dist_key = PRIOR_KEYS[args.prior_source]
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = AtlasGuidedViTUNETR(
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

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)

    with torch.no_grad():
        dice_metric.reset()
        for i, val_batch_data in enumerate(val_loader):
            case_name = os.path.basename(val_files[i]["image"])
            print(f"[{i + 1}/{len(val_files)}] {case_name} ...", end=" ", flush=True)

            val_images = val_batch_data["image"].to(device)
            val_labels = val_batch_data["label"].to(device)
            val_preds_logits = model(val_images, val_batch_data[prob_key].to(device), val_batch_data[dist_key].to(device))
            val_preds_soft = F.softmax(val_preds_logits, dim=1)
            pred_classes = val_preds_soft.argmax(dim=1)
            val_preds_discrete = F.one_hot(pred_classes.long(), NUM_CLASSES).permute(0, 4, 1, 2, 3).float()

            dice_metric(y_pred=val_preds_discrete, y=val_labels)
            print("done")

        per_class_dice, not_nans = dice_metric.aggregate()
        mean_dice = per_class_dice[not_nans.bool()].mean().item()

    if args.limit is not None:
        print(f"\n[PARTIAL, {len(val_files)}/{full_val_count} cases] Validation Dice: {mean_dice:.4f}")
    else:
        print(f"\nValidation Dice: {mean_dice:.4f}")
    print(f"  {'Organ':<25} {'Dice':>6}")
    print("  " + "-" * 33)
    for name, score, present in zip(ORGAN_NAMES, per_class_dice, not_nans):
        tag = f"{score:.4f}" if present else "N/A"
        print(f"  {name:<25} {tag:>6}")


if __name__ == "__main__":
    main()
