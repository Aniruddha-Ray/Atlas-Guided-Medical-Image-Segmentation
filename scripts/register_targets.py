"""
Registers every atlas in the library onto each target CT and caches the
warped atlas labels. This is the expensive, run-once step (research brief
Section 30): training later only reads the cache.

Per target it writes (<variant> = "tps" or "tps_bspline", see --refinement):
    cache/warped_labels/<variant>/<case>.nii.gz   4D uint8 (X, Y, Z, N_atlas), target 2 mm grid
    cache/registrations/<variant>/<case>.json     per-atlas QA (status, inliers, fold fraction, ...)
    cache/qa/<variant>/<case>.png                 visual QA            (--qa-png)
Variants live side by side so they can be compared (ablation) without either
overwriting the other.

Target ground truth is NEVER used to build anything. With --eval-gt it is
read afterwards, only to report how good the atlas-only (majority vote)
segmentation is -- a diagnostic of registration quality, stored separately
under "evaluation_only" in the QA json.

Leakage guard: the script refuses to register an atlas case as a target.

Usage (a small chunk locally, the rest on the remote server):
    python scripts/register_targets.py --split val --limit 2 --eval-gt --qa-png
    python scripts/register_targets.py --split val --limit 2 --eval-gt --qa-png --refinement bspline
    python scripts/register_targets.py --split train --refinement bspline
    # parallel on a server: one process per shard, e.g. 8 shards
    python scripts/register_targets.py --split train --num-shards 8 --shard-index 0   # ... up to 7
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from dataclasses import asdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import SimpleITK as sitk

from atlas.geometry import read_ct, read_label, resample_isotropic, resample_to_grid, to_xyz_array
from atlas.label_warping import save_label_stack, vote_counts, warp_ct, warp_label
from atlas.registration import RegistrationConfig, prepare_target, register_atlas_to_target
from atlas.reliable_features import load_atlas_library

NUM_CLASSES = 13


def split_cases(data_dir: str, split: str, val_fraction: float = 0.2, seed: int = 42):
    """Identical split logic to the notebook / evaluate_checkpoint.py."""
    image_files = sorted(glob.glob(os.path.join(data_dir, "*_0000.nii.gz")))
    label_files = set(sorted(glob.glob(os.path.join(data_dir, "*.nii.gz"))))
    pairs = []
    for img in image_files:
        lbl = os.path.join(data_dir, os.path.basename(img).replace("_0000.nii.gz", ".nii.gz"))
        if lbl in label_files:
            pairs.append({"image": os.path.abspath(img), "label": os.path.abspath(lbl)})
    random.seed(seed)
    random.shuffle(pairs)
    cut = int(len(pairs) * (1 - val_fraction))
    return pairs[:cut] if split == "train" else pairs[cut:]


def case_name(path: str) -> str:
    return os.path.basename(path).replace("_0000.nii.gz", "").replace(".nii.gz", "")


def dice_per_organ(pred: np.ndarray, gt: np.ndarray, num_classes: int = NUM_CLASSES) -> dict:
    out = {}
    for c in range(1, num_classes):
        p, g = pred == c, gt == c
        denom = p.sum() + g.sum()
        out[c] = float(2.0 * (p & g).sum() / denom) if denom > 0 else float("nan")
    return out


def save_qa_png(path, target_grid, warped_ct0, counts, gt, n_atlas):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ct = to_xyz_array(target_grid)
    wct = to_xyz_array(warped_ct0)
    mv = counts.argmax(axis=0)
    # Slice through the atlas-predicted organs, not the body centre: scans that
    # extend into the pelvis put the body centre below every organ.
    organs = np.argwhere(mv > 0)
    if not len(organs):
        organs = np.argwhere(ct > -500)
    cx, cy, cz = organs.mean(axis=0).astype(int) if len(organs) else np.array(ct.shape) // 2

    def conf(slc_counts):
        P = slc_counts.astype(np.float32) / n_atlas
        H = -(P * np.log(P + 1e-8)).sum(axis=0)
        return 1.0 - H / np.log(NUM_CLASSES)

    views = {
        "axial": (lambda a: a[:, :, cz].T, lambda c: c[:, :, :, cz].transpose(0, 2, 1), "upper"),
        "coronal": (lambda a: a[:, cy, :].T, lambda c: c[:, :, cy, :].transpose(0, 2, 1), "lower"),
    }
    cols = ["target CT", "atlas 0 warped CT", "majority vote", "confidence w(v)"] + (["GT (eval only)"] if gt is not None else [])
    fig, axes = plt.subplots(len(views), len(cols), figsize=(4 * len(cols), 8))
    for r, (vname, (sl, slc, origin)) in enumerate(views.items()):
        panels = [
            ("ct", sl(ct)),
            ("ct", sl(wct)),
            ("lab", sl(mv)),
            ("conf", conf(slc(counts))),
        ]
        if gt is not None:
            panels.append(("lab", sl(gt)))
        for c, (kind, img) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(np.clip(sl(ct), -200, 300), cmap="gray", origin=origin)
            if kind == "ct":
                ax.imshow(np.clip(img, -200, 300), cmap="gray", origin=origin)
            elif kind == "lab":
                ax.imshow(np.ma.masked_equal(img, 0), cmap="nipy_spectral", vmin=0, vmax=NUM_CLASSES - 1, alpha=0.5, origin=origin)
            else:
                ax.imshow(img, cmap="magma", vmin=0, vmax=1, origin=origin)
            ax.set_title(f"{vname}: {cols[c]}", fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(_REPO_ROOT), "Dataset060_Merged_Def"))
    parser.add_argument("--atlas-library", default=os.path.join(_REPO_ROOT, "cache", "atlas_library", "atlas_library.npz"))
    parser.add_argument("--selection", default=os.path.join(_REPO_ROOT, "atlas", "atlas_selection.json"))
    parser.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "cache"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--limit", type=int, default=None, help="only the first N targets of the split")
    parser.add_argument("--cases", nargs="*", default=None, help="explicit case names, e.g. amos_0278")
    parser.add_argument("--eval-gt", action="store_true", help="report atlas-only Dice vs GT (evaluation only)")
    parser.add_argument("--qa-png", action="store_true")
    parser.add_argument("--force", action="store_true", help="recompute even if cached output exists")
    parser.add_argument("--device", default=None)
    parser.add_argument("--refinement", choices=["none", "bspline"], default="none", help="dense refinement after landmark TPS")
    parser.add_argument("--num-shards", type=int, default=1, help="split the target list across N parallel processes")
    parser.add_argument("--shard-index", type=int, default=0, help="which shard this process handles (0-based)")
    args = parser.parse_args()

    cfg = RegistrationConfig(nonrigid_refinement=args.refinement)
    variant = "tps" if args.refinement == "none" else "tps_bspline"
    library = load_atlas_library(args.atlas_library)
    with open(args.selection, encoding="utf-8") as f:
        atlas_images = {os.path.normcase(os.path.abspath(c["image"])) for c in json.load(f)["cases"]}

    targets = split_cases(args.data_dir, args.split)
    if args.cases:
        wanted = set(args.cases)
        targets = [t for t in targets if case_name(t["image"]) in wanted]
    before = len(targets)
    targets = [t for t in targets if os.path.normcase(t["image"]) not in atlas_images]
    if before != len(targets):
        print(f"Excluded {before - len(targets)} atlas case(s) from the target list (leakage guard).")
    if args.limit is not None:
        targets = targets[: args.limit]
    if args.num_shards > 1:
        targets = targets[args.shard_index :: args.num_shards]
        print(f"Shard {args.shard_index + 1}/{args.num_shards}: {len(targets)} target(s)")

    dirs = {sub: os.path.join(args.out_dir, sub, variant) for sub in ("warped_labels", "registrations", "qa")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"Loading {len(library)} atlases...")
    atlas_cts = [resample_isotropic(read_ct(e["image"]), cfg.match_spacing_mm, sitk.sitkLinear, -1000.0) for e in library]
    atlas_labels = [read_label(e["label"]) for e in library]

    print(f"Registering {len(library)} atlases onto {len(targets)} target(s) [{args.split}, variant={variant}]")
    for t_idx, target in enumerate(targets):
        name = case_name(target["image"])
        stack_path = os.path.join(dirs["warped_labels"], f"{name}.nii.gz")
        qa_path = os.path.join(dirs["registrations"], f"{name}.json")
        if os.path.exists(stack_path) and os.path.exists(qa_path) and not args.force:
            print(f"[{t_idx + 1}/{len(targets)}] {name}: cached, skipping")
            continue

        t0 = time.time()
        prep = prepare_target(read_ct(target["image"]), cfg)
        warped, per_atlas, warped_ct0 = [], [], None
        for a_idx, entry in enumerate(library):
            ta = time.time()
            reg = register_atlas_to_target(atlas_cts[a_idx], entry["points"], prep, cfg, device=args.device)
            tx = reg.to_sitk_transform(prep["grid"])
            warped.append(warp_label(atlas_labels[a_idx], prep["grid"], tx))
            if a_idx == 0 and args.qa_png:
                warped_ct0 = warp_ct(atlas_cts[0], prep["grid"], tx)
            qa = {k: v for k, v in reg.qa.items() if k != "config"}
            qa.update(atlas=case_name(entry["image"]), time_s=round(time.time() - ta, 1))
            per_atlas.append(qa)
            bs = qa.get("bspline")
            bs_msg = "" if bs is None else f"  bspline {'kept' if bs['kept'] else 'REJECTED'} MI {bs['metric_before']:.3f}->{bs['metric_after']:.3f} fold={bs['fold_fraction']:.4f}"
            print(
                f"    atlas {a_idx}: {qa['status']:<18s} inliers={qa['ransac'].get('num_inliers', 0):3d}"
                f"/{qa['num_candidates']:3d} cand  fold={qa.get('fold_fraction', float('nan')):.4f}{bs_msg}  {qa['time_s']}s"
            )

        stack = save_label_stack(warped, prep["grid"], stack_path)
        record = {
            "target": target["image"],
            "grid_spacing_mm": cfg.match_spacing_mm,
            "variant": variant,
            "config": asdict(cfg),
            "atlases": per_atlas,
            "status_counts": {s: sum(p["status"] == s for p in per_atlas) for s in ("tps", "ransac_affine_only", "coarse_affine_only")},
            "elapsed_s": round(time.time() - t0, 1),
        }

        counts = None
        gt = None
        if args.eval_gt or args.qa_png:
            counts = vote_counts(stack, NUM_CLASSES)
        if args.eval_gt:
            gt = to_xyz_array(resample_to_grid(read_label(target["label"]), prep["grid"], sitk.sitkNearestNeighbor, 0))
            mv = counts.argmax(axis=0)
            mas = dice_per_organ(mv, gt)
            single = [dice_per_organ(stack[..., k], gt) for k in range(stack.shape[-1])]
            record["evaluation_only"] = {
                "note": "GT used only for this report, never for building priors",
                "majority_vote_dice": mas,
                "majority_vote_mean_dice": float(np.nanmean(list(mas.values()))),
                "single_atlas_mean_dice": [float(np.nanmean(list(s.values()))) for s in single],
            }

        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        if args.qa_png:
            save_qa_png(os.path.join(dirs["qa"], f"{name}.png"), prep["grid"], warped_ct0, counts, gt, len(library))

        msg = f"[{t_idx + 1}/{len(targets)}] {name}: {record['status_counts']} in {record['elapsed_s']}s"
        if args.eval_gt:
            ev = record["evaluation_only"]
            msg += f" | atlas-only Dice (eval): majority {ev['majority_vote_mean_dice']:.4f}"
            msg += " | per organ " + " ".join(f"{c}:{d:.2f}" for c, d in ev["majority_vote_dice"].items())
        print(msg)


if __name__ == "__main__":
    main()
