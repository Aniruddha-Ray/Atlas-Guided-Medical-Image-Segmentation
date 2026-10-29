"""
Evaluation-only: atlas-only majority-vote Dice of cached registrations against the target labels.

Same definition as scripts/register_targets.py --eval-gt (per organ Dice, mean over organs that are present or
predicted), but works on stacks that are already cached, so two atlas sets can be compared on the same targets
without registering anything again. The target label is used only for this score, never to build priors.

    python scripts/eval_atlas_priors.py --cache-dir cache --cases amos_0296 s0339
    python scripts/eval_atlas_priors.py --cache-dir cache/atlas_v2 --cases amos_0296 s0339
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from atlas.label_warping import vote_counts
from data.splits import DEFAULT_DATA_DIR

NUM_CLASSES = 13


def dice_per_organ(pred: np.ndarray, gt: np.ndarray) -> dict:
    out = {}
    for c in range(1, NUM_CLASSES):
        p, g = pred == c, gt == c
        denom = p.sum() + g.sum()
        out[c] = float(2.0 * (p & g).sum() / denom) if denom > 0 else float("nan")
    return out


def evaluate_case(name: str, cache_dir: str, data_dir: str, variant: str = "tps") -> dict:
    stack_img = nib.load(os.path.join(cache_dir, "warped_labels", variant, f"{name}.nii.gz"))
    stack = np.asarray(stack_img.dataobj).astype(np.uint8)  # (X, Y, Z, N)
    gt_img = nib.load(os.path.join(data_dir, f"{name}.nii.gz"))
    gt = np.rint(resample_from_to(gt_img, (stack.shape[:3], stack_img.affine), order=0).get_fdata()).astype(np.int16)
    pred = vote_counts(stack, NUM_CLASSES).argmax(axis=0)
    per_organ = dice_per_organ(pred, gt)
    reg = json.load(open(os.path.join(cache_dir, "registrations", variant, f"{name}.json"), encoding="utf-8"))
    return {
        "case": name,
        "mean_dice": float(np.nanmean(list(per_organ.values()))),
        "per_organ": per_organ,
        "status_counts": reg["status_counts"],
        "n_atlases": int(stack.shape[-1]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=os.path.join(_REPO_ROOT, "cache"))
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--variant", default="tps")
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--out", default=None, help="optional JSON file for the per-case results")
    a = p.parse_args()

    results = []
    for name in a.cases:
        r = evaluate_case(name, a.cache_dir, a.data_dir, a.variant)
        results.append(r)
        organs = " ".join(f"{c}:{v:.2f}" for c, v in r["per_organ"].items())
        print(f"{name:10s} mean Dice {r['mean_dice']:.3f}  fallbacks {r['status_counts'].get('coarse_affine_only', 0)}  | {organs}", flush=True)
    d = np.array([r["mean_dice"] for r in results])
    print(f"\n{len(d)} cases: mean {d.mean():.3f}  median {np.median(d):.3f}  min {d.min():.3f}  max {d.max():.3f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
