"""
Dice and HD95 (mm) per organ and case, for one or more prediction folders against one ground-truth folder.

HD95: the larger of the two directed 95th-percentile surface distances (prediction -> truth, truth -> prediction), in
mm on the files' own grid. An organ in the truth but absent from the prediction (or the reverse) gets Dice 0 and HD95 =
the image diagonal (the worst possible), and is counted as "missed" (or "false organ"). Absent from both: not scored.

    python scripts/eval_seg.py --gt-dir <labels> --arm A=<pred dir> --arm C=<pred dir> --cases img0001 ... \
        --out experiments/exp_cascade/results_btcv
writes <out>.csv (case, organ, arm, dice, hd95, status) and <out>.md (per-organ means per arm, worst cases).
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt

ORGANS = ["spleen", "right kidney", "left kidney", "gallbladder", "esophagus", "liver", "stomach", "aorta", "IVC",
          "pancreas", "right adrenal", "left adrenal"]


def surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, border_value=0)


def hd95(a: np.ndarray, b: np.ndarray, spacing) -> float:
    """Symmetric 95th-percentile Hausdorff distance between two non-empty masks (computed in their joint bounding box)."""
    idx = np.argwhere(a | b)
    lo, hi = np.maximum(idx.min(0) - 2, 0), idx.max(0) + 3
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    a, b = a[sl], b[sl]
    sa, sb = surface(a), surface(b)
    da = distance_transform_edt(~sb, sampling=spacing)[sa]  # a's surface -> b's surface
    db = distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(max(np.percentile(da, 95), np.percentile(db, 95)))


def score_case(pred_path: str, gt_path: str) -> list:
    pred_img, gt_img = sitk.ReadImage(pred_path), sitk.ReadImage(gt_path)
    if pred_img.GetSize() != gt_img.GetSize():
        raise RuntimeError(f"{pred_path}: grid differs from {gt_path}")
    pred, gt = sitk.GetArrayFromImage(pred_img), sitk.GetArrayFromImage(gt_img)
    spacing = gt_img.GetSpacing()[::-1]
    diag = float(np.linalg.norm(np.array(gt.shape) * np.array(spacing)))
    rows = []
    for k in range(1, len(ORGANS) + 1):
        p, g = pred == k, gt == k
        if not g.any() and not p.any():
            continue
        if g.any() and p.any():
            d = 2.0 * np.logical_and(p, g).sum() / (p.sum() + g.sum())
            rows.append((k, float(d), hd95(p, g, spacing), "ok"))
        else:
            rows.append((k, 0.0, diag, "missed" if g.any() else "false organ"))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-dir", required=True)
    p.add_argument("--arm", action="append", required=True, help="NAME=prediction folder (repeatable)")
    p.add_argument("--cases", nargs="*", default=None, help="case names (default: every prediction of the first arm)")
    p.add_argument("--out", required=True, help="output path without extension")
    a = p.parse_args()

    arms = [tuple(s.split("=", 1)) for s in a.arm]
    cases = a.cases or sorted(os.path.basename(f)[:-7] for f in glob.glob(os.path.join(arms[0][1], "*.nii.gz")))
    rows = []
    for name, folder in arms:
        for c in cases:
            for k, d, h, status in score_case(os.path.join(folder, f"{c}.nii.gz"), os.path.join(a.gt_dir, f"{c}.nii.gz")):
                rows.append({"arm": name, "case": c, "organ": ORGANS[k - 1], "dice": d, "hd95": h, "status": status})
        print(f"{name}: {len(cases)} cases scored", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    names = [n for n, _ in arms]
    lines = [f"# Dice / HD95 (mm) on {len(cases)} cases", "",
             "| organ | " + " | ".join(f"{n} Dice | {n} HD95" for n in names) + " |",
             "|---|" + "---|---|" * len(names)]
    for organ in ORGANS + ["mean"]:
        cells = []
        for n in names:
            sel = [r for r in rows if r["arm"] == n and (organ == "mean" or r["organ"] == organ)]
            cells.append(f"{np.mean([r['dice'] for r in sel]):.3f} | {np.mean([r['hd95'] for r in sel]):.1f}" if sel else "- | -")
        lines.append(f"| {'**mean**' if organ == 'mean' else organ} | " + " | ".join(cells) + " |")
    lines += ["", "| arm | missed organs | false organs | worst case (mean Dice) | cases with mean Dice < 0.7 |", "|---|---|---|---|---|"]
    for n in names:
        sel = [r for r in rows if r["arm"] == n]
        per_case = {c: np.mean([r["dice"] for r in sel if r["case"] == c]) for c in cases}
        worst = min(per_case, key=per_case.get)
        lines.append(f"| {n} | {sum(r['status'] == 'missed' for r in sel)} | {sum(r['status'] == 'false organ' for r in sel)} | "
                     f"{worst} ({per_case[worst]:.3f}) | {sum(v < 0.7 for v in per_case.values())} |")
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
