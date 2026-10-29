"""
Visual and intensity check of an atlas selection: one axial and one coronal slice per atlas with the label overlaid,
plus per-atlas CT statistics (orientation, HU range, mean HU inside the liver and spleen).

Header and label metrics (scripts/select_quality_atlases.py) cannot see CT artefacts, odd intensity scaling or
gross label errors; this is the cheap way to look at them before a selection is used.

    python scripts/atlas_montage.py --selection atlas/atlas_selection_v2.json --out experiments/atlas_candidates/montage_v2.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from data.splits import case_name

LIVER, SPLEEN = 6, 1  # inferred organ ids (see experiments/registration_log.md); used only for an HU plausibility check


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selection", default=os.path.join(_REPO_ROOT, "atlas", "atlas_selection_v2.json"))
    p.add_argument("--out", default=os.path.join(_REPO_ROOT, "experiments", "atlas_candidates", "montage_v2.png"))
    a = p.parse_args()

    cases = json.load(open(a.selection, encoding="utf-8"))["cases"]
    fig, axes = plt.subplots(len(cases), 2, figsize=(7, 3.2 * len(cases)))
    print(f"{'case':10s} {'orient':6s} {'HU min':>7s} {'HU max':>7s} {'liver HU':>9s} {'spleen HU':>10s}")
    for row, c in enumerate(cases):
        name = case_name(c["image"])
        img = sitk.ReadImage(c["image"])
        lab = sitk.ReadImage(c["label"])
        ct = sitk.GetArrayViewFromImage(img).astype(np.float32)  # (z, y, x)
        lb = np.asarray(sitk.GetArrayViewFromImage(lab)).astype(np.int16)
        sx, sy, sz = img.GetSpacing()
        orient = sitk.DICOMOrientImageFilter.GetOrientationFromDirectionCosines(img.GetDirection())
        liver = ct[lb == LIVER].mean() if (lb == LIVER).any() else float("nan")
        spleen = ct[lb == SPLEEN].mean() if (lb == SPLEEN).any() else float("nan")
        print(f"{name:10s} {orient:6s} {ct.min():7.0f} {ct.max():7.0f} {liver:9.0f} {spleen:10.0f}")

        zc, yc, _ = [int(np.mean(idx)) for idx in np.nonzero(lb > 0)]
        views = [
            (ct[zc], lb[zc], sy / sx, "axial"),
            (ct[:, yc, :], lb[:, yc, :], sz / sx, "coronal"),
        ]
        for col, (im, lm, aspect, title) in enumerate(views):
            ax = axes[row, col]
            ax.imshow(im, cmap="gray", vmin=-150, vmax=250, aspect=aspect, origin="lower" if title == "coronal" else "upper")
            overlay = np.ma.masked_where(lm == 0, lm)
            ax.imshow(overlay, cmap="tab20", alpha=0.45, vmin=0, vmax=20, aspect=aspect, interpolation="nearest",
                      origin="lower" if title == "coronal" else "upper")
            ax.set_title(f"{name}  {title}", fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=55)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
