"""
Builds the reliable-feature atlas library from atlas/atlas_selection.json:
landmarks per atlas -> golden transformations over all ordered atlas pairs ->
per-landmark reliability scores -> selected reliable landmarks.

Writes:
    cache/atlas_library/atlas_library.npz        (used by register_targets.py)
    cache/atlas_library/atlas_library_summary.json (per-atlas, per-organ stats)

Only the 10 atlas labels are read. No target label is ever touched here.

Usage:
    python scripts/build_atlas_library.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from atlas.reliable_features import build_atlas_library, save_atlas_library


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", default=os.path.join(_REPO_ROOT, "atlas", "atlas_selection.json"))
    parser.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "cache", "atlas_library"))
    parser.add_argument("--tolerance-mm", type=float, default=15.0)
    parser.add_argument("--tps-regularization", type=float, default=5.0)
    parser.add_argument("--min-score", type=float, default=0.3)
    parser.add_argument("--min-per-organ", type=int, default=3)
    args = parser.parse_args()

    with open(args.selection, encoding="utf-8") as f:
        selection = json.load(f)
    cases = selection["cases"]
    print(f"Building atlas library from {len(cases)} atlases")

    t0 = time.time()
    library = build_atlas_library(
        cases,
        organ_ids=range(1, selection["num_organs"] + 1),
        tolerance_mm=args.tolerance_mm,
        tps_regularization=args.tps_regularization,
        min_score=args.min_score,
        min_per_organ=args.min_per_organ,
    )
    elapsed = time.time() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    save_atlas_library(library, os.path.join(args.out_dir, "atlas_library.npz"))

    summary = {"config": vars(args), "elapsed_s": round(elapsed, 1), "atlases": []}
    print(f"\nDone in {elapsed:.1f}s. Selected reliable landmarks per atlas:")
    for entry in library:
        per_organ = {}
        for organ_id in np.unique(entry["all_organ_ids"]):
            organ_id = int(organ_id)
            all_mask = entry["all_organ_ids"] == organ_id
            sel_mask = entry["organ_ids"] == organ_id
            per_organ[organ_id] = {
                "total": int(all_mask.sum()),
                "selected": int(sel_mask.sum()),
                "mean_score": round(float(entry["all_scores"][all_mask].mean()), 3),
            }
        name = os.path.basename(entry["image"])
        print(f"  {name:28s} selected {len(entry['points']):3d}/{len(entry['all_points'])}")
        summary["atlases"].append({"image": entry["image"], "selected": int(len(entry["points"])), "per_organ": per_organ})

    print("\nMean reliability per organ (averaged over atlases):")
    for organ_id in sorted(summary["atlases"][0]["per_organ"]):
        scores = [a["per_organ"][organ_id]["mean_score"] for a in summary["atlases"] if organ_id in a["per_organ"]]
        kept = [a["per_organ"][organ_id]["selected"] for a in summary["atlases"] if organ_id in a["per_organ"]]
        print(f"  organ {organ_id:2d}: mean score {np.mean(scores):.3f}, kept per atlas {np.mean(kept):.1f}")

    with open(os.path.join(args.out_dir, "atlas_library_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
