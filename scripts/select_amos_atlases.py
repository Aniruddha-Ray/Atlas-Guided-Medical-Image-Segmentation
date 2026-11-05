"""
AMOS-only study set: a fixed validation split, 10 quality-selected atlases from AMOS, and the training pool.

Pool = the AMOS cases of Dataset061_AMOS (scripts/extract_sources.py). In this order:
  1. The validation cases are drawn first (--val-size, seed --seed) and never used for atlases.
  2. Atlases are selected from the rest with the same label-and-header-only checks as
     scripts/select_quality_atlases.py (all 12 organs, slice <= 3 mm, in-plane <= 1.6 mm, z-extent >= 300 mm,
     largest component >= 90 %, organ volumes 0.3-3x the pool median, kidney ratio <= 2), ranked by volume typicality.
  3. Training pool = everything that is neither validation nor atlas. Atlases are never targets: they are dropped
     from the data and need no registration.

Writes (never overwriting an existing atlas selection unless --force):
    atlas/atlas_selection_amos.json                   the 10 atlases, in atlas_selection.json's format
    experiments/exp_amos/amos_chunk.json              {"train": [...], "val": [...], "atlases": [...]} case names
    experiments/exp_amos/amos_case_metrics.csv        header and label metrics of every scanned case

    python scripts/select_amos_atlases.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data.splits import case_name, paired_cases  # noqa: E402
from scripts.select_quality_atlases import NUM_ORGANS, apply_filters, load_rows, pick, scan_pool, write_rows  # noqa: E402

AMOS_DIR = os.path.join(os.path.dirname(_REPO_ROOT), "Dataset061_AMOS")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=AMOS_DIR)
    p.add_argument("--val-size", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--out", default=os.path.join(_REPO_ROOT, "atlas", "atlas_selection_amos.json"))
    p.add_argument("--chunk-out", default=os.path.join(_REPO_ROOT, "experiments", "exp_amos", "amos_chunk.json"))
    p.add_argument("--metrics-csv", default=os.path.join(_REPO_ROOT, "experiments", "exp_amos", "amos_case_metrics.csv"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--rescan", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max-slice-mm", type=float, default=3.0)
    p.add_argument("--max-inplane-mm", type=float, default=1.6)
    p.add_argument("--min-z-mm", type=float, default=300.0)
    p.add_argument("--min-lcc", type=float, default=0.90)
    p.add_argument("--vol-lo", type=float, default=0.3)
    p.add_argument("--vol-hi", type=float, default=3.0)
    p.add_argument("--kidney-ratio", type=float, default=2.0)
    a = p.parse_args()

    for path in (a.out, a.chunk_out):
        if os.path.exists(path) and not a.force:
            raise SystemExit(f"{path} exists; the selection and the split must stay fixed (use --force to replace)")

    cases = sorted(paired_cases(a.data_dir), key=lambda c: case_name(c["image"]))
    names = [case_name(c["image"]) for c in cases]
    if any(not n.startswith("amos") for n in names):
        raise SystemExit(f"{a.data_dir} holds non-AMOS cases")
    order = list(names)
    random.Random(a.seed).shuffle(order)
    val = sorted(order[: a.val_size])
    pool = [c for c in cases if case_name(c["image"]) not in set(val)]
    print(f"{len(cases)} AMOS cases: {len(val)} validation (fixed, seed {a.seed}), {len(pool)} in the pool for atlases and training")

    if os.path.exists(a.metrics_csv) and not a.rescan:
        rows = load_rows(a.metrics_csv)
        print(f"loaded {len(rows)} cached case metrics from {a.metrics_csv}")
    else:
        print(f"scanning {len(pool)} pool cases (labels and headers only; validation cases are not scanned)...")
        rows = scan_pool(pool, a.workers)
        write_rows(a.metrics_csv, rows)
    assert not ({r["name"] for r in rows} & set(val)), "a validation case is in the scanned pool"

    survivors, funnel = apply_filters(rows, SimpleNamespace(**vars(a)))
    print("\nfilter funnel (cases remaining):")
    for label, n in funnel:
        print(f"  {n:4d}  {label}")
    chosen = pick(survivors, a.count, {"amos": a.count})
    if len(chosen) < a.count:
        raise SystemExit(f"only {len(chosen)} cases pass the filters; loosen them deliberately and say so in the paper")
    print(f"\nselected {len(chosen)} atlases:")
    for r in chosen:
        print(f"  {r['name']:10s} slice {r['sz']:.1f} mm  in-plane {r['sx']:.2f}  z-extent {r['zext_mm']:.0f} mm  typicality {r['typicality']:.3f}")

    by_name = {case_name(c["image"]): c for c in cases}
    atlas = sorted(r["name"] for r in chosen)
    train = [n for n in names if n not in set(val) and n not in set(atlas)]
    os.makedirs(os.path.dirname(a.chunk_out), exist_ok=True)
    json.dump({"train": train, "val": val, "atlases": atlas, "seed": a.seed}, open(a.chunk_out, "w", encoding="utf-8"), indent=1)
    out = {
        "atlas_count": len(chosen), "num_organs": NUM_ORGANS,
        "selection_criteria": (
            f"AMOS only. {a.val_size} validation cases drawn first (seed {a.seed}) and excluded; from the rest: all 12 organs, "
            f"slice<={a.max_slice_mm}mm, in-plane<={a.max_inplane_mm}mm, z-extent>={a.min_z_mm}mm, largest component>={a.min_lcc:.0%}, "
            f"organ volumes {a.vol_lo}-{a.vol_hi}x pool median, kidney ratio<={a.kidney_ratio}; ranked by solid-organ volume "
            "typicality. Header/label checks only: pathology and artefacts need a visual check."),
        "cases": [{"image": by_name[n]["image"], "label": by_name[n]["label"]} for n in atlas],
    }
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {a.out}\nwrote {a.chunk_out}: train {len(train)}, val {len(val)}, atlases {len(atlas)}")


if __name__ == "__main__":
    main()
