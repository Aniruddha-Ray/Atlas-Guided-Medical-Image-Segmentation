"""
Quality-based atlas selection.

The first 10 atlases (scripts/select_atlas_cases.py) were the first cases with all 12 organs present,
round-robin by source, with no quality check; four of them have 5 mm slices. This script scans the
TRAINING split only (never validation) and keeps cases that pass explicit, label-and-header-only checks:

  1. all 12 organs present and the CT header matches the label header
  2. resolution: slice thickness <= --max-slice-mm and in-plane spacing <= --max-inplane-mm
  3. coverage: z-extent >= --min-z-mm. (An organ-to-border margin was tried and dropped: the `s` volumes are
     tight crops, so nearly every one has an organ touching the border without being truncated.)
  4. usable labels: for every organ the largest connected component holds >= --min-lcc of its voxels
  5. plausible anatomy: every organ volume within [--vol-lo, --vol-hi] x the pool median for that organ (this also
     catches truncated organs), and left/right kidney volume ratio within [1/--kidney-ratio, --kidney-ratio]

The bounds are deliberately wide: the aim is suitable, representative atlases, not perfect ones.
Survivors are ranked by how typical their solid-organ and vessel volumes are (mean |log(volume / median)| over
organs 1, 2, 3, 6, 8, 9, 10; the gallbladder, esophagus, stomach and adrenals vary too much between people), then selected
with a per-source quota so every scanner/annotation protocol is represented. Nothing here looks at
registration results or validation cases, so the choice cannot be tuned on validation.

The header and label checks cannot detect pathology or CT artefacts: look at the montage written next to
the selection (scripts/atlas_montage.py) before using a candidate.

Writes a CSV with the metrics of every scanned case and a candidate selection JSON in the same format as
atlas/atlas_selection.json. It never overwrites atlas/atlas_selection.json: switching atlases invalidates
every cached registration, so that is a separate, deliberate step.

    python scripts/select_quality_atlases.py --scan-only
    python scripts/select_quality_atlases.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from data.splits import DEFAULT_DATA_DIR, case_name, train_val_split

NUM_ORGANS = 12
STRUCT = np.ones((3, 3, 3), dtype=bool)
COMPACT_ORGANS = (1, 2, 3, 4, 6, 7, 10, 11, 12)  # margin_mm column only; not used as a filter (see docstring)
TYPICALITY_ORGANS = (1, 2, 3, 6, 8, 9, 10)
SOURCE_RE = re.compile(r"^([a-zA-Z]+)")


def source_of(name: str) -> str:
    m = SOURCE_RE.match(name)
    return m.group(1) if m else "unknown"


def scan_case(case: dict) -> dict:
    name = case_name(case["label"])
    lab = sitk.ReadImage(case["label"])
    sx, sy, sz = lab.GetSpacing()
    arr = np.asarray(sitk.GetArrayViewFromImage(lab)).astype(np.int16)  # (z, y, x)
    nz, ny, nx = arr.shape

    reader = sitk.ImageFileReader()
    reader.SetFileName(case["image"])
    reader.ReadImageInformation()
    geom_ok = (
        tuple(reader.GetSize()) == (nx, ny, nz)
        and np.allclose(reader.GetSpacing(), (sx, sy, sz), atol=1e-3)
        and np.allclose(reader.GetOrigin(), lab.GetOrigin(), atol=1e-2)
    )

    counts = np.bincount(arr[arr > 0].ravel(), minlength=NUM_ORGANS + 1)
    objs = ndimage.find_objects(arr, max_label=NUM_ORGANS)
    row = dict(
        name=name, source=source_of(name), sx=sx, sy=sy, sz=sz, nx=nx, ny=ny, nz=nz,
        zext_mm=nz * sz, geom_ok=int(geom_ok), stray_labels=int(counts[NUM_ORGANS + 1:].sum()),
    )
    margins, present = [], 0
    for c in range(1, NUM_ORGANS + 1):
        sl = objs[c - 1]
        if sl is None or counts[c] == 0:
            row.update({f"vol{c}": 0.0, f"lcc{c}": 0.0, f"ncomp{c}": 0, f"marg{c}": 0.0})
            continue
        present += 1
        sub = arr[sl] == c
        comp, n = ndimage.label(sub, STRUCT)
        sizes = np.bincount(comp.ravel())[1:]
        row[f"vol{c}"] = counts[c] * sx * sy * sz / 1000.0
        row[f"lcc{c}"] = float(sizes.max() / sizes.sum())
        row[f"ncomp{c}"] = int(n)
        m = min(sl[0].start * sz, (nz - sl[0].stop) * sz, sl[1].start * sy, (ny - sl[1].stop) * sy,
                sl[2].start * sx, (nx - sl[2].stop) * sx)
        row[f"marg{c}"] = float(m)
        if c in COMPACT_ORGANS:
            margins.append(m)
    row["n_present"] = present
    row["margin_mm"] = float(min(margins)) if margins else 0.0
    return row


def scan_pool(pool: list, workers: int) -> list:
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = []
        for i, r in enumerate(ex.map(scan_case, pool, chunksize=4), 1):
            rows.append(r)
            if i % 50 == 0 or i == len(pool):
                print(f"  scanned {i}/{len(pool)}", flush=True)
    return rows


def load_rows(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            rows.append({k: (v if k in ("name", "source") else float(v)) for k, v in r.items()})
    return rows


def write_rows(path: str, rows: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def apply_filters(rows: list, a) -> tuple:
    """Returns (survivors with 'typicality', funnel list of (label, count))."""
    funnel = [("scanned (training split)", len(rows))]
    cand = [r for r in rows if r["n_present"] == NUM_ORGANS and r["geom_ok"] == 1]
    funnel.append(("all 12 organs present, CT/label geometry match", len(cand)))

    med = {c: float(np.median([r[f"vol{c}"] for r in cand])) for c in range(1, NUM_ORGANS + 1)}

    steps = [
        (f"slice <= {a.max_slice_mm} mm and in-plane <= {a.max_inplane_mm} mm",
         lambda r: r["sz"] <= a.max_slice_mm and max(r["sx"], r["sy"]) <= a.max_inplane_mm),
        (f"z-extent >= {a.min_z_mm} mm", lambda r: r["zext_mm"] >= a.min_z_mm),
        (f"every organ's largest component >= {a.min_lcc:.0%} of its voxels",
         lambda r: all(r[f"lcc{c}"] >= a.min_lcc for c in range(1, NUM_ORGANS + 1))),
        (f"organ volumes within {a.vol_lo}-{a.vol_hi}x pool median, kidney ratio <= {a.kidney_ratio}",
         lambda r: all(a.vol_lo <= r[f"vol{c}"] / med[c] <= a.vol_hi for c in range(1, NUM_ORGANS + 1))
         and 1 / a.kidney_ratio <= r["vol2"] / max(r["vol3"], 1e-9) <= a.kidney_ratio),
    ]
    for label, fn in steps:
        cand = [r for r in cand if fn(r)]
        funnel.append((label, len(cand)))
    for r in cand:
        r["typicality"] = float(np.mean([abs(np.log(r[f"vol{c}"] / med[c])) for c in TYPICALITY_ORGANS]))
    return sorted(cand, key=lambda r: (r["typicality"], r["sz"])), funnel


def pick(survivors: list, count: int, quota: dict) -> list:
    chosen, by_src = [], {}
    for r in survivors:
        by_src.setdefault(r["source"], []).append(r)
    for src, q in quota.items():
        chosen += by_src.get(src, [])[:q]
    rest = [r for r in survivors if r not in chosen]
    return (chosen + rest)[:count] if len(chosen) < count else chosen[:count]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--metrics-csv", default=os.path.join(_REPO_ROOT, "experiments", "atlas_candidates", "case_metrics.csv"))
    p.add_argument("--out", default=os.path.join(_REPO_ROOT, "atlas", "atlas_selection_v2.json"))
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--quota", default="amos=4,img=1,s=5", help="atlases per source, roughly proportional to the pool (amos 39%, img 4%, s 56%) with every protocol present; filled from the best remaining if short")
    p.add_argument("--rescan", action="store_true")
    p.add_argument("--scan-only", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-slice-mm", type=float, default=3.0)
    p.add_argument("--max-inplane-mm", type=float, default=1.6)
    p.add_argument("--min-z-mm", type=float, default=300.0)
    p.add_argument("--min-lcc", type=float, default=0.90)
    p.add_argument("--vol-lo", type=float, default=0.3)
    p.add_argument("--vol-hi", type=float, default=3.0)
    p.add_argument("--kidney-ratio", type=float, default=2.0)
    a = p.parse_args()

    train, _val = train_val_split(a.data_dir)
    if os.path.exists(a.metrics_csv) and not a.rescan:
        rows = load_rows(a.metrics_csv)
        print(f"loaded {len(rows)} cached case metrics from {a.metrics_csv}")
    else:
        print(f"scanning {len(train)} training-split cases (labels and headers only)...")
        rows = scan_pool(train, a.workers)
        write_rows(a.metrics_csv, rows)
        print(f"wrote {a.metrics_csv}")
    if a.scan_only:
        return

    survivors, funnel = apply_filters(rows, a)
    print("\nfilter funnel (cases remaining):")
    for label, n in funnel:
        print(f"  {n:4d}  {label}")
    by_src = {}
    for r in survivors:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print("survivors by source:", by_src)

    quota = {k: int(v) for k, v in (kv.split("=") for kv in a.quota.split(","))}
    chosen = pick(survivors, a.count, quota)
    print(f"\nselected {len(chosen)} (quota {quota}):")
    for r in chosen:
        print(f"  {r['name']:11s} {r['source']:5s} slice {r['sz']:.1f} mm  in-plane {r['sx']:.2f}  z-extent {r['zext_mm']:.0f} mm  "
              f"typicality {r['typicality']:.3f}")

    by_name = {case_name(c["label"]): c for c in train}
    out = {
        "atlas_count": len(chosen),
        "num_organs": NUM_ORGANS,
        "selection_criteria": (
            "quality-based: all 12 organs, "
            f"slice<={a.max_slice_mm}mm, in-plane<={a.max_inplane_mm}mm, z-extent>={a.min_z_mm}mm, "
            f"largest component>={a.min_lcc:.0%}, organ volumes {a.vol_lo}-{a.vol_hi}x pool median, "
            f"kidney ratio<={a.kidney_ratio}; ranked by solid-organ volume typicality; "
            f"quota {a.quota}; drawn only from the training split (seed=42, val_fraction=0.2), never from validation. "
            "Header/label checks only: pathology and artefacts need a visual check."
        ),
        "cases": [{"image": by_name[r["name"]]["image"], "label": by_name[r["name"]]["label"]} for r in chosen],
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {a.out} (atlas/atlas_selection.json is NOT touched)")


if __name__ == "__main__":
    main()
