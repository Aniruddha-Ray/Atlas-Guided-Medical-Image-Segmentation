"""
Selects the 10 temporary atlas cases (see research_audit.md / discussion log)
from the TRAINING split only, so no atlas ever appears as a validation target
and Experiment 0 vs Experiment 1 stay comparable on the same 136 validation
cases.

Selection criteria, in order:
  1. All 12 organs present in the label (an atlas missing an organ votes 0
     for it everywhere and drags P(v) down for every target).
  2. Balanced across the dataset's 3 source prefixes (amos_*, img*, s*) so
     the atlas library isn't dominated by one scanner/annotation protocol.
  3. Deterministic given the same data directory (sorted candidate order,
     fixed per-source take-counts) -- no RNG, so re-running reproduces the
     exact same 10 cases.

Writes the selection to atlas/atlas_selection.json. This file, once written,
is the source of truth for "which 10 cases are atlases" -- scripts/notebook
cells should read it rather than recomputing the split, so swapping in real
doctor-segmented atlases later is a one-file change.

Usage:
    python scripts/select_atlas_cases.py --data-dir "../Dataset060_Merged_Def"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import nibabel as nib
import numpy as np

NUM_ORGANS = 12  # organ classes 1..12; 0 is background
SOURCE_PATTERN = re.compile(r"^([a-zA-Z]+)")


def build_train_files(data_dir: str, val_fraction: float = 0.2, seed: int = 42):
    """Mirrors the notebook's data-prep cell exactly, but returns the TRAIN portion."""
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

    data_dicts = [
        {"image": os.path.abspath(img), "label": os.path.abspath(lbl)}
        for img, lbl in zip(filtered_image_files, paired_label_files)
    ]
    random.seed(seed)
    random.shuffle(data_dicts)

    train_split = int(len(data_dicts) * (1 - val_fraction))
    return data_dicts[:train_split]


def source_of(case: dict) -> str:
    name = os.path.basename(case["label"])
    m = SOURCE_PATTERN.match(name)
    return m.group(1) if m else "unknown"


def organs_present(label_path: str) -> set[int]:
    data = nib.load(label_path).get_fdata()
    ids = set(np.unique(data).astype(int).tolist())
    ids.discard(0)
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=r"C:\\Users\\Asus\\OneDrive\\Desktop\\Atlas Based Medical Segmentation\\Dataset060_Merged_Def")
    parser.add_argument("--atlas-count", type=int, default=10)
    parser.add_argument("--out", default=os.path.join(_REPO_ROOT, "atlas", "atlas_selection.json"))
    args = parser.parse_args()

    train_files = build_train_files(args.data_dir)
    print(f"Training pool: {len(train_files)} cases")

    by_source = defaultdict(list)
    for case in train_files:
        by_source[source_of(case)].append(case)
    print("Source breakdown in training pool:", {k: len(v) for k, v in by_source.items()})

    print(f"Scanning labels for full organ coverage (all {NUM_ORGANS} organs present)...")
    complete_by_source = defaultdict(list)
    organ_count_histogram = defaultdict(int)
    for i, case in enumerate(train_files):
        ids = organs_present(case["label"])
        organ_count_histogram[len(ids)] += 1
        if len(ids) == NUM_ORGANS:
            complete_by_source[source_of(case)].append(case)
        if (i + 1) % 50 == 0:
            print(f"  scanned {i + 1}/{len(train_files)}...")

    print("\nOrgan-count histogram across training pool (organs present -> num cases):")
    for k in sorted(organ_count_histogram):
        print(f"  {k:2d} organs: {organ_count_histogram[k]} cases")

    print("\nCases with all 12 organs, by source:", {k: len(v) for k, v in complete_by_source.items()})

    sources = sorted(complete_by_source.keys())
    if not sources:
        raise RuntimeError("No case in the training pool has all 12 organs present -- selection criteria need revisiting.")

    # Deterministic round-robin over sources (sorted name, sorted case order within
    # each source) until atlas_count is reached.
    for src in sources:
        complete_by_source[src].sort(key=lambda c: c["label"])

    selection = []
    idx_per_source = {src: 0 for src in sources}
    src_cycle = sources[:]
    while len(selection) < args.atlas_count:
        progressed = False
        for src in src_cycle:
            if len(selection) >= args.atlas_count:
                break
            i = idx_per_source[src]
            if i < len(complete_by_source[src]):
                selection.append(complete_by_source[src][i])
                idx_per_source[src] += 1
                progressed = True
        if not progressed:
            raise RuntimeError(
                f"Only found {len(selection)} eligible cases with all {NUM_ORGANS} organs present, "
                f"need {args.atlas_count}."
            )

    print(f"\nSelected {len(selection)} atlas cases:")
    for c in selection:
        print(f"  {os.path.basename(c['image'])}  (source={source_of(c)})")

    out_payload = {
        "atlas_count": len(selection),
        "num_organs": NUM_ORGANS,
        "selection_criteria": "all 12 organs present; round-robin balance across source prefixes; drawn only from the training split (seed=42, val_fraction=0.2), never from validation",
        "cases": selection,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nWrote atlas selection to {args.out}")


if __name__ == "__main__":
    main()
