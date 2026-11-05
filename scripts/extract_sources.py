"""
Split the merged dataset (Dataset060_Merged_Def) back into its sources, by case-name prefix:

    amos<NNNN>   AMOS22                    -> Dataset061_AMOS
    img<NNNN>    BTCV / Synapse abdomen    -> Dataset062_Synapse   (img0001..img0040; 30 labelled training cases)
    s<NNNN>      1.5 mm isotropic, source not documented (probably TotalSegmentator); left alone

Each output folder is nnU-Net style (<case>_0000.nii.gz + <case>.nii.gz + dataset.json with the same 12 labels) and a
manifest <name>.json (case names, source, atlas cases among them). Files are hard-linked, so no extra disk space is used
(a plain copy is made when linking is impossible, e.g. another drive). The merged dataset is never modified.

    python scripts/extract_sources.py --dry-run
    python scripts/extract_sources.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data.splits import DEFAULT_DATA_DIR, atlas_names, case_name, paired_cases  # noqa: E402

SOURCES = {"amos": ("Dataset061_AMOS", "AMOS22 abdominal CT"),
           "img": ("Dataset062_Synapse", "Synapse / BTCV abdominal CT (Multi-Atlas Labeling Beyond the Cranial Vault)")}


def source_of(name: str) -> str:
    return re.match(r"[A-Za-z]+", name).group().lower()


def link_or_copy(src: str, dst: str) -> str:
    if os.path.exists(dst):
        return "exists"
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", default=None, help="parent of the new dataset folders (default: next to the merged dataset)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    out_parent = a.out_dir or os.path.dirname(os.path.abspath(a.data_dir))
    merged = json.load(open(os.path.join(a.data_dir, "dataset.json"), encoding="utf-8"))
    atlases = atlas_names()
    cases = paired_cases(a.data_dir)
    counts = {}
    for c in cases:
        s = source_of(case_name(c["image"]))
        counts[s] = counts.get(s, 0) + 1
    print("sources in", a.data_dir, counts)

    for prefix, (folder, desc) in SOURCES.items():
        mine = [c for c in cases if source_of(case_name(c["image"])) == prefix]
        names = [case_name(c["image"]) for c in mine]
        in_atlas = sorted(set(names) & atlases)
        dest = os.path.join(out_parent, folder)
        print(f"{folder}: {len(mine)} cases ({desc}); atlas cases among them: {in_atlas or 'none'} -> {dest}")
        if a.dry_run:
            continue
        os.makedirs(dest, exist_ok=True)
        how = {}
        for c in mine:
            for k in ("image", "label"):
                r = link_or_copy(c[k], os.path.join(dest, os.path.basename(c[k])))
                how[r] = how.get(r, 0) + 1
        ds = {**merged, "name": folder, "description": desc, "numTraining": len(mine)}
        json.dump(ds, open(os.path.join(dest, "dataset.json"), "w", encoding="utf-8"), indent=1)
        json.dump({"source": desc, "prefix": prefix, "num_cases": len(mine), "cases": names, "atlas_cases": in_atlas},
                  open(os.path.join(out_parent, f"{folder}_manifest.json"), "w", encoding="utf-8"), indent=1)
        print("   files:", how)


if __name__ == "__main__":
    main()
