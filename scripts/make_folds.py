"""
K-fold split of every usable case (the whole dataset minus the 10 atlases, and minus any case without a cached
registration when --require-cache is given). Cases are grouped by source (amos / img / s, the prefix of the case name),
shuffled with a fixed seed inside each group and dealt out round-robin, so every fold has the same source mix.

Writes <out>/fold_<k>_chunk.json with the keys run.py and the trainer read ("train", "val": case names). Existing fold
files are never overwritten (every rerun must see the same folds) unless --force.

    python scripts/make_folds.py --k 5 --seed 42 --out experiments/exp_cv5 --require-cache
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data.splits import (DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, atlas_names, case_name, paired_cases,  # noqa: E402
                         warped_label_path)


def make_folds(names: list, k: int, seed: int) -> list:
    """List of k validation-name lists (disjoint, covering `names`), stratified by source prefix."""
    groups = {}
    for n in sorted(names):
        groups.setdefault(re.split(r"[_\d]", n)[0], []).append(n)
    folds = [[] for _ in range(k)]
    offset = 0
    for src in sorted(groups):
        members = groups[src]
        random.Random(f"{seed}-{src}").shuffle(members)
        for i, n in enumerate(members):
            folds[(i + offset) % k].append(n)
        offset += len(members)  # continue the deal where the last source stopped, so fold sizes differ by at most 1
    return [sorted(f) for f in folds]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join(REPO, "experiments", "exp_cv5"))
    p.add_argument("--require-cache", action="store_true", help="drop cases without a cached atlas registration")
    p.add_argument("--force", action="store_true", help="overwrite existing fold files")
    a = p.parse_args()

    atlases = atlas_names()
    cases = [c for c in paired_cases(DEFAULT_DATA_DIR) if case_name(c["image"]) not in atlases]
    dropped = []
    if a.require_cache:
        dropped = [case_name(c["image"]) for c in cases if not os.path.exists(warped_label_path(c, "tps", DEFAULT_CACHE_DIR))]
        cases = [c for c in cases if case_name(c["image"]) not in set(dropped)]
    names = [case_name(c["image"]) for c in cases]
    folds = make_folds(names, a.k, a.seed)
    assert sorted(sum(folds, [])) == sorted(names) and len(set(sum(folds, []))) == len(names)  # disjoint and complete
    assert not (set(names) & atlases)

    os.makedirs(a.out, exist_ok=True)
    for j, val in enumerate(folds):
        path = os.path.join(a.out, f"fold_{j}_chunk.json")
        if os.path.exists(path) and not a.force:
            print(f"fold {j}: {path} exists, kept")
            continue
        vs = set(val)
        json.dump({"seed": a.seed, "fold": j, "k": a.k, "train": [n for n in names if n not in vs], "val": val,
                   "excluded_no_registration": dropped}, open(path, "w", encoding="utf-8"), indent=1)
        print(f"fold {j}: {len(names) - len(val)} train / {len(val)} val -> {path}")
    if dropped:
        print(f"{len(dropped)} cases without a registration cache were left out: {dropped[:8]}{' ...' if len(dropped) > 8 else ''}")


if __name__ == "__main__":
    main()
