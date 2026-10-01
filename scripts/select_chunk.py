"""
Picks a small, reproducible chunk of cases for a local from-scratch run.

  - training cases come from the non-atlas training targets, validation cases from the
    official validation split (so the 10 atlases never serve as targets and no validation
    case is trained on)
  - stratified by source prefix (amos / s / img), each source capped at its quota
  - a case qualifies only if all 12 organs are present, each above --min-ml millilitres
    (a 1-voxel "organ" is annotation noise and would make per-organ Dice meaningless)
  - cases whose registration is already cached are taken first (a registration is ~3.5 min)
  - seeded shuffle, so the same arguments give the same chunk

Writes experiments/<experiment>/chunk.json.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import nibabel as nib
import numpy as np

from data.splits import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, case_name, split_with_atlases_removed, warped_label_path
from scripts.select_atlas_cases import source_of

NUM_ORGANS = 12


def organ_volumes_ml(label_path: str) -> dict:
    img = nib.load(label_path)
    voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    ids, counts = np.unique(np.asarray(img.dataobj).astype(np.int64), return_counts=True)
    return {int(i): float(c) * voxel_ml for i, c in zip(ids, counts) if i > 0}


def qualifies(case: dict, min_ml: float):
    vols = organ_volumes_ml(case["label"])
    ok = all(vols.get(o, 0.0) >= min_ml for o in range(1, NUM_ORGANS + 1))
    return ok, vols


def pick(pool: list, quota: dict, min_ml: float, seed: int, cache_dir: str, tag: str) -> list:
    by_src = defaultdict(list)
    for c in pool:
        by_src[source_of(c)].append(c)
    rng = random.Random(seed)
    chosen = []
    for src in sorted(quota):
        cand = sorted(by_src[src], key=lambda c: c["label"])
        rng.shuffle(cand)
        cand.sort(key=lambda c: not os.path.exists(warped_label_path(c, "tps", cache_dir)))  # cached first, stable
        taken = 0
        for c in cand:
            if taken >= quota[src]:
                break
            ok, vols = qualifies(c, min_ml)
            if ok:
                chosen.append({**c, "source": src, "min_organ_ml": round(min(vols[o] for o in range(1, NUM_ORGANS + 1)), 2)})
                taken += 1
        print(f"  {tag} {src}: {taken}/{quota[src]} (pool {len(cand)})", flush=True)
        if taken < quota[src]:
            raise SystemExit(f"only {taken} qualifying {src} cases for the {tag} quota of {quota[src]}")
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--experiment", default="exp_02_prototype_v2")
    p.add_argument("--train-quota", default="amos=10,s=11,img=3")
    p.add_argument("--val-quota", default="amos=2,s=3,img=1")
    p.add_argument("--min-ml", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    parse = lambda s: {k: int(v) for k, v in (kv.split("=") for kv in s.split(","))}
    train_pool, val_pool = split_with_atlases_removed(a.data_dir)
    train = pick(train_pool, parse(a.train_quota), a.min_ml, a.seed, a.cache_dir, "train")
    val = pick(val_pool, parse(a.val_quota), a.min_ml, a.seed, a.cache_dir, "val")

    names = lambda cs: [case_name(c["image"]) for c in cs]
    need = [n for c, n in zip(train + val, names(train) + names(val))
            if not os.path.exists(warped_label_path(c, "tps", a.cache_dir))]
    out = {"seed": a.seed, "min_organ_ml": a.min_ml, "train_quota": a.train_quota, "val_quota": a.val_quota,
           "train": names(train), "val": names(val), "need_registration": need,
           "details": {case_name(c["image"]): {"source": c["source"], "min_organ_ml": c["min_organ_ml"]} for c in train + val}}
    out_dir = os.path.join(_REPO_ROOT, "experiments", a.experiment)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chunk.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"train {len(train)}, val {len(val)}, still to register: {len(need)}")
    print("train:", names(train))
    print("val  :", names(val))


if __name__ == "__main__":
    main()
