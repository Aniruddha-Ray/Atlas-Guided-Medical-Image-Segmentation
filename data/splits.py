"""
The train/validation split and atlas bookkeeping, in one place.

The split reproduces the notebook exactly: pair *_0000.nii.gz images with
labels, random.seed(42), shuffle, first 80% train / last 20% validation.
Changing it would make every result incomparable with the epoch-101 baseline.

Atlas cases (atlas/atlas_selection.json) come from the training portion and
are removed from the training targets, so an atlas never gets a supervised
loss on its own label while also serving as a prior source. Validation is
untouched: it is the same 136 cases as the baseline.
"""
from __future__ import annotations

import glob
import json
import os
import random

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(_REPO_ROOT), "Dataset060_Merged_Def")
DEFAULT_ATLAS_SELECTION = os.path.join(_REPO_ROOT, "atlas", "atlas_selection.json")
DEFAULT_CACHE_DIR = os.path.join(_REPO_ROOT, "cache")


def case_name(path: str) -> str:
    return os.path.basename(path).replace("_0000.nii.gz", "").replace(".nii.gz", "")


def paired_cases(data_dir: str = DEFAULT_DATA_DIR) -> list:
    image_files = sorted(glob.glob(os.path.join(data_dir, "*_0000.nii.gz")))
    label_files = set(glob.glob(os.path.join(data_dir, "*.nii.gz")))
    pairs = []
    for img in image_files:
        lbl = os.path.join(data_dir, os.path.basename(img).replace("_0000.nii.gz", ".nii.gz"))
        if lbl in label_files:
            pairs.append({"image": os.path.abspath(img), "label": os.path.abspath(lbl)})
    return pairs


def train_val_split(data_dir: str = DEFAULT_DATA_DIR, val_fraction: float = 0.2, seed: int = 42):
    pairs = paired_cases(data_dir)
    random.seed(seed)
    random.shuffle(pairs)
    cut = int(len(pairs) * (1 - val_fraction))
    return pairs[:cut], pairs[cut:]


def atlas_names(selection_path: str = DEFAULT_ATLAS_SELECTION) -> set:
    with open(selection_path, encoding="utf-8") as f:
        return {case_name(c["image"]) for c in json.load(f)["cases"]}


def split_with_atlases_removed(data_dir: str = DEFAULT_DATA_DIR, selection_path: str = DEFAULT_ATLAS_SELECTION):
    """(train_targets, val_targets). Raises if an atlas appears in validation."""
    train, val = train_val_split(data_dir)
    atlases = atlas_names(selection_path)
    leaked = atlases & {case_name(c["image"]) for c in val}
    if leaked:
        raise RuntimeError(f"atlas case(s) found in the validation split: {sorted(leaked)}")
    return [c for c in train if case_name(c["image"]) not in atlases], val


def warped_label_path(case: dict, variant: str = "tps", cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    return os.path.join(cache_dir, "warped_labels", variant, f"{case_name(case['image'])}.nii.gz")


def attach_atlas_priors(cases: list, variant: str = "tps", cache_dir: str = DEFAULT_CACHE_DIR):
    """Adds "atlas_labels" (the cached warped-atlas stack) to each case. Returns (with_cache, missing_names)."""
    with_cache, missing = [], []
    for c in cases:
        p = warped_label_path(c, variant, cache_dir)
        if os.path.exists(p):
            with_cache.append({**c, "atlas_labels": p})
        else:
            missing.append(case_name(c["image"]))
    return with_cache, missing
