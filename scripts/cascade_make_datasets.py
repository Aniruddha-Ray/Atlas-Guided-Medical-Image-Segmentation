"""
nnU-Net raw datasets of the cascade study (split: experiments/exp_cascade/cascade_split.json, scripts/cascade_split.py).

  Dataset080_CascadeS1      stage 1: CT + 10-atlas CT-to-CT prior (25 channels, as Dataset073_AMOSPrior)
                            fold 0 = 42 stage-1 AMOS cases / 30 BTCV cases
                            imagesPred/ = the 200 stage-2 AMOS cases + the 30 BTCV cases, for stage-1 prediction
  Dataset081_CascadeImage   stage 2, arm A: CT only
  Dataset082_CascadePlain   stage 2, arm B: CT + the stage-1 prediction as prior channels (one-hot P, its nearness, w = 1)
  Dataset083_CascadeAtlas   stage 2, arm C: CT + 30 atlases registered label-to-label to the stage-1 prediction,
                            anatomy-weighted (scripts/cascade_label_priors.py writes the prior channels of B and C)
  stage-2 folds: 0 = 200 AMOS / 30 BTCV; 1, 2, 3 = nested subsets of 25, 50, 100 of the 200, same 30 BTCV.

Every file sits on the same coarsened grid as Dataset073_AMOSPrior (>= 1.5 mm, scripts/nnunet_make_datasets.py): AMOS
CT, labels and stage-1 priors are hard links to Dataset073's files; BTCV is written here the same way, from its
CT-to-CT registrations in cache_btcv/. All datasets later take Dataset073's plans, so CT normalisation, patch and batch
are identical for every model of the study.

    python scripts/cascade_make_datasets.py stage1
    python scripts/cascade_make_datasets.py stage2          # after scripts/cascade_label_priors.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data.splits import case_name, paired_cases, warped_label_path  # noqa: E402
from nnunet_ext.nn import DEFAULT_ROOT  # noqa: E402
from scripts.nnunet_make_datasets import build_case, dataset_json, link  # noqa: E402

SPLIT = os.path.join(REPO, "experiments", "exp_cascade", "cascade_split.json")
BTCV_DIR = os.path.join(os.path.dirname(REPO), "Dataset062_Synapse")
BTCV_CACHE = os.path.join(REPO, "cache_btcv")
SOURCE = "Dataset073_AMOSPrior"
N_PRIOR = 25
DATASETS = {"s1": "Dataset080_CascadeS1", "A": "Dataset081_CascadeImage", "B": "Dataset082_CascadePlain",
            "C": "Dataset083_CascadeAtlas"}
SUBSETS = (25, 50, 100)


def raw(root, key):
    return os.path.join(root, "nnUNet_raw", DATASETS[key])


def _dirs(d, *subs):
    for s in subs:
        os.makedirs(os.path.join(d, s), exist_ok=True)


def link_case(src_ds: str, dst_ds: str, name: str, channels: int, label: bool = True):
    """Hard-links CT (+ the first `channels` prior files) and the label of `name` from one raw dataset to another."""
    for j in range(channels + 1):
        link(os.path.join(src_ds, "imagesTr", f"{name}_{j:04d}.nii.gz"), os.path.join(dst_ds, "imagesTr", f"{name}_{j:04d}.nii.gz"))
    if label:
        link(os.path.join(src_ds, "labelsTr", f"{name}.nii.gz"), os.path.join(dst_ds, "labelsTr", f"{name}.nii.gz"))


def write_splits(root, key, folds):
    pre = os.path.join(root, "nnUNet_preprocessed", DATASETS[key])
    os.makedirs(pre, exist_ok=True)
    json.dump(folds, open(os.path.join(pre, "splits_final.json"), "w", encoding="utf-8"), indent=1)


def write_json(root, key, n_prior, n_cases, desc):
    src = json.load(open(os.path.join(BTCV_DIR, "dataset.json"), encoding="utf-8"))
    json.dump(dataset_json(src, n_prior, DATASETS[key], n_cases, desc),
              open(os.path.join(raw(root, key), "dataset.json"), "w", encoding="utf-8"), indent=1)


def stage1(root, split):
    src, s1, img = os.path.join(root, "nnUNet_raw", SOURCE), raw(root, "s1"), raw(root, "A")
    _dirs(s1, "imagesTr", "labelsTr", "imagesPred")
    _dirs(img, "imagesTr", "labelsTr")
    for n in split["stage1"]:
        link_case(src, s1, n, N_PRIOR)
    btcv = {case_name(c["image"]): c for c in paired_cases(BTCV_DIR)}
    missing = [n for n in split["test"] if not os.path.exists(warped_label_path(btcv[n], "tps", BTCV_CACHE))]
    if missing:
        raise SystemExit(f"{len(missing)} BTCV cases have no CT-to-CT registration in {BTCV_CACHE}, e.g. {missing[:3]}")
    for i, n in enumerate(split["test"], 1):
        _, secs = build_case(btcv[n], warped_label_path(btcv[n], "tps", BTCV_CACHE), s1, img, "pnw", N_PRIOR, 1.5)
        print(f"  BTCV [{i}/{len(split['test'])}] {n}" + (f" {secs:.1f} s" if secs else " (already there)"), flush=True)
    for n in split["stage2"]:  # inputs for predicting the stage-2 cases
        for j in range(N_PRIOR + 1):
            link(os.path.join(src, "imagesTr", f"{n}_{j:04d}.nii.gz"), os.path.join(s1, "imagesPred", f"{n}_{j:04d}.nii.gz"))
    for n in split["test"]:
        for j in range(N_PRIOR + 1):
            link(os.path.join(s1, "imagesTr", f"{n}_{j:04d}.nii.gz"), os.path.join(s1, "imagesPred", f"{n}_{j:04d}.nii.gz"))
    write_json(root, "s1", N_PRIOR, len(split["stage1"]) + len(split["test"]),
               "Cascade stage 1: CT + 10-atlas CT-to-CT prior; 42 AMOS training cases, BTCV validation")
    write_splits(root, "s1", [{"train": split["stage1"], "val": split["test"]}])
    print(f"stage 1: {len(split['stage1'])} train, {len(split['test'])} val, "
          f"{len(split['stage2']) + len(split['test'])} cases in imagesPred")


def stage2(root, split, seed):
    src, s1 = os.path.join(root, "nnUNet_raw", SOURCE), raw(root, "s1")
    for key in ("A", "B", "C"):
        _dirs(raw(root, key), "imagesTr", "labelsTr")
    for n in split["stage2"]:
        for key in ("A", "B", "C"):
            link_case(src, raw(root, key), n, 0)
    for n in split["test"]:
        for key in ("A", "B", "C"):
            link_case(s1, raw(root, key), n, 0)
    cases = split["stage2"] + split["test"]
    for key in ("B", "C"):
        lacking = [n for n in cases if not os.path.exists(os.path.join(raw(root, key), "imagesTr", f"{n}_{N_PRIOR:04d}.nii.gz"))]
        if lacking:
            raise SystemExit(f"arm {key}: {len(lacking)} cases have no prior channels yet (scripts/cascade_label_priors.py)")
    write_json(root, "A", 0, len(cases), "Cascade stage 2, arm A: CT only")
    write_json(root, "B", N_PRIOR, len(cases), "Cascade stage 2, arm B: CT + stage-1 prediction (one-hot P, nearness, w)")
    write_json(root, "C", N_PRIOR, len(cases), "Cascade stage 2, arm C: CT + 30 atlases registered label-to-label to the "
                                               "stage-1 prediction, anatomy-weighted top-10 fusion")
    order = sorted(split["stage2"])
    random.Random(seed).shuffle(order)
    folds = [{"train": split["stage2"], "val": split["test"]}] + [{"train": sorted(order[:k]), "val": split["test"]} for k in SUBSETS]
    for key in ("A", "B", "C"):
        write_splits(root, key, folds)
    print("stage 2: folds", [(i, len(f["train"])) for i, f in enumerate(folds)], "| val", len(split["test"]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=("stage1", "stage2"))
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--seed", type=int, default=0, help="shuffle for the nested stage-2 subsets")
    a = p.parse_args()
    split = json.load(open(SPLIT, encoding="utf-8"))
    stage1(a.root, split) if a.step == "stage1" else stage2(a.root, split, a.seed)


if __name__ == "__main__":
    main()
