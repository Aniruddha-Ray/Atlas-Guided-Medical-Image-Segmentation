"""
Build the two nnU-Net raw datasets for the atlas-prior study, from the same cases:

  Dataset<prior-id>_<name>Prior   CT (_0000) + prior channels (_0001..), for nnunet_ext/nnUNetTrainerAtlasPrior.py
  Dataset<image-id>_<name>Image   CT only, for the stock nnUNetTrainer (the image-only baseline)

Prior channels (layout "pnw", 25 channels for 12 organs): P_1..P_K, N_k = 1 - D_k, w; computed from the cached
warped-atlas stack (2 mm registration grid) exactly as data/transforms.AtlasPriorsd computes them, resampled onto the
CT's own grid (linear; outside the atlas grid 0 = "no organ, far, no confidence"), stored as uint8 round(x * 250).
"pw" leaves the nearness maps out (13 channels).

Grid: every axis finer than --min-spacing (default 1.5 mm) is coarsened to it, for CT (int16), label and priors alike,
because nnU-Net holds all channels of a case at input resolution in float32 (26 channels at ~0.7 mm in-plane: ~4 GB of
RAM per case). This departs from nnU-Net's native-resolution default, equally for both datasets. --min-spacing 0 keeps
the CT grid and hard-links CT and label. The image-only dataset hard-links the prior dataset's CT and label files, so
both arms see identical images. Cases already written are skipped, so the script can be rerun after an interruption.

Splits (written as splits_final.json into both preprocessed folders, never overwritten without --force):
  fold 0         the chunk's train / val
  folds 1..      nested subsets of the chunk's training cases (--subsets, fixed shuffle --seed), same val cases:
                 the learning curve (Dice versus number of labelled cases)

    python scripts/nnunet_make_datasets.py --limit 3                     # try it on 3 cases
    python scripts/nnunet_make_datasets.py --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from data.splits import DEFAULT_CACHE_DIR, DEFAULT_DATA_DIR, case_name, paired_cases, warped_label_path  # noqa: E402

NNUNET_ROOT = os.environ.get("NNUNET_ROOT", os.path.join(os.path.expanduser("~"), "nnunet_data"))
PRIOR_SCALE = 250.0
NUM_CLASSES = 13


def prior_channels(stack_path: str, layout: str, num_classes: int = NUM_CLASSES):
    """(channels (n, z, y, x) float32 on the stack grid, the stack's 3D sitk geometry image)."""
    import SimpleITK as sitk
    from data.transforms import atlas_prior_maps

    st4 = sitk.ReadImage(stack_path)
    size = list(st4.GetSize())
    vols = [sitk.Extract(st4, size[:3] + [0], [0, 0, 0, i]) for i in range(size[3])]
    lab = np.stack([sitk.GetArrayFromImage(v) for v in vols])  # (N, z, y, x)
    P, D, w = atlas_prior_maps(lab, num_classes, spacing=vols[0].GetSpacing()[::-1])
    parts = [P[1:], 1.0 - D, w] if layout == "pnw" else [P[1:], w]
    return np.concatenate(parts).astype(np.float32), vols[0]


def output_grid(ct_path: str, min_spacing: float):
    """(size, origin, spacing, direction) of the grid everything is written on: the CT's grid, with every axis finer
    than min_spacing coarsened to min_spacing (same physical extent and corner). min_spacing <= 0: the CT's grid."""
    import SimpleITK as sitk

    r = sitk.ImageFileReader()
    r.SetFileName(ct_path)
    r.ReadImageInformation()
    size, spacing = list(r.GetSize()), list(r.GetSpacing())
    if min_spacing <= 0:
        return size, r.GetOrigin(), spacing, r.GetDirection()
    new_sp = [max(s, min_spacing) for s in spacing]
    size = [max(1, int(round(n * s / t))) for n, s, t in zip(size, spacing, new_sp)]
    # keep the physical corner: the first voxel centre moves by half the change in voxel size
    d = np.array(r.GetDirection()).reshape(3, 3)
    shift = d @ ((np.array(new_sp) - np.array(spacing)) / 2.0)
    return size, tuple(np.array(r.GetOrigin()) + shift), new_sp, r.GetDirection()


def _resampler(grid, interpolator, default, pixel_type):
    import SimpleITK as sitk

    size, origin, spacing, direction = grid
    rs = sitk.ResampleImageFilter()
    rs.SetSize([int(x) for x in size]); rs.SetOutputOrigin(tuple(float(x) for x in origin))
    rs.SetOutputSpacing(tuple(float(x) for x in spacing)); rs.SetOutputDirection(direction)
    rs.SetInterpolator(interpolator); rs.SetDefaultPixelValue(default); rs.SetOutputPixelType(pixel_type)
    return rs


def write_image_and_label(ct_path: str, label_path: str, out_image: str, out_label: str, grid):
    """CT (linear, int16 HU) and label (nearest, uint8) on `grid`."""
    import SimpleITK as sitk

    ct = _resampler(grid, sitk.sitkLinear, -1024.0, sitk.sitkFloat32).Execute(sitk.ReadImage(ct_path, sitk.sitkFloat32))
    arr = np.clip(np.rint(sitk.GetArrayFromImage(ct)), -32768, 32767).astype(np.int16)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ct)
    sitk.WriteImage(out, out_image, useCompression=True)
    lab = _resampler(grid, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8).Execute(sitk.ReadImage(label_path, sitk.sitkUInt8))
    sitk.WriteImage(lab, out_label, useCompression=True)


def write_prior_files(stack_path: str, ct_path: str, out_prefix: str, layout: str, min_spacing: float = 0.0) -> int:
    """Writes <out_prefix>_0001.nii.gz ... on output_grid(ct_path, min_spacing). Returns the number of channels."""
    import SimpleITK as sitk

    chans, src = prior_channels(stack_path, layout)
    rs = _resampler(output_grid(ct_path, min_spacing), sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    for j, ch in enumerate(chans, start=1):
        img = sitk.GetImageFromArray(ch)
        img.CopyInformation(src)
        res = rs.Execute(img)
        q = sitk.GetImageFromArray(np.clip(np.rint(sitk.GetArrayFromImage(res) * PRIOR_SCALE), 0, 255).astype(np.uint8))
        q.CopyInformation(res)
        sitk.WriteImage(q, f"{out_prefix}_{j:04d}.nii.gz", useCompression=True)
    return len(chans)


def link(src: str, dst: str):
    if not os.path.exists(dst):
        try:
            os.link(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)


def build_case(case: dict, stack: str, prior_dir: str, image_dir: str, layout: str, n_ch: int, min_spacing: float):
    name = case_name(case["image"])
    img_p = os.path.join(prior_dir, "imagesTr", f"{name}_0000.nii.gz")
    lab_p = os.path.join(prior_dir, "labelsTr", f"{name}.nii.gz")
    prefix = os.path.join(prior_dir, "imagesTr", name)
    secs = 0.0
    if not (os.path.exists(f"{prefix}_{n_ch:04d}.nii.gz") and os.path.exists(img_p) and os.path.exists(lab_p)):
        t = time.time()
        tmp = prefix + ".partial"  # write under a temporary prefix, then rename: an interrupted case is redone
        if min_spacing > 0:
            write_image_and_label(case["image"], case["label"], tmp + "_img.nii.gz", tmp + "_lab.nii.gz",
                                  output_grid(case["image"], min_spacing))
        got = write_prior_files(stack, case["image"], tmp, layout, min_spacing)
        assert got == n_ch, (got, n_ch)
        for j in range(1, n_ch + 1):
            os.replace(f"{tmp}_{j:04d}.nii.gz", f"{prefix}_{j:04d}.nii.gz")
        if min_spacing > 0:
            os.replace(tmp + "_img.nii.gz", img_p)
            os.replace(tmp + "_lab.nii.gz", lab_p)
        else:
            link(case["image"], img_p)
            link(case["label"], lab_p)
        secs = time.time() - t
    # the image-only dataset shares the very same CT and label files
    link(img_p, os.path.join(image_dir, "imagesTr", f"{name}_0000.nii.gz"))
    link(lab_p, os.path.join(image_dir, "labelsTr", f"{name}.nii.gz"))
    return name, secs


def dataset_json(src_json: dict, n_prior: int, name: str, n_cases: int, desc: str) -> dict:
    ch = {"0": "CT", **{str(j): "noNorm" for j in range(1, n_prior + 1)}}
    return {"channel_names": ch, "labels": src_json["labels"], "numTraining": n_cases, "file_ending": ".nii.gz",
            "name": name, "description": desc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-file", default=os.path.join(REPO, "experiments", "exp_04_pipeline", "chunk_150_50_e100_chunk.json"))
    p.add_argument("--name", default="Chunk150")
    p.add_argument("--prior-id", type=int, default=71)
    p.add_argument("--image-id", type=int, default=72)
    p.add_argument("--layout", choices=("pnw", "pw"), default="pnw")
    p.add_argument("--subsets", default="10,25,50,100", help="learning-curve training sizes (folds 1, 2, ...)")
    p.add_argument("--seed", type=int, default=0, help="shuffle of the training cases for the nested subsets")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--variant", default="tps")
    p.add_argument("--nnunet-root", default=NNUNET_ROOT)
    p.add_argument("--min-spacing", type=float, default=1.5,
                   help="write CT, label and priors on the CT grid coarsened to at least this many mm per axis (both "
                        "datasets alike); 0 = the CT's own grid (hard links)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=None, help="only the first N cases (a trial run)")
    p.add_argument("--force", action="store_true", help="overwrite existing splits_final.json")
    a = p.parse_args()

    chunk = json.load(open(a.chunk_file, encoding="utf-8"))
    by_name = {case_name(c["image"]): c for c in paired_cases(a.data_dir)}
    train, val = list(chunk["train"]), list(chunk["val"])
    names = (train + val)[: a.limit] if a.limit else train + val
    missing = [n for n in names if not os.path.exists(warped_label_path(by_name[n], a.variant, a.cache_dir))]
    if missing:
        raise SystemExit(f"{len(missing)} cases have no registration cache, e.g. {missing[:5]}")

    raw = os.path.join(a.nnunet_root, "nnUNet_raw")
    prior_dir = os.path.join(raw, f"Dataset{a.prior_id:03d}_{a.name}Prior")
    image_dir = os.path.join(raw, f"Dataset{a.image_id:03d}_{a.name}Image")
    for d in (prior_dir, image_dir):
        os.makedirs(os.path.join(d, "imagesTr"), exist_ok=True)
        os.makedirs(os.path.join(d, "labelsTr"), exist_ok=True)
    n_ch = 2 * (NUM_CLASSES - 1) + 1 if a.layout == "pnw" else NUM_CLASSES
    src_json = json.load(open(os.path.join(a.data_dir, "dataset.json"), encoding="utf-8"))
    json.dump(dataset_json(src_json, n_ch, os.path.basename(prior_dir), len(names),
                           f"CT + atlas priors ({a.layout}, uint8 x{PRIOR_SCALE:g}) from {os.path.basename(a.chunk_file)}"),
              open(os.path.join(prior_dir, "dataset.json"), "w", encoding="utf-8"), indent=1)
    json.dump(dataset_json(src_json, 0, os.path.basename(image_dir), len(names),
                           f"CT only, same cases as {os.path.basename(prior_dir)}"),
              open(os.path.join(image_dir, "dataset.json"), "w", encoding="utf-8"), indent=1)

    print(f"{len(names)} cases -> {prior_dir} ({n_ch} prior channels) and {image_dir}", flush=True)
    t0 = time.time()
    jobs = [(by_name[n], warped_label_path(by_name[n], a.variant, a.cache_dir)) for n in names]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(build_case, c, s, prior_dir, image_dir, a.layout, n_ch, a.min_spacing) for c, s in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            name, secs = f.result()
            print(f"  [{i}/{len(futs)}] {name}" + (f" {secs:.1f} s" if secs else " (already there)"), flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)

    # splits: fold 0 = the chunk split, then nested learning-curve subsets with the same validation cases
    in_set = set(names)
    tr, va = [n for n in train if n in in_set], [n for n in val if n in in_set]
    order = sorted(tr)
    random.Random(a.seed).shuffle(order)
    sizes = [int(s) for s in a.subsets.split(",") if s.strip()] if a.subsets else []
    folds = [{"train": tr, "val": va}] + [{"train": sorted(order[:k]), "val": va} for k in sizes if k <= len(order)]
    info = {"chunk_file": a.chunk_file, "seed": a.seed,
            "folds": [{"fold": i, "n_train": len(f["train"]), "n_val": len(f["val"])} for i, f in enumerate(folds)]}
    for d in (prior_dir, image_dir):
        pre = os.path.join(a.nnunet_root, "nnUNet_preprocessed", os.path.basename(d))
        os.makedirs(pre, exist_ok=True)
        path = os.path.join(pre, "splits_final.json")
        if os.path.exists(path) and not a.force:
            print(f"kept existing {path}")
        else:
            json.dump(folds, open(path, "w", encoding="utf-8"), indent=1)
            print(f"wrote {path}")
        json.dump(info, open(os.path.join(d, "splits_info.json"), "w", encoding="utf-8"), indent=1)
    print("folds:", [(f["fold"], f["n_train"]) for f in info["folds"]], "| val", len(va))


if __name__ == "__main__":
    main()
