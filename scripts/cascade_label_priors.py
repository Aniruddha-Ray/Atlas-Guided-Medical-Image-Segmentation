"""
Stage-2 prior channels of the cascade study, from the stage-1 predictions (atlas/label_registration.py).

Per case (the 200 stage-2 AMOS cases and the 30 BTCV cases; all predicted by the stage-1 model, which saw none of them):
  arm C  Dataset083_CascadeAtlas/imagesTr/<case>_0001..0025   30 atlases registered label-to-label to the prediction,
                                                               anatomy-weighted top-k fusion
  arm B  Dataset082_CascadePlain/imagesTr/<case>_0001..0025   the same channels from the prediction itself
  QA     experiments/exp_cascade/label_reg/<case>.json         per-atlas fit, chosen atlases and weights, run time;
                                                               "evaluation_only": Dice of the stage-1 prediction and of
                                                               the fused prior's argmax against the ground truth (read
                                                               after everything is written, never used to build it)
Channels are uint8 round(x * 250) on the prediction's grid (= the CT grid of the raw datasets). Finished cases are
skipped, so the script can be rerun after an interruption.

    python scripts/cascade_label_priors.py --limit 2          # time and check two cases
    python scripts/cascade_label_priors.py --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from nnunet_ext.nn import DEFAULT_ROOT  # noqa: E402

SPLIT = os.path.join(REPO, "experiments", "exp_cascade", "cascade_split.json")
SELECTION = os.path.join(REPO, "atlas", "atlas_selection_amos30.json")
QA_DIR = os.path.join(REPO, "experiments", "exp_cascade", "label_reg")
PRIOR_SCALE = 250.0
_ATLASES = None


def _init(selection: str, cfg_kw: dict, threads: int):
    import SimpleITK as sitk

    from atlas.label_registration import LabelRegConfig, prepare_atlas
    from data.splits import case_name

    global _ATLASES
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(threads)
    cfg = LabelRegConfig(**cfg_kw)
    cases = json.load(open(selection, encoding="utf-8"))["cases"]
    _ATLASES = (cfg, [prepare_atlas(case_name(c["image"]), sitk.ReadImage(c["label"]), cfg) for c in cases])


def write_channels(chans: np.ndarray, like, prefix: str):
    import SimpleITK as sitk

    for j, ch in enumerate(chans, start=1):
        img = sitk.GetImageFromArray(np.clip(np.rint(ch * PRIOR_SCALE), 0, 255).astype(np.uint8))
        img.CopyInformation(like)
        sitk.WriteImage(img, f"{prefix}_{j:04d}.nii.gz", useCompression=True)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int(np.logical_and(a, b).sum()) / s if s else float("nan")


def process(name: str, pred_path: str, ct_path: str, label_path: str, out_b: str, out_c: str, qa_path: str):
    import SimpleITK as sitk

    from atlas.label_registration import NUM_ORGANS, fuse_atlases, onehot_organs, prior_channels

    cfg, atlases = _ATLASES
    t0 = time.time()
    pred_img = sitk.ReadImage(pred_path, sitk.sitkUInt8)
    ct = sitk.ReadImage(ct_path)
    if pred_img.GetSize() != ct.GetSize() or not np.allclose(pred_img.GetOrigin(), ct.GetOrigin(), atol=1e-3):
        raise RuntimeError(f"{name}: prediction grid differs from the CT grid")
    pred_img.CopyInformation(ct)
    spacing = pred_img.GetSpacing()[::-1]
    pred = sitk.GetArrayFromImage(pred_img)

    tmp_b, tmp_c = out_b + ".partial", out_c + ".partial"
    write_channels(prior_channels(onehot_organs(pred), spacing), pred_img, tmp_b)
    P, info = fuse_atlases(pred_img, atlases, cfg)
    chans = prior_channels(P, spacing)
    write_channels(chans, pred_img, tmp_c)
    n = chans.shape[0]
    for j in range(1, n + 1):  # the last channel goes last, so its presence marks a finished case
        os.replace(f"{tmp_b}_{j:04d}.nii.gz", f"{out_b}_{j:04d}.nii.gz")
        os.replace(f"{tmp_c}_{j:04d}.nii.gz", f"{out_c}_{j:04d}.nii.gz")

    gt = sitk.GetArrayFromImage(sitk.ReadImage(label_path))  # evaluation only, after the priors are written
    fused = np.concatenate([np.clip(1.0 - P.sum(0, keepdims=True), 0, 1), P]).argmax(0)
    ev = {"stage1_prediction": {}, "fused_prior_argmax": {}}
    for k in range(1, NUM_ORGANS + 1):
        ev["stage1_prediction"][str(k)] = round(_dice(pred == k, gt == k), 4)
        ev["fused_prior_argmax"][str(k)] = round(_dice(fused == k, gt == k), 4)
    for key in ev:
        vals = [v for v in ev[key].values() if not np.isnan(v)]
        ev[key]["mean"] = round(float(np.mean(vals)), 4)
    info.update(case=name, seconds=round(time.time() - t0, 1), evaluation_only=ev)
    json.dump(info, open(qa_path, "w", encoding="utf-8"), indent=1)
    return name, info["seconds"], ev["stage1_prediction"]["mean"], ev["fused_prior_argmax"]["mean"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--pred-dir", default=os.path.join(DEFAULT_ROOT, "cascade", "stage1_pred"))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--threads", type=int, default=2, help="SimpleITK threads per worker")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cases", nargs="*", default=None)
    a = p.parse_args()

    split = json.load(open(SPLIT, encoding="utf-8"))
    names = a.cases or (split["stage2"] + split["test"])
    names = names[: a.limit] if a.limit else names
    raw = os.path.join(a.root, "nnUNet_raw")
    src = {n: os.path.join(raw, "Dataset073_AMOSPrior") for n in split["stage2"]}
    src.update({n: os.path.join(raw, "Dataset080_CascadeS1") for n in split["test"]})
    out_b, out_c = os.path.join(raw, "Dataset082_CascadePlain", "imagesTr"), os.path.join(raw, "Dataset083_CascadeAtlas", "imagesTr")
    for d in (out_b, out_c, QA_DIR):
        os.makedirs(d, exist_ok=True)

    todo = [n for n in names if not (os.path.exists(os.path.join(out_b, f"{n}_0025.nii.gz"))
                                     and os.path.exists(os.path.join(out_c, f"{n}_0025.nii.gz")))]
    lacking = [n for n in todo if not os.path.exists(os.path.join(a.pred_dir, f"{n}.nii.gz"))]
    if lacking:
        raise SystemExit(f"{len(lacking)} cases have no stage-1 prediction in {a.pred_dir}, e.g. {lacking[:3]}")
    print(f"{len(names)} cases, {len(names) - len(todo)} already done, {len(todo)} to do; {a.workers} workers", flush=True)
    t0, s1, fz = time.time(), [], []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(SELECTION, {"top_k": a.top_k}, a.threads)) as ex:
        futs = [ex.submit(process, n, os.path.join(a.pred_dir, f"{n}.nii.gz"),
                          os.path.join(src[n], "imagesTr", f"{n}_0000.nii.gz"), os.path.join(src[n], "labelsTr", f"{n}.nii.gz"),
                          os.path.join(out_b, n), os.path.join(out_c, n), os.path.join(QA_DIR, f"{n}.json")) for n in todo]
        for i, f in enumerate(as_completed(futs), 1):
            name, secs, d1, d2 = f.result()
            s1.append(d1); fz.append(d2)
            print(f"  [{i}/{len(todo)}] {name} {secs:.0f} s | Dice (evaluation only): stage-1 {d1:.3f}, "
                  f"fused atlas prior {d2:.3f}", flush=True)
    if todo:
        print(f"done in {(time.time() - t0) / 60:.1f} min; mean Dice stage-1 {np.mean(s1):.3f}, fused prior {np.mean(fz):.3f}")


if __name__ == "__main__":
    main()
