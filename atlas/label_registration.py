"""
Label-to-label atlas registration and anatomy-weighted fusion: the stage-2 prior of the cascade study.

The target has no ground truth; its "label" is a first-stage segmentation (a stage-1 nnU-Net prediction). Every atlas
is fitted to that prediction, organ by organ, so the prior carries the atlas's anatomy (shape) placed where the
prediction says the organ is (pose and size):

  1. global   similarity transform from the organ centroids present in both (landmark least squares)
  2. organ    per organ an affine transform, refined from the global one on signed distance maps of the atlas organ and
              the predicted organ (mean squares); an affine keeps the atlas's shape, so prediction errors (holes, leaks,
              stray blobs) are not copied into the prior. A refinement that runs off (scale outside 0.5-2, more than
              40 mm from the global fit) falls back to the global transform.
              An organ missing from the prediction is placed by the global transform alone: the anatomy of the other
              organs tells where it should be.
  3. weights  per atlas and organ: Dice between the warped atlas organ and the predicted organ (for a missing organ: the
              atlas's mean Dice over the organs it could be compared on). Per organ the top-k atlases are fused,
              P_k = sum_a w_ak m_ak / sum_a w_ak, with m_ak the warped (soft) atlas organ mask.

Prior channels as in data/transforms.atlas_prior_maps: P_1..P_K, nearness N_k = 1 - min(d_k, 50 mm) / 50 mm with d_k the
distance to {P_k >= 0.5 * max P_k}, confidence w = 1 - H / log C over (P_bg, P_1..P_K) renormalised, P_bg = 1 - sum P_k.
The "plain cascade" ablation uses the same channels computed from the prediction itself (one-hot P, w = 1).

All transforms map target (fixed) points to atlas (moving) points, SimpleITK's convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

NUM_ORGANS = 12
SMALL_ORGANS = (4, 5, 10, 11, 12)  # gallbladder, esophagus, pancreas, adrenals: registered on a 2 mm grid
WORK_SPACING_MM = {"small": 2.0, "large": 3.0}  # large organs on 3 mm: about 4x faster, mean fit within 0.01 Dice
SDM_CLIP_MM = 20.0
DIST_CLIP_MM = 50.0


@dataclass
class LabelRegConfig:
    top_k: int = 10
    roi_margin_mm: float = 20.0
    min_organ_mm3: float = 160.0  # smaller predicted organs count as missing
    iterations: int = 50
    sampling: float = 0.1  # fraction of the fixed ROI's voxels sampled by the metric
    max_scale: float = 2.0
    max_shift_mm: float = 40.0


@dataclass
class AtlasOrgan:
    mask: sitk.Image  # float32 0/1 on a 2 or 3 mm crop around the organ (with margin)
    sdm: sitk.Image  # signed distance (negative inside), clipped to +-SDM_CLIP_MM, same crop
    centroid: np.ndarray  # physical LPS mm


@dataclass
class Atlas:
    name: str
    organs: dict = field(default_factory=dict)  # organ label -> AtlasOrgan


def resample_iso(img: sitk.Image, spacing: float, interpolator=sitk.sitkNearestNeighbor, default=0.0) -> sitk.Image:
    sp = np.array(img.GetSpacing())
    size = [max(1, int(round(n * s / spacing))) for n, s in zip(img.GetSize(), sp)]
    d = np.array(img.GetDirection()).reshape(3, 3)
    origin = np.array(img.GetOrigin()) + d @ ((spacing - sp) / 2.0)
    return sitk.Resample(img, size, sitk.Transform(), interpolator, tuple(origin), (spacing,) * 3, img.GetDirection(),
                         default, img.GetPixelID())


def largest_component(mask: sitk.Image) -> sitk.Image:
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(mask), sortByObjectSize=True)
    return sitk.Cast(cc == 1, sitk.sitkUInt8)


def _crop(mask: sitk.Image, margin_mm: float):
    """(cropped image, ok) around the nonzero voxels of a UInt8 mask, with a margin."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    if not stats.HasLabel(1):
        return None
    x0, y0, z0, sx, sy, sz = stats.GetBoundingBox(1)
    m = [int(np.ceil(margin_mm / s)) for s in mask.GetSpacing()]
    lo = [max(0, a - b) for a, b in zip((x0, y0, z0), m)]
    hi = [min(n, a + s + b) for a, s, b, n in zip((x0, y0, z0), (sx, sy, sz), m, mask.GetSize())]
    return sitk.RegionOfInterest(mask, [h - l for h, l in zip(hi, lo)], lo)


def _sdm(mask_u8: sitk.Image) -> sitk.Image:
    d = sitk.SignedMaurerDistanceMap(mask_u8, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
    return sitk.Clamp(sitk.Cast(d, sitk.sitkFloat32), sitk.sitkFloat32, -SDM_CLIP_MM, SDM_CLIP_MM)


def _centroid(mask_u8: sitk.Image):
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask_u8)
    return np.array(stats.GetCentroid(1)) if stats.HasLabel(1) else None


def prepare_organs(label: sitk.Image, margin_mm: float, min_mm3: float = 0.0, lcc: bool = False) -> dict:
    """organ -> AtlasOrgan on its working grid (largest component only when lcc)."""
    lab = sitk.Cast(label, sitk.sitkUInt8)
    iso = {key: resample_iso(lab, sp) for key, sp in WORK_SPACING_MM.items()}
    out = {}
    for k in range(1, NUM_ORGANS + 1):
        key = "small" if k in SMALL_ORGANS else "large"
        m = sitk.Cast(iso[key] == k, sitk.sitkUInt8)
        if lcc and sitk.GetArrayViewFromImage(m).any():
            m = largest_component(m)
        n = int(sitk.GetArrayViewFromImage(m).sum())
        if n == 0 or n * WORK_SPACING_MM[key] ** 3 < min_mm3:
            continue
        crop = _crop(m, margin_mm)
        out[k] = AtlasOrgan(mask=sitk.Cast(crop, sitk.sitkFloat32), sdm=_sdm(crop), centroid=_centroid(crop))
    return out


def prepare_atlas(name: str, label: sitk.Image, cfg: LabelRegConfig) -> Atlas:
    # the atlas crop gets a wider margin than the target ROI, so the moving image covers the fixed ROI after a shift
    return Atlas(name=name, organs=prepare_organs(label, cfg.roi_margin_mm + cfg.max_shift_mm / 2))


def global_transform(target: dict, atlas: Atlas) -> sitk.Transform:
    common = [k for k in target if k in atlas.organs]
    if len(common) >= 4:
        fixed = np.concatenate([target[k].centroid for k in common]).tolist()
        moving = np.concatenate([atlas.organs[k].centroid for k in common]).tolist()
        return sitk.LandmarkBasedTransformInitializer(sitk.Similarity3DTransform(), fixed, moving)
    tx = sitk.TranslationTransform(3)
    if common:
        tx.SetOffset(tuple(np.mean([atlas.organs[k].centroid - target[k].centroid for k in common], axis=0)))
    return tx


def _as_affine(tx: sitk.Transform) -> sitk.AffineTransform:
    """A similarity or translation transform as an AffineTransform (same mapping)."""
    a = sitk.AffineTransform(3)
    if isinstance(tx, sitk.TranslationTransform) or tx.GetName() == "TranslationTransform":
        a.SetTranslation(sitk.TranslationTransform(tx).GetOffset())
        return a
    s = sitk.Similarity3DTransform(tx)
    a.SetMatrix(s.GetMatrix())
    a.SetTranslation(s.GetTranslation())
    a.SetCenter(s.GetCenter())
    return a


def _plausible(tx: sitk.AffineTransform, init: sitk.Transform, centre: np.ndarray, cfg: LabelRegConfig) -> bool:
    sv = np.linalg.svd(np.array(tx.GetMatrix()).reshape(3, 3), compute_uv=False)
    if sv.min() < 1.0 / cfg.max_scale or sv.max() > cfg.max_scale:
        return False
    shift = np.linalg.norm(np.array(tx.TransformPoint(tuple(centre))) - np.array(init.TransformPoint(tuple(centre))))
    return shift <= cfg.max_shift_mm


def organ_transform(fixed: AtlasOrgan, moving: AtlasOrgan, init: sitk.Transform, cfg: LabelRegConfig):
    """(affine transform, refined) fitting the atlas organ to the predicted organ, starting from `init`."""
    start = _as_affine(init)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(cfg.sampling, seed=0)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=0.01, numberOfIterations=cfg.iterations,
                                                 relaxationFactor=0.7)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([2.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(sitk.AffineTransform(start), inPlace=False)
    try:
        res = reg.Execute(fixed.sdm, moving.sdm)  # a CompositeTransform holding the optimised affine
        res = sitk.CompositeTransform(res)
        out = sitk.AffineTransform(res.GetNthTransform(res.GetNumberOfTransforms() - 1))
    except RuntimeError:  # e.g. too few samples inside the moving crop
        return start, False
    if not _plausible(out, start, fixed.centroid, cfg):
        return start, False
    return out, True


def _target_box(atlas_organ: AtlasOrgan, tx: sitk.Transform, grid: sitk.Image, margin_mm: float):
    """(index lo, size) on `grid` covering the atlas organ crop mapped into target space, or None if outside."""
    inv = tx.GetInverse()
    m = atlas_organ.mask
    corners = [(i, j, k) for i in (0, m.GetSize()[0] - 1) for j in (0, m.GetSize()[1] - 1) for k in (0, m.GetSize()[2] - 1)]
    pts = np.array([inv.TransformPoint(m.TransformIndexToPhysicalPoint(c)) for c in corners])
    idx = np.array([grid.TransformPhysicalPointToContinuousIndex(tuple(p)) for p in pts])
    pad = np.array([margin_mm / s for s in grid.GetSpacing()])
    lo = np.maximum(np.floor(idx.min(0) - pad), 0).astype(int)
    hi = np.minimum(np.ceil(idx.max(0) + pad) + 1, grid.GetSize()).astype(int)
    if np.any(hi <= lo):
        return None
    return lo, hi - lo


def warp_organ(atlas_organ: AtlasOrgan, tx: sitk.Transform, grid: sitk.Image):
    """(soft mask array (z, y, x) on a sub-box of `grid`, numpy slices of that box) or None."""
    box = _target_box(atlas_organ, tx, grid, margin_mm=4.0)
    if box is None:
        return None
    lo, size = box
    ref = sitk.RegionOfInterest(grid, [int(s) for s in size], [int(v) for v in lo])
    out = sitk.Resample(atlas_organ.mask, ref, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    sl = tuple(slice(int(l), int(l + s)) for l, s in zip(lo[::-1], size[::-1]))
    return sitk.GetArrayFromImage(out), sl


def dice_on_box(soft: np.ndarray, sl, pred_mask: np.ndarray, pred_total: int) -> float:
    a = soft >= 0.5
    inter = int(np.logical_and(a, pred_mask[sl]).sum())
    denom = int(a.sum()) + pred_total
    return 2.0 * inter / denom if denom else 0.0


def prior_channels(P_org: np.ndarray, spacing_zyx) -> np.ndarray:
    """(2K + 1, z, y, x) float32: P_1..P_K, N_1..N_K, w, from organ probabilities P_org (K, z, y, x)."""
    P_org = np.clip(P_org, 0.0, 1.0).astype(np.float32)
    p_bg = np.clip(1.0 - P_org.sum(0, keepdims=True), 0.0, 1.0)
    q = np.concatenate([p_bg, P_org])
    q = q / np.clip(q.sum(0, keepdims=True), 1e-8, None)
    H = -(q * np.log(np.clip(q, 1e-8, None))).sum(0)
    conf = (1.0 - H / np.log(q.shape[0])).astype(np.float32)
    near = np.zeros_like(P_org)
    for k in range(P_org.shape[0]):
        peak = P_org[k].max()
        if peak <= 0:
            continue  # no organ anywhere: nearness 0 ("far") everywhere
        region = P_org[k] >= 0.5 * peak
        dt = distance_transform_edt(~region, sampling=spacing_zyx)
        near[k] = 1.0 - np.minimum(dt, DIST_CLIP_MM) / DIST_CLIP_MM
    return np.concatenate([P_org, near, conf[None]]).astype(np.float32)


def onehot_organs(pred: np.ndarray) -> np.ndarray:
    return np.stack([(pred == k) for k in range(1, NUM_ORGANS + 1)]).astype(np.float32)


def fuse_atlases(pred_img: sitk.Image, atlases: list, cfg: LabelRegConfig):
    """Anatomy-weighted fusion of all atlases fitted to the prediction `pred_img` (label map on the target grid).

    Returns (P_org (K, z, y, x) float32 on the target grid, info dict with per-atlas weights and QA)."""
    pred = sitk.GetArrayFromImage(pred_img).astype(np.uint8)
    target = prepare_organs(pred_img, cfg.roi_margin_mm, cfg.min_organ_mm3, lcc=True)
    pred_masks = {k: pred == k for k in range(1, NUM_ORGANS + 1)}
    pred_tot = {k: int(m.sum()) for k, m in pred_masks.items()}
    warped = {k: [] for k in range(1, NUM_ORGANS + 1)}  # organ -> [(atlas idx, weight or None, soft, slices)]
    qa = []
    for ai, atlas in enumerate(atlases):
        g = global_transform(target, atlas)
        refined, dices = 0, {}
        for k, a_org in atlas.organs.items():
            if k in target:
                tx, ok = organ_transform(target[k], a_org, g, cfg)
                refined += ok
            else:
                tx = g
            w = warp_organ(a_org, tx, pred_img)
            if w is None:
                continue
            soft, sl = w
            d = dice_on_box(soft, sl, pred_masks[k], pred_tot[k]) if k in target else None
            if d is not None:
                dices[k] = d
            warped[k].append([ai, d, (np.clip(soft, 0, 1) * 250).astype(np.uint8), sl])
        mean_d = float(np.mean(list(dices.values()))) if dices else 0.0
        for k in warped:  # an organ missing from the prediction takes the atlas's overall fit as its weight
            for item in warped[k]:
                if item[0] == ai and item[1] is None:
                    item[1] = mean_d
        qa.append({"atlas": atlas.name, "organs_refined": refined, "mean_dice_to_prediction": round(mean_d, 4),
                   "dice": {str(k): round(v, 4) for k, v in dices.items()}})

    P = np.zeros((NUM_ORGANS,) + pred.shape, dtype=np.float32)
    chosen = {}
    for k, items in warped.items():
        items = sorted(items, key=lambda t: -t[1])[: cfg.top_k]
        wsum = sum(max(t[1], 1e-3) for t in items)
        for ai, w, soft, sl in items:
            P[k - 1][sl] += max(w, 1e-3) / wsum * (soft.astype(np.float32) / 250.0)
        chosen[str(k)] = [{"atlas": atlases[ai].name, "weight": round(w, 4)} for ai, w, _, _ in items]
    info = {"organs_in_prediction": sorted(target), "atlases": qa, "fused": chosen}
    return P, info
