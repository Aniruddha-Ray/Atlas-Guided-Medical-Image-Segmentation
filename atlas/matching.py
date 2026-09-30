"""
Locating atlas landmarks in an UNLABELED target CT (research report Section 1.4).

The target has no organ labels, so atlas landmarks cannot be matched by organ
id the way atlas <-> atlas golden transformations are. They have to be found
from image intensities alone, in two steps:

1. coarse_affine: low-resolution (default 4 mm), body-masked mutual-
   information registration, run coarse-to-fine so scale is never free before
   position is right:
     a. in-plane: align body-mask centres of mass (x, y);
     b. z: exhaustive search over z offsets. Scans cover different stretches
        of the body, so a centre-of-mass guess in z is biased by the
        field-of-view mismatch. An earlier unconstrained version collapsed
        z to 0.19x on one atlas pair;
     c. similarity (rigid + ONE uniform scale);
     d. affine, accepted only if every axis scale stays within
        `affine_scale_range`, otherwise the similarity result is kept.
   This only needs to put each landmark inside the patch-search window of its
   true location. It is not the final transform.

2. ncc_block_match: for every landmark, take a small intensity patch around
   it in the atlas (already resampled into the target grid by the coarse
   affine) and search a local window of the target for the offset that
   maximises normalised cross-correlation (NCC). Implemented as a grouped
   3D convolution in torch, so all landmarks are matched in one batched call,
   on GPU when available.

NCC is invariant to linear intensity changes (contrast phase, scanner
calibration). Intensities are windowed to soft tissue first (default
[-200, 300] HU) so fat/air/bone extremes don't dominate the correlation.
Landmarks whose atlas patch is almost flat (e.g. deep inside a homogeneous
organ) are flagged invalid: a flat template matches everywhere equally, so
its "best" offset is noise.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from atlas.geometry import resample_isotropic, sitk_affine_to_matrix


def _body_mask(image: sitk.Image, threshold_hu: float = -500.0) -> sitk.Image:
    mask = sitk.BinaryThreshold(image, lowerThreshold=threshold_hu, upperThreshold=1e6, insideValue=1, outsideValue=0)
    return sitk.BinaryMorphologicalOpening(mask, [1, 1, 1])


def _mask_stats(mask: sitk.Image):
    """Physical centroid and physical z-extent of a binary body mask."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    centroid = np.array(stats.GetCentroid(1))
    x0, y0, z0, sx, sy, sz = stats.GetBoundingBox(1)
    corners = [
        mask.TransformIndexToPhysicalPoint((x, y, z))
        for x in (x0, x0 + sx - 1)
        for y in (y0, y0 + sy - 1)
        for z in (z0, z0 + sz - 1)
    ]
    zs = [c[2] for c in corners]
    return centroid, (min(zs), max(zs))


def _registration_method(fixed_mask, moving_mask, iterations: int, seed: int, levels=True):
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed)
    reg.SetMetricFixedMask(fixed_mask)
    reg.SetMetricMovingMask(moving_mask)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=iterations, gradientMagnitudeTolerance=1e-6
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    if levels:
        reg.SetShrinkFactorsPerLevel([2, 1])
        reg.SetSmoothingSigmasPerLevel([1.0, 0.0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    return reg


def coarse_affine(
    atlas_ct: sitk.Image,
    target_ct: sitk.Image,
    spacing_mm: float = 4.0,
    z_search_mm: float = 200.0,
    z_step_mm: float = 10.0,
    min_z_overlap: float = 0.5,
    affine_scale_range=(0.8, 1.25),
    iterations: int = 200,
    seed: int = 42,
):
    """
    Coarse registration with fixed = target and moving = atlas (see module docstring).

    Returns (M_t2a (4, 4), info). M_t2a maps target LPS physical points to
    atlas LPS physical points, the direction sitk.Resample and label warping
    both need. info records which stage was kept and why.
    """
    fixed = resample_isotropic(target_ct, spacing_mm, sitk.sitkLinear, -1000.0)
    moving = resample_isotropic(atlas_ct, spacing_mm, sitk.sitkLinear, -1000.0)
    fmask = _body_mask(fixed)
    mmask = _body_mask(moving)
    c_f, (fz0, fz1) = _mask_stats(fmask)
    c_m, (mz0, mz1) = _mask_stats(mmask)
    t0 = c_m - c_f
    min_extent = min(fz1 - fz0, mz1 - mz0)

    # b. z-offset search (translation only, body-masked MI).
    probe = _registration_method(fmask, mmask, iterations=1, seed=seed, levels=False)
    best = None
    for dz in np.arange(-z_search_mm, z_search_mm + 1e-6, z_step_mm):
        t = t0 + np.array([0.0, 0.0, dz])
        overlap = max(0.0, min(fz1 + t[2], mz1) - max(fz0 + t[2], mz0)) / max(min_extent, 1e-6)
        if overlap < min_z_overlap:
            continue
        probe.SetInitialTransform(sitk.TranslationTransform(3, t.tolist()))
        value = probe.MetricEvaluate(fixed, moving)
        if best is None or value < best[0]:
            best = (value, dz, t, overlap)
    if best is None:
        best = (float("nan"), 0.0, t0, 0.0)
    _, dz_best, t_best, overlap_best = best

    # c. similarity: rigid + one uniform scale.
    sim = sitk.Similarity3DTransform()
    sim.SetCenter(c_f.tolist())
    sim.SetTranslation(t_best.tolist())
    reg = _registration_method(fmask, mmask, iterations=iterations, seed=seed)
    reg.SetInitialTransform(sim, inPlace=True)
    reg.Execute(fixed, moving)
    sim_metric = float(reg.GetMetricValue())

    aff_sim = sitk.AffineTransform(3)
    aff_sim.SetCenter(sim.GetCenter())
    aff_sim.SetMatrix(sim.GetMatrix())
    aff_sim.SetTranslation(sim.GetTranslation())
    M_sim = sitk_affine_to_matrix(aff_sim)

    # d. affine, only kept if its per-axis scales stay plausible.
    aff = sitk.AffineTransform(aff_sim)
    reg = _registration_method(fmask, mmask, iterations=iterations, seed=seed)
    reg.SetInitialTransform(aff, inPlace=True)
    reg.Execute(fixed, moving)
    M_aff = sitk_affine_to_matrix(aff)
    sv = np.linalg.svd(M_aff[:3, :3], compute_uv=False)
    plausible = np.linalg.det(M_aff[:3, :3]) > 0 and sv.min() >= affine_scale_range[0] and sv.max() <= affine_scale_range[1]

    M = M_aff if plausible else M_sim
    info = {
        "stage": "affine" if plausible else "similarity",
        "z_offset_mm": float(dz_best),
        "z_overlap": float(overlap_best),
        "similarity_scale": float(sim.GetScale()),
        "affine_scales": [float(s) for s in sv],
        "metric": float(reg.GetMetricValue()) if plausible else sim_metric,
    }
    return M, info


def window_intensities(array: np.ndarray, window=(-200.0, 300.0)) -> np.ndarray:
    lo, hi = window
    return ((np.clip(array, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def ncc_block_match(
    fixed: np.ndarray,
    moving: np.ndarray,
    centers: np.ndarray,
    patch_radius: int = 4,
    search_radius: int = 10,
    min_template_std: float = 0.02,
    device: str | None = None,
    batch_size: int = 256,
):
    """
    fixed:   target volume (any consistent axis order), float32, windowed.
    moving:  atlas volume resampled onto the SAME grid as `fixed`, float32, windowed.
    centers: (L, 3) integer voxel indices (same axis order) of each landmark's
             predicted location in that grid.

    For landmark l, the template is moving[center_l +- patch_radius] and it is
    searched over fixed[center_l + offset +- patch_radius] for every
    |offset| <= search_radius on each axis.

    Returns:
        offsets (L, 3) int: best offset per landmark (fixed index = center + offset)
        peak_ncc (L,) float
        valid (L,) bool: False if the template is too flat to be trusted
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    P = 2 * patch_radius + 1
    S = 2 * search_radius + 1
    R = patch_radius + search_radius
    W = P + 2 * search_radius

    pad_value = float(min(fixed.min(), moving.min()))
    fixed_p = np.pad(fixed, R, mode="constant", constant_values=pad_value)
    moving_p = np.pad(moving, R, mode="constant", constant_values=pad_value)
    centers = np.asarray(centers, dtype=np.int64) + R

    L = centers.shape[0]
    offsets = np.zeros((L, 3), dtype=np.int64)
    peak = np.full(L, -1.0, dtype=np.float32)
    valid = np.zeros(L, dtype=bool)
    box = torch.ones((1, 1, P, P, P), device=device)
    n = float(P**3)

    for start in range(0, L, batch_size):
        c = centers[start : start + batch_size]
        b = c.shape[0]
        templates = np.stack(
            [moving_p[x - patch_radius : x + patch_radius + 1, y - patch_radius : y + patch_radius + 1, z - patch_radius : z + patch_radius + 1] for x, y, z in c]
        )
        regions = np.stack([fixed_p[x - R : x - R + W, y - R : y - R + W, z - R : z - R + W] for x, y, z in c])

        t = torch.from_numpy(templates).to(device)
        r = torch.from_numpy(regions).to(device)

        t_centered = t - t.mean(dim=(1, 2, 3), keepdim=True)
        t_norm = torch.sqrt((t_centered**2).sum(dim=(1, 2, 3)))  # (b,)
        t_std = torch.sqrt((t_centered**2).mean(dim=(1, 2, 3)))

        # Grouped conv: each landmark's template correlates only with its own region.
        cross = F.conv3d(r.unsqueeze(0), t_centered.unsqueeze(1), groups=b)[0]  # (b, S, S, S)
        local_sum = F.conv3d(r.unsqueeze(1), box)[:, 0]  # (b, S, S, S)
        local_sq = F.conv3d((r**2).unsqueeze(1), box)[:, 0]
        local_var = torch.clamp(local_sq - local_sum**2 / n, min=1e-8)
        ncc = cross / (t_norm.view(b, 1, 1, 1) * torch.sqrt(local_var) + 1e-8)

        flat = ncc.reshape(b, -1)
        best_val, best_idx = flat.max(dim=1)
        best_idx = best_idx.cpu().numpy()
        o = np.stack(np.unravel_index(best_idx, (S, S, S)), axis=1) - search_radius

        offsets[start : start + b] = o
        peak[start : start + b] = best_val.cpu().numpy()
        valid[start : start + b] = (t_std >= min_template_std).cpu().numpy()

    return offsets, peak, valid
