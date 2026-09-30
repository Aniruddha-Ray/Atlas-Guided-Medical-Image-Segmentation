"""
Landmark-based atlas -> target registration (research report Section 1.4):

    coarse MI affine  ->  NCC patch matching of reliable atlas landmarks
    ->  RANSAC affine (outlier rejection)  ->  smoothing TPS on the inliers

The result is a mapping from TARGET physical points to ATLAS physical points
(both LPS, mm), which is the direction label warping needs: for every target
voxel, look up which atlas voxel it corresponds to. Fitting in this direction
avoids having to invert a TPS, which has no closed-form inverse.

Fallbacks are explicit and recorded in the QA dict, never silent:
    "tps"                 normal path
    "ransac_affine_only"  TPS kept folding even after extra smoothing
    "coarse_affine_only"  too few RANSAC inliers to trust anything finer
A registration that fell back should be looked at before its warped labels
are trusted.

Optional dense refinement (cfg.nonrigid_refinement = "bspline"): landmark
TPS alone leaves ~8-25 mm per-organ error, which erases overlap for small
organs. A B-spline is then optimised on body-masked mutual information with
the landmark result held fixed as its starting point:

    atlas point = T_landmark( p + b(p) ),   b = B-spline displacement in target space

so the landmarks still do the large deformation and the B-spline only
corrects locally. It is kept only if it does not fold; otherwise the landmark
result is used and the rejection is recorded. The paper's method is the
landmark TPS; "tps" vs "tps + bspline" is reported as an ablation.

The older intensity-only B-spline registration in atlas/atlas_pipeline.py
(no landmarks at all) is kept for reference; it is not used here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import SimpleITK as sitk

from atlas.geometry import (
    affine_matrix_to_sitk,
    apply_affine,
    index_to_physical,
    make_reference_grid,
    physical_to_index,
    resample_isotropic,
    resample_to_grid,
    to_xyz_array,
)
from atlas.label_warping import displacement_transform
from atlas.matching import _body_mask, coarse_affine, ncc_block_match, window_intensities
from atlas.ransac import ransac_affine
from atlas.tps import ThinPlateSpline


@dataclass
class RegistrationConfig:
    coarse_spacing_mm: float = 4.0
    match_spacing_mm: float = 2.0
    patch_radius: int = 4  # 9^3 voxels = 18 mm patch at 2 mm
    search_radius: int = 10  # +-20 mm at 2 mm, first pass around the coarse affine
    refine_search_radius: int = 6  # +-12 mm, second pass around the RANSAC affine (0 disables)
    intensity_window: tuple = (-200.0, 300.0)
    min_template_std: float = 0.02
    min_ncc: float = 0.3
    ransac_threshold_mm: float = 12.0
    ransac_max_iters: int = 2000
    min_inliers: int = 15
    tps_regularization: float = 5.0
    tps_max_retries: int = 5  # lambda ladder 5 -> 5120; phi(r)=r entries are ~100s of mm, so small lambda barely smooths
    max_fold_fraction: float = 0.001
    fold_check_spacing_mm: float = 10.0
    nonrigid_refinement: str = "none"  # "none" | "bspline"
    bspline_spacing_mm: float = 3.0  # image resolution the B-spline is optimised at
    bspline_grid_mm: float = 80.0  # initial control-point spacing, halved at the finer level
    bspline_iterations: int = 50
    bspline_sampling: float = 0.1


def prepare_target(target_ct: sitk.Image, cfg: RegistrationConfig) -> dict:
    """Per-target preprocessing shared by all atlases. Compute it once per target."""
    grid = resample_isotropic(target_ct, cfg.match_spacing_mm, sitk.sitkLinear, -1000.0)
    return {
        "ct": target_ct,
        "grid": grid,
        "array": window_intensities(to_xyz_array(grid), cfg.intensity_window),
        "body_mask": to_xyz_array(_body_mask(grid)).astype(bool),
    }


def _fold_fraction(tps: ThinPlateSpline, target: dict, cfg: RegistrationConfig) -> float:
    step = max(1, int(round(cfg.fold_check_spacing_mm / cfg.match_spacing_mm)))
    body = target["body_mask"][::step, ::step, ::step]
    idx = np.argwhere(body) * step
    if idx.shape[0] == 0:
        return 0.0
    pts = index_to_physical(target["grid"], idx.astype(np.float64))
    det = tps.jacobian_determinant(pts)
    return float((det <= 0).mean())


class Registration:
    def __init__(self, status: str, qa: dict, affine_t2a: np.ndarray, tps: ThinPlateSpline | None, debug: dict):
        self.status = status
        self.qa = qa
        self.affine_t2a = affine_t2a
        self.tps = tps
        self.debug = debug
        self.refined = None  # sitk transform (target -> atlas) when B-spline refinement was kept

    def map_landmark(self, points_lps: np.ndarray) -> np.ndarray:
        """The landmark-based mapping only (TPS, or its affine fallback)."""
        if self.tps is not None:
            return self.tps.transform(points_lps)
        return apply_affine(self.affine_t2a, points_lps)

    def map_target_to_atlas(self, points_lps: np.ndarray) -> np.ndarray:
        if self.refined is None:
            return self.map_landmark(points_lps)
        return np.array([self.refined.TransformPoint(tuple(map(float, p))) for p in points_lps])

    def to_sitk_transform(self, reference: sitk.Image):
        """The full mapping as a SimpleITK transform, for resampling onto `reference`."""
        if self.refined is not None:
            return self.refined
        return displacement_transform(self.map_landmark, reference)


def bspline_refine(atlas_ct: sitk.Image, target: dict, reg: Registration, cfg: RegistrationConfig):
    """
    Dense B-spline refinement on top of reg's landmark mapping (see module
    docstring). Returns (transform or None, info). None means the refinement
    folded and must not be used.
    """
    fixed = resample_isotropic(target["ct"], cfg.bspline_spacing_mm, sitk.sitkLinear, -1000.0)
    fmask = _body_mask(fixed)
    init = displacement_transform(reg.map_landmark, fixed)

    extent = np.array(fixed.GetSize()) * np.array(fixed.GetSpacing())
    mesh = [max(1, int(round(e / cfg.bspline_grid_mm))) for e in extent]
    bspline = sitk.BSplineTransformInitializer(fixed, mesh, order=3)

    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    r.SetMetricSamplingStrategy(r.RANDOM)
    r.SetMetricSamplingPercentage(cfg.bspline_sampling, 42)
    r.SetMetricFixedMask(fmask)
    r.SetInterpolator(sitk.sitkLinear)
    # LBFGS2, not LBFGSB: LBFGSB fixes its scales to the first level's
    # parameter count and fails once the mesh is refined (scaleFactors below).
    r.SetOptimizerAsLBFGS2(solutionAccuracy=1e-2, numberOfIterations=cfg.bspline_iterations, deltaConvergenceTolerance=0.01)
    r.SetShrinkFactorsPerLevel([2, 1])
    r.SetSmoothingSigmasPerLevel([1.0, 0.0])
    r.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()
    r.SetMovingInitialTransform(init)
    r.SetInitialTransformAsBSpline(bspline, inPlace=True, scaleFactors=[1, 2])

    metric_before = float(r.MetricEvaluate(fixed, atlas_ct))
    optimized = r.Execute(fixed, atlas_ct)
    metric_after = float(r.GetMetricValue())

    # Applied in reverse order of addition: `optimized` (target space) first, then the landmark field.
    composite = sitk.CompositeTransform(init)
    composite.AddTransform(optimized)

    # Folding check on the full composed mapping.
    check = make_reference_grid(fixed, cfg.fold_check_spacing_mm, pad_voxels=0)
    field = sitk.TransformToDisplacementField(
        composite, sitk.sitkVectorFloat64, check.GetSize(), check.GetOrigin(), check.GetSpacing(), check.GetDirection()
    )
    jac = to_xyz_array(sitk.DisplacementFieldJacobianDeterminant(field))
    body = to_xyz_array(resample_to_grid(fmask, check, sitk.sitkNearestNeighbor, 0)).astype(bool)
    fold = float((jac[body] <= 0).mean()) if body.any() else 0.0

    info = {
        "mesh": mesh,
        "metric_before": metric_before,
        "metric_after": metric_after,
        "fold_fraction": fold,
        "kept": fold <= cfg.max_fold_fraction,
    }
    return (composite if info["kept"] else None), info


def _match_and_ransac(atlas_ct, atlas_points_lps, target, M_base, search_radius, cfg, device):
    """
    One matching pass: resample the atlas into the target grid with M_base
    (target -> atlas), predict each landmark's target position with M_base^-1,
    NCC-search +-search_radius voxels around it, then RANSAC the candidates.
    """
    grid = target["grid"]
    atlas_in_target = resample_to_grid(atlas_ct, grid, sitk.sitkLinear, -1000.0, transform=affine_matrix_to_sitk(M_base))
    moving = window_intensities(to_xyz_array(atlas_in_target), cfg.intensity_window)

    predicted = apply_affine(np.linalg.inv(M_base), atlas_points_lps)
    idx = np.rint(physical_to_index(grid, predicted)).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < np.array(grid.GetSize())), axis=1)

    offsets, ncc, valid = ncc_block_match(
        target["array"],
        moving,
        idx[inside],
        patch_radius=cfg.patch_radius,
        search_radius=search_radius,
        min_template_std=cfg.min_template_std,
        device=device,
    )
    target_matches = index_to_physical(grid, (idx[inside] + offsets).astype(np.float64))
    atlas_pts = atlas_points_lps[inside]
    candidates = valid & (ncc >= cfg.min_ncc)

    M, inliers_c, rinfo = ransac_affine(
        target_matches[candidates],
        atlas_pts[candidates],
        threshold_mm=cfg.ransac_threshold_mm,
        max_iters=cfg.ransac_max_iters,
    )
    inliers = np.zeros_like(candidates)
    inliers[np.where(candidates)[0][inliers_c]] = True
    return {
        "M": M,
        "inliers": inliers,
        "target_matches": target_matches,
        "atlas_pts": atlas_pts,
        "ncc": ncc,
        "candidates": candidates,
        "predicted": predicted[inside],
        "num_inside": int(inside.sum()),
        "summary": {
            "search_radius": int(search_radius),
            "num_candidates": int(candidates.sum()),
            "median_ncc_candidates": float(np.median(ncc[candidates])) if candidates.any() else float("nan"),
            "ransac": rinfo,
        },
    }


def register_atlas_to_target(
    atlas_ct: sitk.Image,
    atlas_points_lps: np.ndarray,
    target: dict,
    cfg: RegistrationConfig | None = None,
    device: str | None = None,
) -> Registration:
    """
    atlas_ct: atlas CT (any resolution; 2 mm is plenty for matching).
    atlas_points_lps: (K, 3) reliable atlas landmarks (atlas/reliable_features.py).
    target: output of prepare_target().
    """
    cfg = cfg or RegistrationConfig()
    reg = _register_landmarks(atlas_ct, atlas_points_lps, target, cfg, device)
    if cfg.nonrigid_refinement == "bspline":
        reg.refined, reg.qa["bspline"] = bspline_refine(atlas_ct, target, reg, cfg)
    elif cfg.nonrigid_refinement != "none":
        raise ValueError(f"unknown nonrigid_refinement: {cfg.nonrigid_refinement!r}")
    return reg


def _register_landmarks(atlas_ct, atlas_points_lps, target, cfg, device) -> Registration:
    qa = {"config": asdict(cfg), "num_atlas_landmarks": int(len(atlas_points_lps))}

    # 1. Coarse registration (target -> atlas).
    M0, coarse_info = coarse_affine(atlas_ct, target["ct"], cfg.coarse_spacing_mm)
    qa["coarse"] = coarse_info
    qa["coarse_mi_metric"] = coarse_info["metric"]

    # 2-3. Patch matching around the coarse estimate, then RANSAC.
    p1 = _match_and_ransac(atlas_ct, atlas_points_lps, target, M0, cfg.search_radius, cfg, device)
    qa["num_inside_target"] = p1["num_inside"]
    qa["pass1"] = p1["summary"]
    best = p1

    # Second pass: re-match around the RANSAC affine with a tighter window.
    # Landmarks that were just outside the first window can now be found,
    # and a smaller window has fewer chances of a spurious NCC peak.
    if cfg.refine_search_radius > 0 and p1["M"] is not None and p1["inliers"].sum() >= cfg.min_inliers:
        p2 = _match_and_ransac(atlas_ct, atlas_points_lps, target, p1["M"], cfg.refine_search_radius, cfg, device)
        qa["pass2"] = p2["summary"]
        if p2["M"] is not None and p2["inliers"].sum() >= p1["inliers"].sum():
            best = p2
    qa["pass_used"] = 2 if best is not p1 else 1

    M1, inliers, target_matches, atlas_pts = best["M"], best["inliers"], best["target_matches"], best["atlas_pts"]
    qa["num_candidates"] = best["summary"]["num_candidates"]
    qa["median_ncc_candidates"] = best["summary"]["median_ncc_candidates"]
    qa["ransac"] = best["summary"]["ransac"]
    debug = {k: best[k] for k in ("atlas_pts", "target_matches", "ncc", "candidates", "inliers", "predicted")}

    if M1 is None or inliers.sum() < cfg.min_inliers:
        qa["status"] = "coarse_affine_only"
        return Registration("coarse_affine_only", qa, M0, None, debug)

    # How much RANSAC moved things relative to the coarse affine, on the landmarks.
    qa["ransac_vs_coarse_mean_shift_mm"] = float(
        np.linalg.norm(apply_affine(M1, target_matches[inliers]) - apply_affine(M0, target_matches[inliers]), axis=1).mean()
    )

    # 4. Smoothing TPS on the inliers, with a folding check.
    lam = cfg.tps_regularization
    for attempt in range(cfg.tps_max_retries + 1):
        tps = ThinPlateSpline(target_matches[inliers], atlas_pts[inliers], regularization=lam)
        fold = _fold_fraction(tps, target, cfg)
        if fold <= cfg.max_fold_fraction:
            residual = np.linalg.norm(tps.transform(target_matches[inliers]) - atlas_pts[inliers], axis=1)
            qa.update(
                status="tps",
                tps_regularization_used=lam,
                tps_retries=attempt,
                fold_fraction=fold,
                tps_inlier_residual_mean_mm=float(residual.mean()),
            )
            return Registration("tps", qa, M1, tps, debug)
        lam *= 4.0

    qa.update(status="ransac_affine_only", fold_fraction=fold, tps_regularization_used=lam / 4.0)
    return Registration("ransac_affine_only", qa, M1, None, debug)
