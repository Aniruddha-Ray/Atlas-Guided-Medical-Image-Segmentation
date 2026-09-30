"""
RANSAC affine estimation for atlas -> target landmark correspondences
(research report Section 1.4, step 1).

Patch matching (atlas/matching.py) always returns a best match, even for a
landmark with no distinctive structure nearby, so some candidates are wrong.
RANSAC repeatedly fits an affine to 4 random correspondences, counts how many
other correspondences agree within `threshold_mm`, and keeps the largest
consistent set. Only those inliers go into the TPS.

Each 4-point hypothesis is also required to be anatomically plausible:
  - no reflection (det > 0), since a patient is never mirrored,
  - per-axis scale (singular values) within `scale_range`, since two
    patients' abdomens differ in size but not by 3x.
"""
from __future__ import annotations

import numpy as np


def fit_affine_lstsq(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 4x4 affine M with dst ~= M[:3,:3] @ src + M[:3,3]. src/dst: (N, 3), N >= 4."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    X = np.concatenate([src, np.ones((src.shape[0], 1))], axis=1)
    sol, *_ = np.linalg.lstsq(X, dst, rcond=None)  # (4, 3)
    M = np.eye(4)
    M[:3, :3] = sol[:3].T
    M[:3, 3] = sol[3]
    return M


def _residuals(M: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    pred = src @ M[:3, :3].T + M[:3, 3]
    return np.linalg.norm(pred - dst, axis=1)


def _is_plausible(M: np.ndarray, scale_range) -> bool:
    A = M[:3, :3]
    if not np.all(np.isfinite(A)) or np.linalg.det(A) <= 0:
        return False
    sv = np.linalg.svd(A, compute_uv=False)
    return bool(sv.min() >= scale_range[0] and sv.max() <= scale_range[1])


def _is_degenerate(points: np.ndarray, min_extent_mm: float) -> bool:
    """4 points that are (nearly) coplanar or collinear can't fix a 3D affine."""
    centered = points - points.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    return bool(sv[-1] < min_extent_mm)


def ransac_affine(
    src: np.ndarray,
    dst: np.ndarray,
    threshold_mm: float = 12.0,
    max_iters: int = 2000,
    confidence: float = 0.999,
    scale_range=(0.6, 1.6),
    min_extent_mm: float = 5.0,
    seed: int = 0,
):
    """
    Robustly fits dst ~= M(src).

    Returns (M (4, 4), inlier_mask (N,) bool, info dict). If no plausible
    hypothesis is found, M is None and inlier_mask is all False.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    empty = (None, np.zeros(n, dtype=bool), {"iterations": 0, "num_inliers": 0})
    if n < 4:
        return empty

    rng = np.random.default_rng(seed)
    best_mask = None
    best_count = 0
    needed = max_iters
    it = 0
    while it < min(max_iters, needed):
        it += 1
        idx = rng.choice(n, size=4, replace=False)
        if _is_degenerate(src[idx], min_extent_mm):
            continue
        M = fit_affine_lstsq(src[idx], dst[idx])
        if not _is_plausible(M, scale_range):
            continue
        mask = _residuals(M, src, dst) < threshold_mm
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
            w = count / n
            if w >= 1.0:
                needed = 0
            else:
                # Standard adaptive stopping: iterations needed to draw one
                # all-inlier 4-sample with the requested confidence.
                needed = int(np.ceil(np.log(1 - confidence) / np.log(1 - w**4 + 1e-12)))

    if best_mask is None or best_count < 4:
        return empty

    # Refine: refit on all inliers, recompute inliers once, refit again.
    M = fit_affine_lstsq(src[best_mask], dst[best_mask])
    mask = _residuals(M, src, dst) < threshold_mm
    if mask.sum() >= 4:
        M = fit_affine_lstsq(src[mask], dst[mask])
        mask = _residuals(M, src, dst) < threshold_mm
    if not _is_plausible(M, scale_range):
        return empty

    res = _residuals(M, src[mask], dst[mask])
    return M, mask, {
        "iterations": it,
        "num_inliers": int(mask.sum()),
        "inlier_fraction": float(mask.mean()),
        "inlier_rmse_mm": float(np.sqrt(np.mean(res**2))) if res.size else float("nan"),
    }
