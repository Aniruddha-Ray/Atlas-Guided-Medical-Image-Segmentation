"""
Per-organ landmark extraction for the golden-transformation step (atlas <->
atlas registration). For each organ:

  - if the organ is "small" (physical volume below a threshold), its
    landmarks are a handful of INTERIOR points (farthest-point sampled over
    the whole organ, not just its boundary) -- small organs (10, 11, 12 in
    the current baseline) are exactly where boundary/surface points are
    noisiest (annotator disagreement is concentrated at organ edges), and a
    single centroid point gives no redundancy for the reliability scoring in
    atlas/golden_transform.py.
  - otherwise, landmarks are farthest-point-sampled points on the organ's
    surface (boundary voxels), which spreads points evenly instead of
    clustering wherever the boundary happens to be densely sampled by the
    voxel grid.

All landmark coordinates are returned in WORLD (mm) space via the image's
affine, not voxel indices, since two atlases are never guaranteed to share a
voxel grid, spacing, or orientation, but a golden transformation must operate
in a physically meaningful, comparable space.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion


def _voxel_to_world(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """voxel_coords: (N, 3) array of (i, j, k) indices. Returns (N, 3) world (mm) coords."""
    homogeneous = np.concatenate([voxel_coords, np.ones((voxel_coords.shape[0], 1))], axis=1)
    world = homogeneous @ affine.T
    return world[:, :3]


def _boundary_voxels(mask: np.ndarray) -> np.ndarray:
    """mask: bool (D, H, W). Returns (N, 3) voxel indices on the mask's boundary."""
    eroded = binary_erosion(mask, iterations=1)
    boundary = mask & ~eroded
    if not boundary.any():
        # A mask thinner than the erosion structuring element (e.g. a
        # 1-voxel-thick sliver) has no interior to erode away -- the whole
        # mask IS the boundary.
        boundary = mask
    return np.argwhere(boundary)


def farthest_point_sample(points: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """
    Greedy farthest-point sampling: picks k points from `points` (N, D) that
    are spread out over the point set, instead of a uniform-random subset
    that could cluster by chance. Returns (min(k, N), D).
    """
    n = points.shape[0]
    if n <= k:
        return points

    rng = np.random.default_rng(seed)
    selected_idx = [int(rng.integers(0, n))]
    dist = np.linalg.norm(points - points[selected_idx[0]], axis=1)

    for _ in range(1, k):
        next_idx = int(np.argmax(dist))
        selected_idx.append(next_idx)
        new_dist = np.linalg.norm(points - points[next_idx], axis=1)
        dist = np.minimum(dist, new_dist)

    return points[selected_idx]


def extract_organ_landmarks(
    label_data: np.ndarray,
    affine: np.ndarray,
    organ_ids,
    small_organ_volume_mm3: float = 8000.0,
    points_per_organ: int = 40,
    small_organ_num_points: int = 5,
    max_boundary_candidates: int = 3000,
    seed: int = 0,
):
    """
    label_data: integer label volume (D, H, W), voxel values = organ ids (0 = background).
    affine: the image's 4x4 voxel-to-world affine (e.g. from nib.load(...).affine).
    organ_ids: iterable of organ ids to extract (e.g. range(1, 13)).

    small_organ_volume_mm3 is a PHYSICAL volume threshold (default 8000 mm^3
    = 8 mL), not a voxel count: the merged dataset mixes sources with very
    different slice thickness (0.5mm-5mm seen in practice), so raw voxel
    count is not comparable across cases -- a 5mm-slice case needs ~9x fewer
    voxels than a 0.57mm-slice case to represent the same physical organ.
    Voxel volume is derived from the affine's determinant.

    Small organs get `small_organ_num_points` INTERIOR points (farthest-point
    sampled over the whole organ volume, not just its boundary), rather than
    a single centroid. A single point gives the golden-transformation
    reliability score (atlas/golden_transform.py) nothing to average over:
    real data showed a single-centroid organ can swing from 0% to 100%
    reliable across different atlas pairs purely from one noisy estimate.
    Interior points are still deliberately NOT boundary/surface points --
    the organ's surface is exactly where two annotators are most likely to
    disagree, which is the noise this is meant to avoid in the first place.

    Returns: dict organ_id -> {
        "points": (N, 3) float array, world (mm) coordinates,
        "mode": "interior" or "surface",
        "voxel_count": int,
        "volume_mm3": float,
    }
    Organs absent from this label (voxel_count == 0) are omitted from the result.
    """
    result = {}
    rng = np.random.default_rng(seed)
    voxel_volume_mm3 = abs(np.linalg.det(affine[:3, :3]))

    for organ_id in organ_ids:
        mask = label_data == organ_id
        voxel_count = int(mask.sum())
        if voxel_count == 0:
            continue
        volume_mm3 = voxel_count * voxel_volume_mm3

        if volume_mm3 < small_organ_volume_mm3:
            interior_voxels = np.argwhere(mask).astype(np.float64)
            if interior_voxels.shape[0] > max_boundary_candidates:
                keep_idx = rng.choice(interior_voxels.shape[0], size=max_boundary_candidates, replace=False)
                interior_voxels = interior_voxels[keep_idx]
            k = min(small_organ_num_points, interior_voxels.shape[0])
            sampled_voxels = farthest_point_sample(interior_voxels, k, seed=seed)
            world_points = _voxel_to_world(sampled_voxels, affine)
            result[organ_id] = {
                "points": world_points,
                "mode": "interior",
                "voxel_count": voxel_count,
                "volume_mm3": volume_mm3,
            }
            continue

        boundary_voxels = _boundary_voxels(mask)
        if boundary_voxels.shape[0] > max_boundary_candidates:
            keep_idx = rng.choice(boundary_voxels.shape[0], size=max_boundary_candidates, replace=False)
            boundary_voxels = boundary_voxels[keep_idx]

        boundary_world = _voxel_to_world(boundary_voxels.astype(np.float64), affine)
        sampled = farthest_point_sample(boundary_world, points_per_organ, seed=seed)
        result[organ_id] = {
            "points": sampled,
            "mode": "surface",
            "voxel_count": voxel_count,
            "volume_mm3": volume_mm3,
        }

    return result
