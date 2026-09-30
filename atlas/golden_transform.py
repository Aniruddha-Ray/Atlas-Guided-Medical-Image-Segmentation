"""
Golden transformation construction between a pair of FULLY LABELED atlases
(research report Section 1.2-1.3): a high-confidence reference registration
Ai <-> Aj, used to score which of their landmark correspondences are
reliable enough to later search for in an unlabeled target (atlas/matching.py).

Unlike atlas-to-target registration, both sides here have complete organ
labels, so point correspondence between the two landmark sets doesn't need
patch-intensity matching -- organ id itself is a correct, unambiguous
correspondence key (organ 3 in atlas i IS organ 3 in atlas j, anatomically).
What's NOT already known is which of atlas_i's 40 surface points for organ 3
corresponds to which of atlas_j's 40 surface points for the same organ,
since farthest-point sampling was run independently on each atlas.

Pipeline:
  1. Coarse rigid (Kabsch) pre-alignment using one mean point per organ
     (unambiguous 1:1 correspondence by construction).
  2. Per-organ nearest-neighbor matching between atlas_i's points and the
     coarsely-aligned atlas_j points, restricted to the same organ id (a
     match is never allowed to cross organ boundaries).
  3. Fit a regularized 3D TPS (atlas/tps.py) through all matched pairs ->
     this is the golden transformation T_G(j -> i).
  4. Residual r_k = ||x_i,k - T_G(x_j,k)|| for every matched pair k.
  5. Reliability score s_k = max(T - r_k, 0) / T (report Section 1.3),
     tolerance T in mm. Keep the Top-K (or score > 0) most reliable pairs.

A correspondence with high residual after a SMOOTH global fit means that
particular point is not well explained by the overall atlas-to-atlas
deformation -- i.e. it's a locally noisy/ambiguous point (flat, low-curvature
patch of organ surface, or a boundary-labeling inconsistency between the two
atlases' annotators/protocols) and should not be trusted as a feature to hunt
for in a target CT.
"""
from __future__ import annotations

import numpy as np

from atlas.tps import ThinPlateSpline


def _organ_mean_point(landmark_entry: dict) -> np.ndarray:
    return landmark_entry["points"].mean(axis=0)


def kabsch_rigid_transform(source_points: np.ndarray, target_points: np.ndarray):
    """
    Standard Kabsch/Umeyama rigid alignment (rotation + translation only, no
    scaling): finds R, t minimizing sum ||target_k - (R @ source_k + t)||^2.
    Returns (R (3,3), t (3,)).
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.diag([1.0, 1.0, d])
    R = Vt.T @ correction @ U.T

    t = target_centroid - R @ source_centroid
    return R, t


def match_landmarks_by_organ(landmarks_i: dict, landmarks_j: dict, R: np.ndarray, t: np.ndarray):
    """
    For each organ present in both landmark sets, nearest-neighbor-matches
    atlas_i's points against atlas_j's points after applying the coarse
    rigid transform (R, t) to atlas_j's points (transform brings atlas_j
    into atlas_i's frame, for finding matches only -- the returned pairs use
    the ORIGINAL, untransformed atlas_j points, since the golden TPS should
    learn the full deformation, not just the residual after the rigid step).

    Returns: (source_points (N,3) from atlas_j, target_points (N,3) from
    atlas_i, organ_ids (N,) the organ each pair belongs to) -- all as the
    correspondences to fit the golden TPS on (source=atlas_j, target=atlas_i,
    i.e. T_G maps atlas_j -> atlas_i).
    """
    matched_source = []
    matched_target = []
    matched_organ = []

    common_organs = sorted(set(landmarks_i.keys()) & set(landmarks_j.keys()))
    for organ_id in common_organs:
        pts_i = landmarks_i[organ_id]["points"]
        pts_j = landmarks_j[organ_id]["points"]
        pts_j_aligned = pts_j @ R.T + t

        # One-directional NN: for each atlas_i point, find its closest
        # (rigid-aligned) atlas_j point of the SAME organ.
        diffs = pts_i[:, None, :] - pts_j_aligned[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        nearest_j_idx = np.argmin(dists, axis=1)

        matched_target.append(pts_i)
        matched_source.append(pts_j[nearest_j_idx])  # original (untransformed) atlas_j points
        matched_organ.append(np.full(pts_i.shape[0], organ_id))

    return (
        np.concatenate(matched_source, axis=0),
        np.concatenate(matched_target, axis=0),
        np.concatenate(matched_organ, axis=0),
    )


def build_golden_transformation(
    landmarks_i: dict,
    landmarks_j: dict,
    tolerance_mm: float = 15.0,
    tps_regularization: float = 5.0,
):
    """
    landmarks_i, landmarks_j: output of atlas.landmarks.extract_organ_landmarks
    for two different atlases (both fully labeled).

    Returns a dict:
        "transform": ThinPlateSpline mapping atlas_j points -> atlas_i points
        "source_points", "target_points", "organ_ids": the correspondences used
        "residuals": (N,) mm
        "reliability": (N,) in [0, 1], s_k = max(T - r_k, 0) / T
        "reliable_mask": (N,) bool, reliability > 0
    """
    common_organs = sorted(set(landmarks_i.keys()) & set(landmarks_j.keys()))
    if len(common_organs) < 4:
        raise ValueError(
            f"need correspondences spanning >=4 organs for a well-posed 3D TPS, "
            f"got {len(common_organs)} organs in common"
        )

    centroid_source = np.stack([_organ_mean_point(landmarks_j[o]) for o in common_organs])
    centroid_target = np.stack([_organ_mean_point(landmarks_i[o]) for o in common_organs])
    R, t = kabsch_rigid_transform(centroid_source, centroid_target)

    source_points, target_points, organ_ids = match_landmarks_by_organ(landmarks_i, landmarks_j, R, t)

    transform = ThinPlateSpline(source_points, target_points, regularization=tps_regularization)

    predicted = transform.transform(source_points)
    residuals = np.linalg.norm(predicted - target_points, axis=1)
    reliability = np.maximum(tolerance_mm - residuals, 0.0) / tolerance_mm

    return {
        "transform": transform,
        "source_points": source_points,
        "target_points": target_points,
        "organ_ids": organ_ids,
        "residuals": residuals,
        "reliability": reliability,
        "reliable_mask": reliability > 0.0,
        "rigid_R": R,
        "rigid_t": t,
    }


def select_reliable_features(golden_result: dict, top_k: int | None = None):
    """
    Applies the Top-K selection (report Section 1.3) on top of
    build_golden_transformation's output. Returns the indices (into the
    original correspondence arrays) of the selected reliable features,
    sorted by descending reliability.
    """
    reliability = golden_result["reliability"]
    reliable_idx = np.where(golden_result["reliable_mask"])[0]
    order = reliable_idx[np.argsort(-reliability[reliable_idx])]
    if top_k is not None:
        order = order[:top_k]
    return order
