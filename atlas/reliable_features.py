"""
Global reliable-feature selection across the whole atlas library
(research report Section 1.3).

For atlas i, each of its landmarks f_k gets a reliability score averaged over
the golden transformations against every other atlas j:

    Score_i(f_k) = (1 / (N-1)) * sum_{j != i} s_k^{(i,j)},
    s_k^{(i,j)} = max(T - r_k^{(i,j)}, 0) / T

where r_k^{(i,j)} is the residual of f_k under the golden TPS T_G(j -> i)
(atlas/golden_transform.py). A landmark that is only consistent with one or
two other atlases -- a boundary point that annotators placed inconsistently,
or a genuinely variable spot of anatomy -- ends up with a low average and is
not used for atlas -> target matching.

Selection keeps landmarks with Score >= min_score. Organ coverage also matters:
if an organ lost all of its landmarks, the atlas -> target TPS would have no
control over that region. So up to `min_per_organ` best landmarks per organ are
kept even when below min_score, as long as their score is > 0. A landmark that
no golden transformation supports at all (score 0) is never kept.

All landmark coordinates are in SimpleITK physical space (LPS, mm) -- see
atlas/geometry.py.
"""
from __future__ import annotations

import numpy as np

from atlas.geometry import read_label, sitk_index_to_lps_affine, to_xyz_array
from atlas.golden_transform import build_golden_transformation
from atlas.landmarks import extract_organ_landmarks


def extract_atlas_landmarks_lps(label_path: str, organ_ids, **landmark_kwargs) -> dict:
    """Landmarks for one atlas label file, directly in LPS physical space."""
    label = read_label(label_path)
    return extract_organ_landmarks(
        to_xyz_array(label), sitk_index_to_lps_affine(label), organ_ids=organ_ids, **landmark_kwargs
    )


def score_atlas_library(all_landmarks: list, tolerance_mm: float = 15.0, tps_regularization: float = 5.0):
    """
    all_landmarks: list (one entry per atlas) of extract_organ_landmarks outputs.

    Returns: list (one per atlas) of dict organ_id -> (n_points,) mean
    reliability over all other atlases, aligned with that atlas's
    landmarks[organ_id]["points"] order.
    """
    n_atlas = len(all_landmarks)
    if n_atlas < 2:
        raise ValueError("need at least 2 atlases to score reliability")

    sums = [{o: np.zeros(len(lm[o]["points"])) for o in lm} for lm in all_landmarks]
    counts = [{o: 0 for o in lm} for lm in all_landmarks]

    for i in range(n_atlas):
        for j in range(n_atlas):
            if i == j:
                continue
            result = build_golden_transformation(
                all_landmarks[i], all_landmarks[j], tolerance_mm=tolerance_mm, tps_regularization=tps_regularization
            )
            # Correspondences are indexed by atlas i's own points, grouped by
            # organ in the same order as landmarks_i[organ]["points"].
            for organ_id in np.unique(result["organ_ids"]):
                organ_id = int(organ_id)
                sums[i][organ_id] += result["reliability"][result["organ_ids"] == organ_id]
                counts[i][organ_id] += 1

    return [{o: sums[i][o] / max(counts[i][o], 1) for o in sums[i]} for i in range(n_atlas)]


def select_reliable_landmarks(landmarks: dict, scores: dict, min_score: float = 0.3, min_per_organ: int = 3):
    """
    Flattens one atlas's landmarks into arrays and keeps the reliable subset.

    Returns dict with "points" (K, 3), "organ_ids" (K,), "scores" (K,).
    """
    keep_points, keep_organs, keep_scores = [], [], []
    for organ_id in sorted(landmarks):
        pts = landmarks[organ_id]["points"]
        s = scores[organ_id]
        order = np.argsort(-s)
        keep = s >= min_score
        # Coverage floor: best few per organ, but never a completely unsupported point.
        floor_idx = [k for k in order[:min_per_organ] if s[k] > 0]
        keep[floor_idx] = True
        keep_points.append(pts[keep])
        keep_organs.append(np.full(int(keep.sum()), organ_id))
        keep_scores.append(s[keep])
    return {
        "points": np.concatenate(keep_points, axis=0),
        "organ_ids": np.concatenate(keep_organs, axis=0),
        "scores": np.concatenate(keep_scores, axis=0),
    }


def build_atlas_library(
    atlas_cases: list,
    organ_ids=range(1, 13),
    tolerance_mm: float = 15.0,
    tps_regularization: float = 5.0,
    min_score: float = 0.3,
    min_per_organ: int = 3,
    landmark_kwargs: dict | None = None,
    log=print,
):
    """
    atlas_cases: list of {"image": path, "label": path} (atlas/atlas_selection.json "cases").

    Returns a list (one per atlas) of dicts:
        image, label        : source file paths
        points, organ_ids,
        scores              : the SELECTED reliable landmarks (LPS mm)
        all_points, all_organ_ids,
        all_scores          : every landmark before selection (for analysis)
    """
    landmark_kwargs = landmark_kwargs or {}
    all_landmarks = []
    for k, case in enumerate(atlas_cases):
        log(f"  [{k + 1}/{len(atlas_cases)}] landmarks: {case['label']}")
        all_landmarks.append(extract_atlas_landmarks_lps(case["label"], organ_ids, **landmark_kwargs))

    log("  scoring golden transformations across all ordered atlas pairs...")
    scores = score_atlas_library(all_landmarks, tolerance_mm=tolerance_mm, tps_regularization=tps_regularization)

    library = []
    for case, lm, sc in zip(atlas_cases, all_landmarks, scores):
        selected = select_reliable_landmarks(lm, sc, min_score=min_score, min_per_organ=min_per_organ)
        everything = select_reliable_landmarks(lm, sc, min_score=-1.0, min_per_organ=0)
        library.append(
            {
                "image": case["image"],
                "label": case["label"],
                "points": selected["points"],
                "organ_ids": selected["organ_ids"],
                "scores": selected["scores"],
                "all_points": everything["points"],
                "all_organ_ids": everything["organ_ids"],
                "all_scores": everything["scores"],
            }
        )
    return library


def save_atlas_library(library: list, path: str):
    arrays = {}
    for i, entry in enumerate(library):
        for key in ("points", "organ_ids", "scores", "all_points", "all_organ_ids", "all_scores"):
            arrays[f"atlas{i}_{key}"] = entry[key]
    arrays["images"] = np.array([e["image"] for e in library])
    arrays["labels"] = np.array([e["label"] for e in library])
    np.savez_compressed(path, **arrays)


def load_atlas_library(path: str) -> list:
    data = np.load(path, allow_pickle=False)
    images = data["images"].tolist()
    labels = data["labels"].tolist()
    library = []
    for i, (image, label) in enumerate(zip(images, labels)):
        entry = {"image": image, "label": label}
        for key in ("points", "organ_ids", "scores", "all_points", "all_organ_ids", "all_scores"):
            entry[key] = data[f"atlas{i}_{key}"]
        library.append(entry)
    return library
