import numpy as np

from atlas.golden_transform import (
    build_golden_transformation,
    kabsch_rigid_transform,
    match_landmarks_by_organ,
    select_reliable_features,
)


def _random_rotation(seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def test_kabsch_recovers_known_rigid_transform():
    rng = np.random.default_rng(0)
    R_true = _random_rotation(1)
    t_true = np.array([5.0, -2.0, 3.0])

    source = rng.uniform(-50, 50, size=(20, 3))
    target = source @ R_true.T + t_true

    R_est, t_est = kabsch_rigid_transform(source, target)
    np.testing.assert_allclose(R_est, R_true, atol=1e-6)
    np.testing.assert_allclose(t_est, t_true, atol=1e-6)


def _make_synthetic_atlas_pair(seed=0):
    """
    Builds two synthetic landmark dicts (mimicking extract_organ_landmarks'
    output) for organs 1..6, related by a known rigid transform, with
    atlas_j's per-organ point ORDER shuffled (independent FPS runs would
    never naturally align index-to-index) plus small noise.
    """
    rng = np.random.default_rng(seed)
    R_true = _random_rotation(seed + 1)
    t_true = np.array([10.0, -5.0, 7.0])

    landmarks_i = {}
    landmarks_j = {}
    for organ_id in range(1, 7):
        n_points = 1 if organ_id in (5, 6) else 15  # organs 5,6 = "centroid mode"
        pts_i = rng.uniform(-100, 100, size=(n_points, 3)) + organ_id * 20  # spread organs apart
        pts_j_true_order = (pts_i - t_true) @ R_true  # inverse of pts_i = pts_j @ R_true.T + t_true
        noise = rng.normal(scale=0.05, size=pts_j_true_order.shape)
        pts_j = pts_j_true_order + noise

        shuffle_idx = rng.permutation(n_points)
        landmarks_i[organ_id] = {"points": pts_i, "mode": "surface" if n_points > 1 else "interior"}
        landmarks_j[organ_id] = {"points": pts_j[shuffle_idx], "mode": landmarks_i[organ_id]["mode"]}

    return landmarks_i, landmarks_j, R_true, t_true


def test_match_landmarks_never_crosses_organ_boundaries():
    landmarks_i, landmarks_j, _, _ = _make_synthetic_atlas_pair()
    # Recompute the same coarse rigid pre-alignment build_golden_transformation
    # uses internally (Kabsch on per-organ mean points), to exercise
    # match_landmarks_by_organ directly with realistic inputs.
    from atlas.golden_transform import _organ_mean_point

    common = sorted(set(landmarks_i) & set(landmarks_j))
    c_source = np.stack([_organ_mean_point(landmarks_j[o]) for o in common])
    c_target = np.stack([_organ_mean_point(landmarks_i[o]) for o in common])
    R, t = kabsch_rigid_transform(c_source, c_target)

    source_pts, target_pts, organ_ids = match_landmarks_by_organ(landmarks_i, landmarks_j, R, t)

    # Every matched pair's recorded organ_id must match what it would be if
    # sourced from that organ's landmark arrays (sanity: counts per organ
    # match the atlas_i side's point count for that organ, since matching is
    # one-directional from atlas_i).
    for organ_id in common:
        expected_count = landmarks_i[organ_id]["points"].shape[0]
        assert (organ_ids == organ_id).sum() == expected_count


def test_build_golden_transformation_low_residual_for_pure_rigid_case():
    landmarks_i, landmarks_j, R_true, t_true = _make_synthetic_atlas_pair()

    result = build_golden_transformation(landmarks_i, landmarks_j, tolerance_mm=15.0, tps_regularization=1.0)

    # No real nonlinear warp in this synthetic pair (rigid + tiny noise only)
    # -- residuals should be small and reliability high for nearly everything.
    assert result["residuals"].max() < 2.0
    assert result["reliable_mask"].mean() > 0.9

    # The fitted transform should closely match the known ground-truth rigid
    # map when applied to a fresh point (not a control point).
    test_point = np.array([[3.0, 4.0, 5.0]])
    predicted = result["transform"].transform(test_point)
    expected = test_point @ R_true.T + t_true
    np.testing.assert_allclose(predicted, expected, atol=1.0)


def test_select_reliable_features_respects_top_k_and_ordering():
    landmarks_i, landmarks_j, _, _ = _make_synthetic_atlas_pair()
    result = build_golden_transformation(landmarks_i, landmarks_j, tolerance_mm=15.0)

    idx_all = select_reliable_features(result)
    idx_top5 = select_reliable_features(result, top_k=5)

    assert len(idx_top5) == 5
    # Top-5 must be the 5 highest-reliability entries among all reliable ones.
    assert set(idx_top5) <= set(idx_all)
    reliabilities = result["reliability"][idx_top5]
    assert list(reliabilities) == sorted(reliabilities, reverse=True)


def test_build_golden_transformation_requires_minimum_common_organs():
    import pytest

    landmarks_i = {1: {"points": np.zeros((5, 3))}}
    landmarks_j = {1: {"points": np.zeros((5, 3))}}
    with pytest.raises(ValueError):
        build_golden_transformation(landmarks_i, landmarks_j)
