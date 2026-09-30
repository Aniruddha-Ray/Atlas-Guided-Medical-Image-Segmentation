import numpy as np

from atlas.landmarks import _voxel_to_world, extract_organ_landmarks, farthest_point_sample


def test_small_organ_uses_multiple_interior_points():
    label = np.zeros((30, 30, 30), dtype=np.int32)
    label[8:14, 8:14, 8:14] = 5  # 6^3=216 voxels, 1mm^3 each -- well below the small-organ threshold

    landmarks = extract_organ_landmarks(
        label, np.eye(4), organ_ids=[5], small_organ_volume_mm3=8000.0, small_organ_num_points=5
    )

    assert 5 in landmarks
    assert landmarks[5]["mode"] == "interior"
    # Multiple points, not a single centroid -- gives the golden-transformation
    # reliability scoring something to average over (see module docstring).
    assert landmarks[5]["points"].shape == (5, 3)
    assert landmarks[5]["voxel_count"] == 216
    assert landmarks[5]["volume_mm3"] == 216.0
    # every sampled point must be an actual interior voxel of the block, i.e.
    # within [8, 14) on every axis -- not a boundary/surface-only point and
    # not outside the organ.
    pts = landmarks[5]["points"]
    assert np.all((pts >= 8) & (pts < 14))


def test_small_organ_falls_back_to_fewer_points_than_requested():
    label = np.zeros((10, 10, 10), dtype=np.int32)
    label[0, 0, 0:3] = 7  # only 3 voxels total

    landmarks = extract_organ_landmarks(
        label, np.eye(4), organ_ids=[7], small_organ_volume_mm3=8000.0, small_organ_num_points=5
    )
    assert landmarks[7]["points"].shape == (3, 3)  # capped at the available voxel count


def test_large_organ_uses_surface_sampling():
    label = np.zeros((40, 40, 40), dtype=np.int32)
    label[5:35, 5:35, 5:35] = 3  # 30^3 = 27000 voxels/mm^3 -- comfortably "large"

    landmarks = extract_organ_landmarks(
        label, np.eye(4), organ_ids=[3], small_organ_volume_mm3=8000.0, points_per_organ=40
    )

    assert landmarks[3]["mode"] == "surface"
    pts = landmarks[3]["points"]
    assert pts.shape[0] == 40
    # every sampled point should lie ON the cube's boundary (a face), not the interior.
    on_boundary = np.any((pts == 5) | (pts == 34), axis=1)
    assert on_boundary.all()


def test_absent_organ_is_omitted():
    label = np.zeros((10, 10, 10), dtype=np.int32)
    landmarks = extract_organ_landmarks(label, np.eye(4), organ_ids=[1, 2, 3])
    assert landmarks == {}


def test_world_coordinates_respect_affine():
    label = np.zeros((20, 20, 20), dtype=np.int32)
    label[9:11, 9:11, 9:11] = 1  # tiny interior-mode organ, voxel centroid (9.5, 9.5, 9.5)

    affine = np.eye(4)
    affine[:3, :3] *= 2.0  # 2mm isotropic spacing
    affine[:3, 3] = [100.0, 0.0, 0.0]  # origin offset

    landmarks = extract_organ_landmarks(label, affine, organ_ids=[1], small_organ_volume_mm3=8000.0)
    # Every returned point is one of this 2x2x2 block's 8 voxels put through
    # the affine -- so each point's per-axis coordinate must be one of the
    # two values the affine maps voxel index 9 or 10 to, on every axis. This
    # confirms the affine is actually applied (scale + origin offset) without
    # depending on which specific points farthest-point sampling picked.
    possible_per_axis = np.array([9.0, 10.0]) * 2.0 + np.array([100.0, 0.0, 0.0])[:, None]
    # possible_per_axis[axis] = the 2 valid world coords on that axis
    pts = landmarks[1]["points"]
    for axis in range(3):
        assert np.isin(pts[:, axis], possible_per_axis[axis]).all()

    # Every one of the 8 corners must map exactly onto one of the 8 possible
    # affine-transformed corner positions (cross-checks scale AND origin
    # together, not just per-axis membership).
    all_corners_world = _voxel_to_world(
        np.array([[i, j, k] for i in (9, 10) for j in (9, 10) for k in (9, 10)], dtype=float), affine
    )
    for p in pts:
        assert np.any(np.all(np.isclose(all_corners_world, p), axis=1))


def test_farthest_point_sample_spreads_out_and_is_deterministic():
    rng = np.random.default_rng(0)
    points = rng.uniform(0, 100, size=(500, 3))

    sampled_a = farthest_point_sample(points, k=20, seed=1)
    sampled_b = farthest_point_sample(points, k=20, seed=1)
    np.testing.assert_array_equal(sampled_a, sampled_b)  # deterministic given the same seed

    assert sampled_a.shape == (20, 3)

    # Spread check: mean pairwise distance among FPS points should exceed the
    # mean pairwise distance of a plain random sample of the same size.
    def mean_pairwise_dist(pts):
        diffs = pts[:, None, :] - pts[None, :, :]
        d = np.linalg.norm(diffs, axis=-1)
        iu = np.triu_indices(len(pts), k=1)
        return d[iu].mean()

    random_sample = points[np.random.default_rng(2).choice(len(points), 20, replace=False)]
    assert mean_pairwise_dist(sampled_a) > mean_pairwise_dist(random_sample)


def test_farthest_point_sample_returns_all_points_when_k_exceeds_n():
    points = np.random.default_rng(0).uniform(0, 1, size=(5, 3))
    sampled = farthest_point_sample(points, k=10)
    assert sampled.shape == (5, 3)
