import numpy as np

from atlas.ransac import fit_affine_lstsq, ransac_affine


def _true_affine():
    M = np.eye(4)
    M[:3, :3] = [[1.05, 0.03, 0.0], [-0.02, 0.97, 0.04], [0.01, 0.0, 1.1]]
    M[:3, 3] = [12.0, -8.0, 5.0]
    return M


def test_fit_affine_lstsq_exact():
    rng = np.random.default_rng(0)
    M = _true_affine()
    src = rng.uniform(-100, 100, size=(10, 3))
    dst = src @ M[:3, :3].T + M[:3, 3]
    np.testing.assert_allclose(fit_affine_lstsq(src, dst), M, atol=1e-9)


def test_ransac_recovers_affine_with_40_percent_outliers():
    rng = np.random.default_rng(1)
    M = _true_affine()
    n = 200
    src = rng.uniform(-150, 150, size=(n, 3))
    dst = src @ M[:3, :3].T + M[:3, 3] + rng.normal(scale=1.0, size=(n, 3))
    outliers = rng.choice(n, size=80, replace=False)
    dst[outliers] += rng.uniform(30, 80, size=(80, 3)) * rng.choice([-1, 1], size=(80, 3))

    M_est, mask, info = ransac_affine(src, dst, threshold_mm=6.0, seed=2)

    assert M_est is not None
    expected_mask = np.ones(n, dtype=bool)
    expected_mask[outliers] = False
    # nearly all true inliers found, (almost) no outliers accepted
    assert (mask & expected_mask).sum() >= 0.95 * expected_mask.sum()
    assert (mask & ~expected_mask).sum() <= 2
    probe = np.array([[20.0, 30.0, -40.0]])
    np.testing.assert_allclose(probe @ M_est[:3, :3].T + M_est[:3, 3], probe @ M[:3, :3].T + M[:3, 3], atol=1.0)
    assert info["num_inliers"] == int(mask.sum())


def test_ransac_rejects_mirrored_solution():
    """A reflection is never a plausible patient-to-patient mapping."""
    rng = np.random.default_rng(3)
    src = rng.uniform(-100, 100, size=(50, 3))
    dst = src * np.array([-1.0, 1.0, 1.0])  # mirrored in x
    M_est, mask, _ = ransac_affine(src, dst, threshold_mm=5.0)
    assert M_est is None
    assert not mask.any()


def test_ransac_handles_too_few_points():
    src = np.zeros((3, 3))
    M_est, mask, info = ransac_affine(src, src)
    assert M_est is None and mask.shape == (3,) and info["num_inliers"] == 0
