import numpy as np

from atlas.tps import ThinPlateSpline


def _random_points(n, seed=0, scale=50.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n, 3))


def test_exact_interpolation_recovers_pure_affine_transform():
    """
    If the correspondences come from a pure affine map with no local warp,
    an exact-interpolation (regularization=0) TPS should reproduce that
    affine map almost exactly everywhere, not just at the control points --
    the warp weights should be ~0.
    """
    rng = np.random.default_rng(1)
    A = np.eye(3) + 0.1 * rng.standard_normal((3, 3))
    b = np.array([5.0, -3.0, 2.0])

    source = _random_points(20, seed=2)
    target = source @ A + b

    tps = ThinPlateSpline(source, target, regularization=0.0)

    # Held-out points, not used as control points.
    held_out = _random_points(15, seed=3)
    predicted = tps.transform(held_out)
    expected = held_out @ A + b
    np.testing.assert_allclose(predicted, expected, atol=1e-6)

    # The nonlinear warp contribution should be negligible.
    assert np.abs(tps.weights).max() < 1e-6


def test_exact_interpolation_passes_through_control_points():
    source = _random_points(12, seed=4)
    # A nonlinear (non-affine) target: sinusoidal perturbation on top of identity.
    target = source + 3.0 * np.sin(source / 10.0)

    tps = ThinPlateSpline(source, target, regularization=0.0)
    recovered = tps.transform(source)
    np.testing.assert_allclose(recovered, target, atol=1e-6)


def test_regularization_produces_smoothing_not_exact_interpolation():
    """
    One control point is a deliberate outlier (large local perturbation).
    Exact interpolation must pass through it exactly; a smoothing spline
    should NOT, since the whole point of smoothing is to not let one bad
    correspondence force a sharp local bend.
    """
    source = _random_points(15, seed=5)
    target = source.copy()
    outlier_idx = 0
    target[outlier_idx] += np.array([40.0, 0.0, 0.0])  # a big, implausible local jump

    exact_tps = ThinPlateSpline(source, target, regularization=0.0)
    smooth_tps = ThinPlateSpline(source, target, regularization=20.0)

    exact_residual = np.linalg.norm(exact_tps.transform(source[[outlier_idx]])[0] - target[outlier_idx])
    smooth_residual = np.linalg.norm(smooth_tps.transform(source[[outlier_idx]])[0] - target[outlier_idx])

    assert exact_residual < 1e-6
    assert smooth_residual > 1.0  # smoothing spline does not chase the outlier


def test_jacobian_determinant_of_pure_affine_matches_analytic_value():
    rng = np.random.default_rng(6)
    A = np.eye(3) + 0.05 * rng.standard_normal((3, 3))
    b = np.array([1.0, 1.0, 1.0])

    source = _random_points(10, seed=7)
    target = source @ A + b
    tps = ThinPlateSpline(source, target, regularization=0.0)

    query_points = _random_points(5, seed=8)
    det_j = tps.jacobian_determinant(query_points)
    expected_det = np.linalg.det(A)
    np.testing.assert_allclose(det_j, expected_det, atol=1e-3)


def test_folding_is_detectable_via_negative_jacobian():
    """A synthetic, deliberately-inverting correspondence set should produce
    a negative Jacobian determinant somewhere -- confirming the folding
    check actually catches something rather than always reading positive."""
    # 4 points of a small tetrahedron, target = point-reflected through the
    # centroid (orientation-reversing), which any smooth interpolant must
    # fold to reach.
    source = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    )
    centroid = source.mean(axis=0)
    target = 2 * centroid - source  # point reflection: orientation-reversing

    tps = ThinPlateSpline(source, target, regularization=0.0)
    det_j = tps.jacobian_determinant(np.array([centroid]))
    assert det_j[0] < 0


def test_raises_on_too_few_correspondences():
    import pytest

    source = _random_points(3, seed=9)
    target = _random_points(3, seed=10)
    with pytest.raises(ValueError):
        ThinPlateSpline(source, target)
