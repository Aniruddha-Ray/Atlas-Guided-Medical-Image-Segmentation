import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("SimpleITK")

from scipy.ndimage import gaussian_filter, shift as nd_shift  # noqa: E402

from atlas.matching import ncc_block_match, window_intensities  # noqa: E402


def _textured_volume(shape=(60, 60, 60), seed=0):
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.standard_normal(shape), sigma=1.5).astype(np.float32)


def test_ncc_block_match_recovers_known_integer_shift():
    moving = _textured_volume()
    true_shift = np.array([3, -2, 4])
    # fixed[x] = moving[x - shift]  ->  landmark at c in moving is at c + shift in fixed
    fixed = nd_shift(moving, true_shift, order=0, mode="nearest").astype(np.float32)

    centers = np.array([[30, 30, 30], [25, 35, 28], [34, 27, 33]])
    offsets, peak, valid = ncc_block_match(fixed, moving, centers, patch_radius=4, search_radius=6, device="cpu")

    assert valid.all()
    for o in offsets:
        np.testing.assert_array_equal(o, true_shift)
    assert (peak > 0.99).all()


def test_ncc_is_invariant_to_linear_intensity_change():
    moving = _textured_volume(seed=1)
    fixed = 3.0 * moving + 7.0  # contrast/brightness change, no shift
    centers = np.array([[30, 30, 30]])
    offsets, peak, _ = ncc_block_match(fixed.astype(np.float32), moving, centers, search_radius=4, device="cpu")
    np.testing.assert_array_equal(offsets[0], [0, 0, 0])
    assert peak[0] > 0.99


def test_flat_template_is_flagged_invalid():
    moving = np.zeros((40, 40, 40), dtype=np.float32)
    fixed = _textured_volume((40, 40, 40))
    _, _, valid = ncc_block_match(fixed, moving, np.array([[20, 20, 20]]), device="cpu")
    assert not valid[0]


def test_landmark_near_volume_edge_does_not_crash():
    moving = _textured_volume((30, 30, 30))
    offsets, _, _ = ncc_block_match(moving, moving, np.array([[0, 0, 0], [29, 29, 29]]), search_radius=5, device="cpu")
    assert offsets.shape == (2, 3)


def test_window_intensities_range():
    arr = np.array([-1000.0, -200.0, 50.0, 300.0, 1500.0])
    out = window_intensities(arr, (-200.0, 300.0))
    np.testing.assert_allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0])
