"""
atlas/label_registration.py (label-to-label registration, anatomy-weighted fusion, prior channels) and
scripts/eval_seg.py (Dice, HD95) on synthetic label maps with known geometry.
"""
import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from atlas.label_registration import (NUM_ORGANS, LabelRegConfig, fuse_atlases, onehot_organs,  # noqa: E402
                                      prepare_atlas, prior_channels)
from scripts.eval_seg import hd95  # noqa: E402

SPACING = (1.5, 1.5, 3.0)  # x, y, z
SIZE = (100, 100, 50)  # 150 x 150 x 150 mm


def _organs(scale=1.0, shift_mm=(0.0, 0.0, 0.0), drop=(), boxes=False):
    """12 ellipsoid (or box) 'organs' on a 3 x 2 x 2 layout, as a sitk label image; scaled about the centre, then shifted."""
    zz, yy, xx = np.meshgrid(*[np.arange(n) * s for n, s in zip(SIZE[::-1], SPACING[::-1])], indexing="ij")
    lab = np.zeros(SIZE[::-1], dtype=np.uint8)
    centre = np.array([75.0, 75.0, 75.0])
    k = 0
    for cx in (35.0, 75.0, 115.0):
        for cy in (50.0, 100.0):
            for cz in (50.0, 100.0):
                k += 1
                if k in drop:
                    continue
                c = centre + scale * (np.array([cx, cy, cz]) - centre) + np.array(shift_mm)
                r = scale * np.array([12.0 + k % 3 * 2, 9.0 + k % 2 * 3, 11.0])
                u = [np.abs(v - cc) / rr for v, cc, rr in zip((xx, yy, zz), c, r)]
                inside = np.maximum(np.maximum(u[0], u[1]), u[2]) <= 0.8 if boxes else u[0] ** 2 + u[1] ** 2 + u[2] ** 2 <= 1.0
                lab[inside] = k
    img = sitk.GetImageFromArray(lab)
    img.SetSpacing(SPACING)
    return img


def _dice(a, b):
    return 2.0 * np.logical_and(a, b).sum() / (a.sum() + b.sum())


@pytest.fixture(scope="module")
def fused():
    cfg = LabelRegConfig(top_k=2)
    atlases = [prepare_atlas("a1", _organs(1.08, (6.0, -4.0, 3.0)), cfg),
               prepare_atlas("a2", _organs(0.93, (-5.0, 3.0, -6.0)), cfg),
               prepare_atlas("a3", _organs(1.0, (40.0, 0.0, 0.0), boxes=True), cfg)]  # other shapes: no affine fits
    target = _organs()
    pred = _organs(drop=(5,))  # the "first-stage prediction" misses organ 5
    P, info = fuse_atlases(pred, atlases, cfg)
    return target, pred, P, info


def test_fused_prior_matches_the_target_anatomy(fused):
    target, _, P, info = fused
    truth = sitk.GetArrayFromImage(target)
    full = np.concatenate([np.clip(1 - P.sum(0, keepdims=True), 0, 1), P])
    am = full.argmax(0)
    d = [_dice(am == k, truth == k) for k in range(1, NUM_ORGANS + 1) if k != 5]
    assert min(d) > 0.8, d
    assert P.shape == (NUM_ORGANS,) + truth.shape and P.min() >= 0 and P.max() <= 1.0 + 1e-5


def test_organ_missing_from_the_prediction_is_placed_by_the_anatomy_of_the_others(fused):
    target, _, P, info = fused
    truth = sitk.GetArrayFromImage(target) == 5
    assert 5 not in info["organs_in_prediction"]
    assert _dice(P[4] >= 0.5, truth) > 0.5


def test_weights_follow_the_fit(fused):
    _, _, _, info = fused
    for k, chosen in info["fused"].items():
        assert "a3" not in [c["atlas"] for c in chosen], (k, chosen)  # top-2 of 3: the ill-fitting atlas is left out


def test_prior_channels_ranges_and_plain_cascade_confidence():
    pred = sitk.GetArrayFromImage(_organs())
    ch = prior_channels(onehot_organs(pred), SPACING[::-1])
    assert ch.shape == (2 * NUM_ORGANS + 1,) + pred.shape
    assert ch.min() >= 0 and ch.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(ch[-1], 1.0, atol=1e-5)  # one-hot: no uncertainty
    near_liver = ch[NUM_ORGANS + 5]
    assert near_liver[pred == 6].min() == 1.0 and near_liver[pred == 0].min() < 0.2


def test_hd95_of_a_shifted_box():
    a = np.zeros((30, 40, 40), bool)
    a[10:20, 10:30, 10:30] = True
    b = np.roll(a, 3, axis=2)  # 3 voxels along x
    assert hd95(a, a, (2.0, 1.0, 1.0)) == 0.0
    assert hd95(a, b, (2.0, 1.0, 1.0)) == pytest.approx(3.0, abs=0.01)
