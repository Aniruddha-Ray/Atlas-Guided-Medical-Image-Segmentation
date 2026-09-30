import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from scipy.ndimage import gaussian_filter  # noqa: E402

from atlas.registration import Registration, RegistrationConfig, bspline_refine  # noqa: E402


def test_composite_applies_last_added_transform_first():
    """bspline_refine relies on: composite(p) = landmark_field(bspline(p))."""
    first = sitk.AffineTransform(3)
    first.SetMatrix([2, 0, 0, 0, 2, 0, 0, 0, 2])  # scale x2, meant to be applied FIRST
    then = sitk.TranslationTransform(3, (10.0, 0.0, 0.0))  # applied second

    composite = sitk.CompositeTransform(then)
    composite.AddTransform(first)

    p = (1.0, 2.0, 3.0)
    np.testing.assert_allclose(composite.TransformPoint(p), then.TransformPoint(first.TransformPoint(p)))
    # and it is NOT the other order
    assert not np.allclose(composite.TransformPoint(p), first.TransformPoint(then.TransformPoint(p)))


def _synthetic_ct(shift_mm=(0.0, 0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    arr = gaussian_filter(rng.standard_normal((48, 48, 48)), 3.0)
    arr = (arr / arr.std() * 150.0).astype(np.float32)
    arr[:4], arr[-4:] = -1000, -1000  # some "air" so a body mask exists but most voxels are tissue
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((3.0, 3.0, 3.0))
    img.SetOrigin(tuple(float(s) for s in shift_mm))
    return img


@pytest.mark.slow
def test_bspline_refine_improves_metric_and_does_not_fold():
    target_ct = _synthetic_ct()
    atlas_ct = _synthetic_ct(shift_mm=(4.0, 0.0, 0.0))  # same content, shifted 4 mm
    target = {"ct": target_ct}

    reg = Registration("tps", {}, np.eye(4), None, {})  # landmark stage = identity (deliberately imperfect)
    cfg = RegistrationConfig(bspline_spacing_mm=3.0, bspline_grid_mm=40.0, bspline_iterations=30, bspline_sampling=0.3)
    refined, info = bspline_refine(atlas_ct, target, reg, cfg)

    assert info["metric_after"] <= info["metric_before"]  # MI is minimised (more negative = better)
    assert info["fold_fraction"] <= cfg.max_fold_fraction
    assert refined is not None
