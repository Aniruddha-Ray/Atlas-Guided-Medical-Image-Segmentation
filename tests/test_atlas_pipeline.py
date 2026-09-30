"""
Shape/value sanity checks for the atlas pipeline (Section 32 of the project
brief): P.shape, 0<=P<=1 and sums to ~1 per voxel across classes, D.shape and
D>=0, and select_atlas_library's determinism. Registration accuracy on real
CT is NOT covered here -- see scripts/validate_registration.py (visual
inspection) for that, per Section 33 of the brief.
"""
import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from atlas.atlas_pipeline import build_atlas_priors, select_atlas_library  # noqa: E402


def _synthetic_atlas(size=24, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(loc=0.3, scale=0.05, size=(size, size, size)).astype(np.float32)
    label = np.zeros((size, size, size), dtype=np.uint8)
    c = size // 2
    r = size // 6
    zz, yy, xx = np.mgrid[0:size, 0:size, 0:size]
    blob = (zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2 <= r**2
    label[blob] = 1
    img[blob] += 0.4
    return img, label


@pytest.mark.slow
def test_build_atlas_priors_shapes_and_ranges():
    target_img, _ = _synthetic_atlas(seed=1)
    atlas_list = [_synthetic_atlas(seed=s) for s in (2, 3)]
    num_classes = 2  # background + one organ

    P, D_map, confidence = build_atlas_priors(target_img, atlas_list, num_classes)

    assert P.shape == (num_classes,) + target_img.shape
    assert D_map.shape == (num_classes,) + target_img.shape
    assert confidence.shape == target_img.shape

    assert P.min() >= 0.0 - 1e-6
    assert P.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(P.sum(axis=0), 1.0, atol=1e-5)

    assert (D_map >= 0).all()
    assert confidence.min() >= 0.0 - 1e-6
    assert confidence.max() <= 1.0 + 1e-6


def test_select_atlas_library_is_deterministic_and_disjoint():
    cases = [{"image": f"img_{i}.nii.gz", "label": f"lbl_{i}.nii.gz"} for i in range(20)]

    atlas_a, targets_a = select_atlas_library(cases, atlas_count=5, seed=42)
    atlas_b, targets_b = select_atlas_library(cases, atlas_count=5, seed=42)
    assert atlas_a == atlas_b
    assert targets_a == targets_b

    assert len(atlas_a) == 5
    assert len(targets_a) == 15
    atlas_images = {c["image"] for c in atlas_a}
    target_images = {c["image"] for c in targets_a}
    assert atlas_images.isdisjoint(target_images)


def test_select_atlas_library_rejects_invalid_counts():
    cases = [{"image": f"img_{i}.nii.gz", "label": f"lbl_{i}.nii.gz"} for i in range(3)]
    with pytest.raises(ValueError):
        select_atlas_library(cases, atlas_count=0)
    with pytest.raises(ValueError):
        select_atlas_library(cases, atlas_count=3)  # must leave >=1 target
