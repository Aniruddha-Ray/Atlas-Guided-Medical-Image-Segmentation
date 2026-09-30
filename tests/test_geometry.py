import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
nib = pytest.importorskip("nibabel")

from atlas.geometry import (  # noqa: E402
    affine_matrix_to_sitk,
    grid_physical_points,
    index_to_physical,
    physical_to_index,
    sitk_affine_to_matrix,
    sitk_index_to_ras_affine,
    to_xyz_array,
)


def _oblique_image():
    img = sitk.Image([7, 5, 4], sitk.sitkFloat32)
    img.SetSpacing((0.8, 1.3, 2.5))
    img.SetOrigin((-12.0, 30.0, 5.0))
    # Rotation about z by 20 degrees, plus a flipped y axis.
    a = np.deg2rad(20)
    D = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]) @ np.diag([1, -1, 1])
    img.SetDirection(D.ravel().tolist())
    return img


def test_index_to_physical_matches_sitk():
    img = _oblique_image()
    idx = np.array([[0, 0, 0], [3, 2, 1], [6, 4, 3], [1.5, 0.25, 2.75]])
    ours = index_to_physical(img, idx)
    ref = np.array([img.TransformContinuousIndexToPhysicalPoint(tuple(map(float, i))) for i in idx])
    np.testing.assert_allclose(ours, ref, atol=1e-9)
    np.testing.assert_allclose(physical_to_index(img, ours), idx, atol=1e-9)


def test_grid_points_order_matches_get_array_from_image():
    img = _oblique_image()
    pts = grid_physical_points(img)
    sx, sy, sz = img.GetSize()
    # array flat index -> (z, y, x) -> sitk index (x, y, z)
    k = 37
    z, y, x = np.unravel_index(k, (sz, sy, sx))
    expected = img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
    np.testing.assert_allclose(pts[k], expected, atol=1e-9)


def test_to_xyz_array_indexing():
    img = sitk.Image([4, 3, 2], sitk.sitkUInt8)
    img.SetPixel(3, 1, 0, 9)
    arr = to_xyz_array(img)
    assert arr.shape == (4, 3, 2)
    assert arr[3, 1, 0] == 9


def test_ras_affine_round_trips_through_nibabel_and_sitk(tmp_path):
    img = _oblique_image()
    data = np.arange(np.prod(img.GetSize()), dtype=np.uint8).reshape(img.GetSize())  # (x, y, z)
    path = tmp_path / "roundtrip.nii.gz"
    nib.save(nib.Nifti1Image(data, sitk_index_to_ras_affine(img)), str(path))

    back = sitk.ReadImage(str(path))
    np.testing.assert_allclose(back.GetOrigin(), img.GetOrigin(), atol=1e-4)
    np.testing.assert_allclose(back.GetSpacing(), img.GetSpacing(), atol=1e-4)
    np.testing.assert_allclose(back.GetDirection(), img.GetDirection(), atol=1e-4)
    np.testing.assert_array_equal(to_xyz_array(back), data)


def test_affine_matrix_sitk_round_trip():
    M = np.eye(4)
    M[:3, :3] = [[1.1, 0.05, 0.0], [-0.02, 0.95, 0.1], [0.0, 0.03, 1.02]]
    M[:3, 3] = [4.0, -7.0, 2.5]
    tx = affine_matrix_to_sitk(M)
    p = (10.0, -3.0, 22.0)
    np.testing.assert_allclose(tx.TransformPoint(p), M[:3, :3] @ np.array(p) + M[:3, 3], atol=1e-9)

    centered = sitk.AffineTransform(3)
    centered.SetCenter((5.0, 5.0, 5.0))
    centered.SetMatrix(M[:3, :3].ravel().tolist())
    centered.SetTranslation((1.0, 2.0, 3.0))
    M2 = sitk_affine_to_matrix(centered)
    np.testing.assert_allclose(M2[:3, :3] @ np.array(p) + M2[:3, 3], centered.TransformPoint(p), atol=1e-9)
