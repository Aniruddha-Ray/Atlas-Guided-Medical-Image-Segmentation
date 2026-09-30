import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
nib = pytest.importorskip("nibabel")

from atlas.geometry import to_xyz_array  # noqa: E402
from atlas.label_warping import displacement_transform, save_label_stack, vote_counts, warp_label  # noqa: E402


def _label_image():
    img = sitk.Image([40, 40, 40], sitk.sitkUInt8)
    img.SetSpacing((2.0, 2.0, 2.0))
    img.SetOrigin((-40.0, -40.0, -40.0))
    arr = np.zeros((40, 40, 40), dtype=np.uint8)  # (z, y, x)
    arr[15:25, 15:25, 15:25] = 3
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def test_identity_mapping_preserves_label():
    label = _label_image()
    tx = displacement_transform(lambda p: p, label, field_spacing_mm=6.0)
    warped = warp_label(label, label, tx)
    np.testing.assert_array_equal(sitk.GetArrayFromImage(warped), sitk.GetArrayFromImage(label))


def test_translation_mapping_shifts_label():
    """mapping is target -> atlas: target point p looks up atlas point p + d."""
    label = _label_image()
    d = np.array([6.0, 0.0, 0.0])  # 3 voxels in +x
    tx = displacement_transform(lambda p: p + d, label, field_spacing_mm=6.0)
    warped = to_xyz_array(warp_label(label, label, tx))
    orig = to_xyz_array(label)
    # target voxel x reads atlas voxel x+3 -> the organ appears 3 voxels lower in x
    np.testing.assert_array_equal(warped[12:22, 15:25, 15:25], orig[15:25, 15:25, 15:25])
    assert warped[23, 20, 20] == 0


def test_save_label_stack_round_trip(tmp_path):
    label = _label_image()
    path = tmp_path / "stack.nii.gz"
    stack = save_label_stack([label, label], label, path)
    assert stack.shape == (40, 40, 40, 2)
    back = nib.load(str(path))
    assert back.shape == (40, 40, 40, 2)
    np.testing.assert_array_equal(np.asarray(back.dataobj), stack)


def test_vote_counts():
    stack = np.array([[[[1, 1, 2]]]], dtype=np.uint8)  # one voxel, 3 atlases
    counts = vote_counts(stack, num_classes=3)
    assert counts.shape == (3, 1, 1, 1)
    np.testing.assert_array_equal(counts[:, 0, 0, 0], [0, 2, 1])
