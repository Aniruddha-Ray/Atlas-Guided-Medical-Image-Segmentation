"""
AtlasPriorsd invariants (brief Section 32) and end-to-end spatial alignment of
the atlas-prior data pipeline (the priors must land on the same voxels as the
CT/label they belong to, including after random augmentation).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
sitk = pytest.importorskip("SimpleITK")

from atlas.geometry import make_reference_grid, resample_to_grid  # noqa: E402
from atlas.label_warping import save_label_stack  # noqa: E402
from data.transforms import AtlasPriorsd, build_transforms  # noqa: E402

C = 13


def test_atlas_priors_invariants():
    stack = np.zeros((4, 20, 20, 20), dtype=np.uint8)
    stack[:, 5:10, 5:10, 5:10] = 3  # all 4 atlases agree on organ 3 here
    stack[:2, 12:16, 12:16, 12:16] = 7  # only half agree on organ 7 here
    out = AtlasPriorsd(keys=["atlas_labels"], num_classes=C)({"atlas_labels": torch.from_numpy(stack)})

    P, D, w = out["prob"], out["dist"], out["conf"]
    assert P.shape == (C, 20, 20, 20)
    assert D.shape == (C - 1, 20, 20, 20)
    assert w.shape == (1, 20, 20, 20)
    assert "atlas_labels" not in out

    torch.testing.assert_close(P.sum(0), torch.ones(20, 20, 20))
    assert P.min() >= 0 and P.max() <= 1
    assert D.min() >= 0 and D.max() <= 1
    assert w.min() >= -1e-6 and w.max() <= 1 + 1e-6

    assert P[3, 7, 7, 7] == 1.0 and w[0, 7, 7, 7] == pytest.approx(1.0)  # full agreement -> confident
    assert P[7, 14, 14, 14] == 0.5 and w[0, 14, 14, 14] < 1.0  # split vote -> less confident
    assert D[3 - 1, 7, 7, 7] == 0.0  # inside organ 3
    assert D[3 - 1, 0, 0, 0] > D[3 - 1, 4, 4, 4] > 0  # grows with distance outside it
    assert torch.all(D[5 - 1] == 1.0)  # organ 5 absent from every atlas -> max distance


def test_distance_uses_voxel_spacing_from_affine():
    monai = pytest.importorskip("monai")
    stack = np.zeros((1, 1, 1, 30), dtype=np.uint8)
    stack[..., 0] = 1
    affine = np.diag([1.0, 1.0, 2.0, 1.0])  # 2 mm along the last axis
    mt_stack = monai.data.MetaTensor(torch.from_numpy(stack), affine=torch.from_numpy(affine))
    out = AtlasPriorsd(keys=["atlas_labels"], num_classes=C, dist_clip_mm=100.0)({"atlas_labels": mt_stack})
    # voxel 10 along z is 10 voxels = 20 mm from the organ -> 20/100
    assert out["dist"][0, 0, 0, 10].item() == pytest.approx(0.20)


ORGAN_HU = 600.0  # bright synthetic organ, so its true location is visible in the resized CT itself


def _write_synthetic_case(tmp_path):
    """
    CT-like image at a realistic scale (~300 mm field of view, anisotropic,
    oblique-free but axis-flipped like the dataset headers), its label, and an
    'atlas stack' that is the label itself resampled onto a 2 mm grid (x3).
    Organ 6 is also bright in the CT, so the test can check the prior against
    where the organ really is in the image, not only against the label.
    """
    size = (120, 100, 60)  # x, y, z voxels at (2.5, 2.5, 5.0) mm -> 300 x 250 x 300 mm
    arr_ct = np.full(size[::-1], -1000.0, dtype=np.float32)  # (z, y, x)
    arr_lab = np.zeros(size[::-1], dtype=np.uint8)
    zz, yy, xx = np.mgrid[0 : size[2], 0 : size[1], 0 : size[0]]
    body = ((xx - 60) / 55.0) ** 2 + ((yy - 50) / 45.0) ** 2 <= 1.0
    arr_ct[body] = 40.0
    liver = ((xx - 42) ** 2 + (yy - 45) ** 2 <= 14**2) & (zz > 15) & (zz < 40)
    spleen = ((xx - 82) ** 2 + (yy - 55) ** 2 <= 8**2) & (zz > 22) & (zz < 34)
    arr_lab[liver] = 6
    arr_lab[spleen] = 1
    arr_ct[liver] = ORGAN_HU

    ct = sitk.GetImageFromArray(arr_ct)
    lab = sitk.GetImageFromArray(arr_lab)
    for img in (ct, lab):
        img.SetSpacing((2.5, 2.5, 5.0))
        img.SetOrigin((150.0, 120.0, -310.0))
        img.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))  # same convention as the dataset headers
    ct_path, lab_path, stack_path = tmp_path / "c_0000.nii.gz", tmp_path / "c.nii.gz", tmp_path / "stack.nii.gz"
    sitk.WriteImage(ct, str(ct_path))
    sitk.WriteImage(lab, str(lab_path))

    grid = make_reference_grid(ct, 2.0)
    lab_2mm = resample_to_grid(lab, grid, sitk.sitkNearestNeighbor, 0)
    save_label_stack([lab_2mm] * 3, grid, stack_path)
    return {"image": str(ct_path), "label": str(lab_path), "atlas_labels": str(stack_path)}


def _dice(a, b):
    return 2.0 * (a & b).sum().item() / max((a.sum() + b.sum()).item(), 1)


@pytest.mark.parametrize("train", [False, True])
def test_atlas_prior_pipeline_is_spatially_aligned(tmp_path, train):
    """
    The stack is built from the case's own label, so with correct affine
    handling the voted prior must sit where organ 6 is in the resized CT.
    A flipped axis or a shifted grid anywhere in the chain would drive the
    overlap towards 0 and the centroid offset to many voxels. Train mode
    also checks that random flips/rotations move image and prior together.
    """
    case = _write_synthetic_case(tmp_path)
    tf = build_transforms((32, 32, 32), C, train=train, prior_source="atlas")
    tf.set_random_state(seed=3)
    out = tf(case)

    assert out["image"].shape == (1, 32, 32, 32)
    assert out["prob"].shape == (C, 32, 32, 32)
    assert out["dist"].shape == (C - 1, 32, 32, 32)

    img = out["image"][0]
    body_level = (40.0 + 1000.0) / 2000.0
    organ_level = (ORGAN_HU + 1000.0) / 2000.0
    ct_organ = img > (body_level + organ_level) / 2  # organ 6 as it appears in the CT
    prior_organ = out["prob"].argmax(0) == 6

    offset = torch.nonzero(prior_organ).float().mean(0) - torch.nonzero(ct_organ).float().mean(0)
    assert torch.all(offset.abs() < 0.75), offset  # sub-voxel on a ~9 mm/voxel grid
    assert _dice(prior_organ, ct_organ) > 0.8

    spleen_prior = out["prob"].argmax(0) == 1
    spleen_label = out["label"].argmax(0) == 1
    # Small organ (radius ~2 voxels at 32^3), compared against the LABEL rather than
    # the CT: both went through nearest-neighbour resize/rotation, and the label's
    # nearest resize is itself ~0.5-0.9 voxel off-centre (PyTorch "nearest" is
    # floor-based). Still clearly aligned; a misaligned axis would give ~0.
    assert _dice(spleen_prior, spleen_label) > 0.5
