"""
nnunet_ext (atlas prior on nnU-Net v2) and scripts/nnunet_make_datasets.py: the fused logits, the protection of the
prior channels from nnU-Net's intensity augmentation, prior dropout, and the alignment of the written prior files.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("nnunetv2")
sitk = pytest.importorskip("SimpleITK")

from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform  # noqa: E402
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms  # noqa: E402
from batchgeneratorsv2.transforms.utils.random import RandomTransform  # noqa: E402
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402

from nnunet_ext.nnUNetTrainerAtlasPrior import (LOG_PRIOR_EPS, PRIOR_SCALE, AtlasPriorFusion, PriorDropout,  # noqa: E402
                                                PutBackPriors, SetAsidePriors, prior_layout, protect_priors)

C, K = 13, 12
N_IN = 1 + 2 * K + 1  # CT + P + N + w


class _Net(torch.nn.Module):
    """Stands in for nnU-Net's network: fixed logits at two deep-supervision scales; records its input."""

    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.deep_supervision = True
        self.seen = None

    def forward(self, x):
        self.seen = x
        b, _, *s = x.shape
        full = torch.arange(C, dtype=torch.float32).view(1, C, 1, 1, 1).expand(b, C, *s).clone()
        return [full, full[:, :, ::2, ::2, ::2].clone()] if self.decoder.deep_supervision else full


def _inputs(b=1, s=(8, 8, 8), seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(b, C, *s, generator=g)
    p = p / p.sum(1, keepdim=True)  # class probabilities summing to 1
    near = torch.rand(b, K, *s, generator=g)
    w = torch.rand(b, 1, *s, generator=g)
    ct = torch.randn(b, 1, *s, generator=g)
    prior = torch.cat([p[:, 1:], near, w], 1)
    return torch.cat([ct, torch.round(prior * PRIOR_SCALE)], 1), p, w


def test_layout():
    assert prior_layout(N_IN, C) == "pnw"
    assert prior_layout(1 + K + 1, C) == "pw"
    with pytest.raises(ValueError):
        prior_layout(5, C)


def test_fusion_adds_w_log_prior_at_every_scale_and_rescales_the_input():
    net = _Net()
    model = AtlasPriorFusion(net, N_IN, C, gamma_init=1.0)
    x, p, w = _inputs()
    out = model(x)
    assert isinstance(out, list) and len(out) == 2
    q = (x[:, 1:] / PRIOR_SCALE)  # what the stored uint8 values mean
    p_org = q[:, :K]
    p_full = torch.cat([(1 - p_org.sum(1, keepdim=True)).clamp(0, 1), p_org], 1)
    expected = net(x)[0] + q[:, -1:] * torch.log(p_full + LOG_PRIOR_EPS)
    model(x)
    torch.testing.assert_close(out[0], expected)
    assert out[1].shape[2:] == (4, 4, 4)
    # the network itself sees the priors in [0, 1], not 0..250
    assert net.seen[:, 1:].max() <= 1.0 and torch.equal(net.seen[:, :1], x[:, :1])
    # nnU-Net switches deep supervision through `.decoder`
    model.decoder.deep_supervision = False
    assert torch.is_tensor(model(x))


def test_gamma_zero_is_the_plain_network():
    net = _Net()
    model = AtlasPriorFusion(net, N_IN, C, gamma_init=0.0)
    x, _, _ = _inputs()
    torch.testing.assert_close(model(x)[0], net(x)[0])


def test_untrained_fusion_with_flat_logits_is_the_atlas_vote():
    class Flat(_Net):
        def forward(self, x):
            b, _, *s = x.shape
            return torch.zeros(b, C, *s)

    x, p, w = _inputs()
    x[:, -1:] = PRIOR_SCALE  # full confidence
    out = AtlasPriorFusion(Flat(), N_IN, C)(x)
    q = x[:, 1:1 + K] / PRIOR_SCALE
    full = torch.cat([1 - q.sum(1, keepdim=True), q], 1)
    top2 = full.topk(2, dim=1).values
    clear = (top2[:, 0] - top2[:, 1]) > 1e-4  # skip exact ties (P_bg = 1 - sum P_k rounds differently)
    assert clear.float().mean() > 0.5  # random P quantised to 1/250 ties often
    assert torch.equal(out.argmax(1)[clear], full.argmax(1)[clear])


def test_intensity_augmentation_leaves_the_priors_alone():
    noise = ComposeTransforms([RandomTransform(GaussianNoiseTransform(noise_variance=(0.5, 0.5)), apply_probability=1.0)])
    t = protect_priors(noise, C, prior_dropout=0.0)
    img = torch.rand(N_IN, 6, 6, 6)
    out = t(image=img.clone(), segmentation=torch.zeros(1, 6, 6, 6))
    assert out["image"].shape == img.shape and "_atlas_priors" not in out
    assert torch.equal(out["image"][1:], img[1:])  # priors untouched
    assert not torch.equal(out["image"][:1], img[:1])  # CT got the noise


def test_protect_priors_on_nnunets_own_pipeline():
    base = nnUNetTrainer.get_training_transforms(
        (32, 32, 32), (-0.5, 0.5), None, (0, 1, 2), False)
    t = protect_priors(base, C, prior_dropout=0.3)
    names = [type(x).__name__ for x in t.transforms]
    i_set, i_put = names.index("SetAsidePriors"), names.index("PutBackPriors")
    assert names[0] == "SpatialTransform" and 0 < i_set < i_put and names[i_put + 1] == "PriorDropout"
    assert names.index("MirrorTransform") > i_put  # mirroring still moves CT and priors together
    torch.manual_seed(0)
    img = torch.cat([torch.randn(1, 40, 40, 40), torch.full((N_IN - 1, 40, 40, 40), 100.0)])
    out = t(image=img, segmentation=torch.zeros(1, 40, 40, 40))
    assert out["image"].shape == (N_IN, 32, 32, 32)
    pr = out["image"][1:]
    # spatial resampling / zero padding may change a constant map only towards 0 (the neutral value), never past 100
    assert pr.max() <= 100.0 + 1e-3 and pr.min() >= -1e-3


def test_prior_dropout_values():
    d = PriorDropout(1.0, C)
    img = torch.full((N_IN, 3, 3, 3), 7.0)
    out = d(image=img, segmentation=torch.zeros(1, 3, 3, 3))["image"]
    assert torch.all(out[0] == 7.0)  # CT kept
    torch.testing.assert_close(out[1:1 + K], torch.full((K, 3, 3, 3), PRIOR_SCALE / C))
    assert torch.all(out[1 + K:] == 0)  # nearness and confidence: none
    assert torch.all(img[1] == 7.0)  # input not modified in place
    kept = PriorDropout(0.0, C)(image=img.clone(), segmentation=torch.zeros(1, 3, 3, 3))["image"]
    assert torch.equal(kept, img)


def test_written_prior_files_align_with_the_label(tmp_path):
    from scripts.nnunet_make_datasets import write_prior_files
    from tests.test_atlas_priors import _write_synthetic_case

    case = _write_synthetic_case(tmp_path)
    n = write_prior_files(case["atlas_labels"], case["image"], str(tmp_path / "out"), "pnw")
    assert n == N_IN - 1
    ct = sitk.ReadImage(case["image"])
    lab = sitk.GetArrayFromImage(sitk.ReadImage(case["label"]))
    for j, organ in ((6, 6), (1, 1)):
        ch = sitk.ReadImage(str(tmp_path / f"out_{j:04d}.nii.gz"))
        assert ch.GetSize() == ct.GetSize() and np.allclose(ch.GetSpacing(), ct.GetSpacing())
        assert np.allclose(ch.GetOrigin(), ct.GetOrigin()) and np.allclose(ch.GetDirection(), ct.GetDirection())
        arr = sitk.GetArrayFromImage(ch)
        assert arr.dtype == np.uint8 and arr.max() == 250
        pred, gt = arr >= 125, lab == organ
        dice = 2 * (pred & gt).sum() / (pred.sum() + gt.sum())
        assert dice > 0.85, (organ, dice)
    near_liver = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / f"out_{K + 6:04d}.nii.gz")))
    assert near_liver[lab == 6].min() >= 245  # nearness 1 inside the organ
    assert near_liver[lab == 0].min() < 50  # falls off away from it


def test_coarsened_grid_keeps_ct_label_and_priors_aligned(tmp_path):
    from scripts.nnunet_make_datasets import output_grid, write_image_and_label, write_prior_files
    from tests.test_atlas_priors import ORGAN_HU, _write_synthetic_case

    case = _write_synthetic_case(tmp_path)  # 2.5 x 2.5 x 5 mm
    grid = output_grid(case["image"], 4.0)
    assert np.allclose(grid[2], (4.0, 4.0, 5.0))  # finer axes coarsened, the 5 mm axis kept
    write_image_and_label(case["image"], case["label"], str(tmp_path / "ct.nii.gz"), str(tmp_path / "lab.nii.gz"), grid)
    write_prior_files(case["atlas_labels"], case["image"], str(tmp_path / "p"), "pnw", 4.0)
    ct, lab = sitk.ReadImage(str(tmp_path / "ct.nii.gz")), sitk.ReadImage(str(tmp_path / "lab.nii.gz"))
    pr = sitk.ReadImage(str(tmp_path / "p_0006.nii.gz"))
    for img in (lab, pr):
        assert img.GetSize() == ct.GetSize() and np.allclose(img.GetOrigin(), ct.GetOrigin())
    assert sitk.GetArrayFromImage(ct).dtype == np.int16
    ct_a, lab_a, pr_a = (sitk.GetArrayFromImage(x) for x in (ct, lab, pr))
    bright = ct_a > (40 + ORGAN_HU) / 2
    for a, b in ((pr_a >= 125, lab_a == 6), (bright, lab_a == 6)):
        assert 2 * (a & b).sum() / (a.sum() + b.sum()) > 0.85
