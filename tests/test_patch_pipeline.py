"""
Patch pipeline (data/transforms.build_patch_transforms) and whole-volume evaluation (training/patch_eval.py):
priors must land on the same voxels as the CT and label in a training patch and in the whole evaluation volume, and the
sliding-window evaluation must split the input channels the way the model expects.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")
pytest.importorskip("SimpleITK")

from data.transforms import PRIOR_QUANT, build_patch_transforms  # noqa: E402
from tests.test_atlas_priors import ORGAN_HU, _dice, _write_synthetic_case  # noqa: E402
from training.patch_eval import DiceTally, atlas_vote_volumes, organ_dice, validate_volumes  # noqa: E402

C = 13
PATCH, SPACING = (32, 32, 32), (4.0, 4.0, 4.0)  # 128 mm patches of the ~300 mm synthetic volume


def _batch(sample):
    return {k: v[None] for k, v in sample.items() if torch.is_tensor(v)}


def test_training_patch_shapes_and_alignment(tmp_path):
    case = _write_synthetic_case(tmp_path)
    tf = build_patch_transforms(PATCH, SPACING, C, train=True)
    for seed in range(6):
        tf.set_random_state(seed=seed)
        out = tf(case)
        assert isinstance(out, list) and len(out) == 1
        s = out[0]
        assert s["image"].shape == (1, *PATCH) and s["image"].dtype == torch.float32
        assert s["label"].shape == (C, *PATCH)
        assert s["prob"].shape == (C, *PATCH) and s["dist"].shape == (C - 1, *PATCH) and s["conf"].shape == (1, *PATCH)
        assert "priors" not in s and "label_cls_indices" not in s
        torch.testing.assert_close(s["prob"].sum(0), torch.ones(PATCH), atol=0.02, rtol=0)  # trilinear keeps sum ~1
        assert s["dist"].min() >= 0 and s["dist"].max() <= 1
        # the stack is the case's own label: wherever an organ is in the patch, the voted prior must cover it
        for organ in (1, 6):
            lab = s["label"][organ] > 0.5
            if lab.sum() > 30:
                assert _dice(s["prob"].argmax(0) == organ, lab) > 0.7, (seed, organ)


def test_crop_is_class_balanced(tmp_path):
    """The spleen is ~0.2 % of the volume; with a uniform choice among background, spleen and liver it should appear in
    roughly a third or more of the patches, not in a few percent."""
    case = _write_synthetic_case(tmp_path)
    tf = build_patch_transforms(PATCH, SPACING, C, train=True)
    hits = 0
    for seed in range(30):
        tf.set_random_state(seed=seed)
        hits += int(tf(case)[0]["label"][1].sum() > 0)
    assert hits >= 8


def test_evaluation_volume_alignment_and_dtypes(tmp_path):
    case = _write_synthetic_case(tmp_path)
    s = build_patch_transforms(PATCH, SPACING, C, train=False)(case)
    shape = tuple(s["image"].shape[1:])
    assert s["label"].shape == (1, *shape) and s["label"].dtype == torch.uint8
    assert s["prob"].shape == (C, *shape) and s["prob"].dtype == torch.float16
    assert s["dist"].shape == (C - 1, *shape) and s["conf"].shape == (1, *shape)
    # the whole volume at 4 mm: ~300 mm field of view -> ~75 voxels, never smaller than a patch
    assert all(n >= p for n, p in zip(shape, PATCH)) and max(shape) > 60

    img = s["image"][0].float()
    ct_organ = img > ((40.0 + 1000) / 2000 + (ORGAN_HU + 1000) / 2000) / 2
    prior_organ = s["prob"].float().argmax(0) == 6
    offset = torch.nonzero(prior_organ).float().mean(0) - torch.nonzero(ct_organ).float().mean(0)
    assert torch.all(offset.abs() < 1.0), offset
    assert _dice(prior_organ, ct_organ) > 0.85

    # the uint8 cache costs at most half a quantisation step
    assert (s["prob"].float() * PRIOR_QUANT - (s["prob"].float() * PRIOR_QUANT).round()).abs().max() < 0.6

    assert atlas_vote_volumes([_batch(s)], C) > 0.8


class _PriorEcho(torch.nn.Module):
    """Logits = 10 * P: its prediction is the atlas vote, so it reveals any mix-up of the input channels."""

    def forward(self, x_in, prob, dist, conf):
        assert x_in.shape[1] == 1 and prob.shape[1] == C and dist.shape[1] == C - 1 and conf.shape[1] == 1
        return 10.0 * prob + 0.0 * (x_in.sum() + dist.sum() + conf.sum())


def test_sliding_window_validation_matches_the_atlas_vote(tmp_path):
    case = _write_synthetic_case(tmp_path)
    b = _batch(build_patch_transforms(PATCH, SPACING, C, train=False)(case))
    val, per = validate_volumes(_PriorEcho(), [b], torch.device("cpu"), PATCH, overlap=0.25)
    assert val == pytest.approx(atlas_vote_volumes([b], C), abs=1e-6)
    assert per[0] is not None and per[5] is not None  # spleen and liver scored
    assert per[1] is None  # right kidney absent from the label: not counted


def test_dice_rule_matches_monai_dicemetric():
    from monai.metrics import DiceMetric

    rng = np.random.default_rng(0)
    preds = [torch.from_numpy(rng.integers(0, 4, (10, 10, 10))) for _ in range(3)]
    labels = [torch.from_numpy(rng.integers(0, 4, (10, 10, 10))) for _ in range(3)]
    labels[1][labels[1] == 2] = 0  # class 2 absent from case 1: must be skipped, not scored 0
    tally = DiceTally(4)
    metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=True)
    oh = lambda t: torch.nn.functional.one_hot(t.long(), 4).permute(3, 0, 1, 2)[None].float()  # noqa: E731
    for p, g in zip(preds, labels):
        tally.add(organ_dice(p, g, 4))
        metric(y_pred=oh(p), y=oh(g))
    ref, nn_ = metric.aggregate()
    mean, per = tally.result()
    np.testing.assert_allclose(per, ref.tolist(), atol=1e-6)
    assert mean == pytest.approx(ref[nn_.bool()].mean().item(), abs=1e-6)
