import pytest
import torch

from models.swin_atlas import LOG_PRIOR_EPS, AtlasAwareSwinUNETR

SIZE, C = 64, 13


def _model(**kw):
    torch.manual_seed(0)
    return AtlasAwareSwinUNETR(1, C, SIZE, feature_size=12, hidden_size=64, **kw).eval()


def _inputs(b=1):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(b, 1, SIZE, SIZE, SIZE, generator=g)
    prob = torch.softmax(torch.randn(b, C, SIZE, SIZE, SIZE, generator=g), 1)
    dist = torch.rand(b, C - 1, SIZE, SIZE, SIZE, generator=g)
    conf = torch.rand(b, 1, SIZE, SIZE, SIZE, generator=g)
    return x, prob, dist, conf


def test_forward_shape_and_new_parameters():
    m = _model()
    with torch.no_grad():
        out = m(*_inputs())
    assert out.shape == (1, C, SIZE, SIZE, SIZE) and torch.isfinite(out).all()
    new = m.new_parameters()
    assert new["cross_gates"].shape == (3,) and torch.all(new["cross_gates"] == 1) and new["gamma"].shape == (C,)


def test_output_fusion_adds_gamma_w_log_p():
    m = _model()
    x, prob, dist, conf = _inputs()
    with torch.no_grad():
        base = m(x, prob, dist, conf)
        m.output_fusion_weight.fill_(1.0)
        fused = m(x, prob, dist, conf)
    assert torch.allclose(fused - base, conf * torch.log(prob + LOG_PRIOR_EPS), atol=1e-4)


def test_cross_attention_reads_the_prior_tokens():
    m = _model()
    x, prob, dist, conf = _inputs()
    with torch.no_grad():
        a = m(x, prob, dist, conf)
        m.cross["3"].gate.zero_()
        b = m(x, prob, dist, conf)
    assert not torch.allclose(a, b)


def test_gradients_reach_gate_gamma_and_token_mlp():
    m = _model().train()
    m.output_fusion_weight.data.fill_(1.0)
    x, prob, dist, conf = _inputs()
    m(x, prob, dist, conf).mean().backward()
    assert m.output_fusion_weight.grad.abs().sum() > 0
    assert all(c.gate.grad is not None and c.gate.grad.abs() > 0 for c in m.cross.values())
    assert m.prior_tokens.mlp[0].weight.grad.abs().sum() > 0


def test_flags_off_drops_the_prior_paths():
    m = _model(conf_gating=False, voxel_prior=False, output_fusion=False)
    assert not hasattr(m, "cross") and not hasattr(m, "output_fusion_weight")
    with torch.no_grad():
        out = m(*_inputs())
    assert out.shape[1] == C


def _cored_inputs():
    """Inputs where organs 1-3 have a confident prior core with their own intensity (organ k at (0.3 + 0.1 k))."""
    x, prob, dist, conf = _inputs()
    x = torch.full_like(x, 0.5)
    prob = torch.full_like(prob, 1.0 / C)
    conf = torch.zeros_like(conf)
    for k in (1, 2, 3):
        sl = (slice(None), slice(None), slice(8 * k, 8 * k + 6), slice(8, 30), slice(8, 30))
        prob[sl[:2] + sl[2:]] = 0.0
        prob[:, k][(slice(None),) + sl[2:]] = 0.97
        prob[:, 0][(slice(None),) + sl[2:]] = 0.03
        conf[(slice(None), slice(None)) + sl[2:]] = 1.0
        x[(slice(None), slice(None)) + sl[2:]] = 0.3 + 0.1 * k
    return x, prob, dist, conf


def test_hu_likelihood_maps_follow_the_atlas_core():
    from training.physics_losses import hu_likelihood_maps
    x, prob, dist, conf = _cored_inputs()
    ll, valid = hu_likelihood_maps(x, prob, conf)
    assert ll.shape == (1, C - 1, SIZE, SIZE, SIZE) and valid[0, :3].all() and not valid[0, 3:].any()
    core1 = (slice(None), 0, slice(8, 14), slice(8, 30), slice(8, 30))
    assert torch.allclose(ll[core1], torch.zeros(1), atol=1e-6) or ll[core1].abs().max() < 1e-4  # at its own median
    assert ll[0, 0, 16:22, 8:30, 8:30].max() < -1  # organ 2's voxels are implausible for organ 1
    _, none_valid = hu_likelihood_maps(x, torch.full_like(prob, 1.0 / C), torch.zeros_like(conf))
    assert not none_valid.any()  # prior dropped or no-prior validation: no organ has a core


def test_hu_input_model_starts_as_the_checkpoint_and_beta_acts():
    from models.swin_atlas import load_state_dict_expanding
    base, big = _model(), _model(hu_input=True)
    grown, absent = load_state_dict_expanding(big, base.state_dict())
    assert grown and set(absent) == {"hu_fusion_weight"}
    x, prob, dist, conf = _cored_inputs()
    with torch.no_grad():
        ref, out = base(x, prob, dist, conf), big(x, prob, dist, conf)
        assert torch.allclose(ref, out, atol=1e-4)  # zero new channels and beta 0: identical logits
        big.hu_fusion_weight.fill_(1.0)
        moved = big(x, prob, dist, conf)
    diff = (moved - out).abs().amax(dim=(0, 2, 3, 4))
    assert diff[0] == 0 and (diff[1:4] > 0).all() and (diff[4:] == 0).all()  # only organs with a core, never background
    assert big.new_parameters()["beta"].shape == (C - 1,)


def test_hu_input_gradient_reaches_beta_and_new_channels():
    from models.swin_atlas import load_state_dict_expanding
    big = _model(hu_input=True).train()
    big.hu_fusion_weight.data.fill_(0.5)
    big(*_cored_inputs()).mean().backward()
    assert big.hu_fusion_weight.grad.abs().sum() > 0
    assert big.encoder1.layer.conv1.conv.weight.grad[:, -(C - 1):].abs().sum() > 0


def test_expanding_loader_rejects_other_mismatches():
    from models.swin_atlas import load_state_dict_expanding
    big = _model(hu_input=True)
    bad = {k: v.clone() for k, v in _model().state_dict().items()}
    bad["extra"] = torch.zeros(1)
    with pytest.raises(KeyError):
        load_state_dict_expanding(big, bad)
    del bad["extra"]
    bad["out.conv.conv.weight"] = torch.zeros(C, 3, 1, 1, 1)
    with pytest.raises(ValueError):
        load_state_dict_expanding(big, bad)
