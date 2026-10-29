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
