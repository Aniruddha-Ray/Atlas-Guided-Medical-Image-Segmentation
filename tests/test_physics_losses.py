import numpy as np
import pytest
import torch
from monai.data import MetaTensor

from training.physics_losses import (edge_aware_loss, gaussian_blur3d, hu_consistency_loss, organ_hu_stats,
                                     spacing_from_affine, to_hu)

C = 13
S = 24


def to_unit(hu):
    return (np.clip(hu, -1000, 1000) + 1000) / 2000


def phantom(seed=0):
    """Organ 1 (60 HU) occupies x < 12, the rest is fat (-100 HU), with 5 HU noise. Returns image, prob, conf, mask."""
    rng = np.random.default_rng(seed)
    hu = np.full((S, S, S), -100.0)
    hu[: S // 2] = 60.0
    hu += rng.normal(0, 5, hu.shape)
    mask = np.zeros((S, S, S), bool)
    mask[: S // 2] = True
    image = torch.tensor(to_unit(hu), dtype=torch.float32)[None, None]
    prob = torch.zeros(1, C, S, S, S)
    prob[0, 1][torch.tensor(mask)] = 1.0
    prob[0, 0][~torch.tensor(mask)] = 1.0
    conf = torch.ones(1, 1, S, S, S)
    return image, prob, conf, torch.tensor(mask)


def logits_for(mask, strength=10.0):
    logits = torch.zeros(1, C, S, S, S)
    logits[0, 1][mask] = strength
    logits[0, 0][~mask] = strength
    return logits.requires_grad_(True)


def test_to_hu_inverts_the_pipeline_scaling():
    assert torch.allclose(to_hu(torch.tensor([0.0, 0.5, 1.0])), torch.tensor([-1000.0, 0.0, 1000.0]))


def test_organ_stats_come_from_the_atlas_core_and_are_robust():
    image, prob, conf, _ = phantom()
    mu, sigma, valid = organ_hu_stats(to_hu(image), prob, conf, sigma_min=1.0)
    assert valid[0, 1] and not valid[0, 0] and not valid[0, 2:].any()
    assert abs(mu[0, 1].item() - 60.0) < 1.0
    assert abs(sigma[0, 1].item() - 5.0) < 0.5  # 1.4826 * MAD estimates the 5 HU noise
    _, sigma_floor, _ = organ_hu_stats(to_hu(image), prob, conf, sigma_min=10.0)
    assert sigma_floor[0, 1].item() == pytest.approx(10.0)


def test_hu_loss_penalises_predicting_the_organ_on_wrong_tissue():
    image, prob, conf, mask = phantom()
    right, _ = hu_consistency_loss(logits_for(mask), image, prob, conf)
    too_big = mask.clone()
    too_big[: S // 2 + 6] = True  # organ spills 6 slices into fat (-100 HU, i.e. 16 sigma away)
    wrong, n = hu_consistency_loss(logits_for(too_big), image, prob, conf)
    assert n == 1
    assert right.item() < 0.6  # only noise left: about E[z^2] / 2 = 0.5 * (5 / sigma)^2 with sigma = 10 HU floor
    assert wrong.item() > 5 * right.item()


def test_hu_loss_is_switched_off_without_atlas_evidence_and_keeps_a_gradient_path():
    image, prob, _, mask = phantom()
    no_conf = torch.zeros(1, 1, S, S, S)  # what prior dropout or a failed registration looks like
    logits = logits_for(mask)
    loss, n = hu_consistency_loss(logits, image, prob, no_conf)
    assert n == 0 and loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None and torch.all(logits.grad == 0)


def test_hu_loss_gradient_pushes_wrong_voxels_away_from_the_organ():
    image, prob, conf, mask = phantom()
    too_big = mask.clone()
    too_big[: S // 2 + 6] = True
    logits = logits_for(too_big, strength=2.0)
    loss, _ = hu_consistency_loss(logits, image, prob, conf)
    loss.backward()
    g = logits.grad[0, 1]
    assert torch.isfinite(logits.grad).all()
    assert g[S // 2 + 2].mean() > 0  # fat voxels predicted as organ: gradient descent lowers the organ logit
    assert g[S // 2 + 2].mean() > g[: S // 2].mean()


def test_gaussian_blur_preserves_constant_images_and_mass():
    x = torch.full((1, 1, 10, 11, 12), 0.3)
    assert torch.allclose(gaussian_blur3d(x, 1.0), x, atol=1e-6)
    y = torch.zeros(1, 1, 15, 15, 15)
    y[0, 0, 7, 7, 7] = 1.0
    blurred = gaussian_blur3d(y, 1.0)
    assert blurred.sum().item() == pytest.approx(1.0, abs=1e-4)
    assert blurred[0, 0, 7, 7, 7] < 1.0 and blurred[0, 0, 7, 7, 8] > 0


def test_edge_loss_prefers_boundaries_on_ct_edges():
    image, _, _, mask = phantom()
    on_edge = edge_aware_loss(logits_for(mask), image)
    shifted = mask.clone()
    shifted[:] = False
    shifted[: S // 2 - 6] = True  # same boundary shape, moved 6 voxels into the flat organ interior
    off_edge = edge_aware_loss(logits_for(shifted), image)
    assert off_edge.item() > 3 * on_edge.item()


def test_edge_loss_uses_physical_spacing():
    image, _, _, mask = phantom()
    off_edge = torch.zeros_like(mask)
    off_edge[: S // 2 - 6] = True  # a boundary in a flat region, so its weight is not ~0
    logits = logits_for(off_edge)
    base = edge_aware_loss(logits, image, torch.tensor([[1.0, 1.0, 1.0]]))
    coarse = edge_aware_loss(logits, image, torch.tensor([[2.0, 2.0, 2.0]]))
    # The edge weight is scale-free (kappa is the scan's own median gradient); the prediction gradient halves.
    assert coarse.item() == pytest.approx(base.item() / 2, rel=0.05)


def test_edge_loss_gradient_is_finite_for_a_flat_prediction():
    image, _, _, _ = phantom()
    logits = torch.zeros(1, C, S, S, S, requires_grad=True)
    loss = edge_aware_loss(logits, image)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_losses_accept_half_precision_logits():
    image, prob, conf, mask = phantom()
    logits = logits_for(mask).detach().half()
    hu, _ = hu_consistency_loss(logits, image, prob, conf)
    edge = edge_aware_loss(logits, image)
    assert hu.dtype == torch.float32 and edge.dtype == torch.float32
    assert torch.isfinite(hu) and torch.isfinite(edge)


def test_spacing_from_affine_reads_column_norms_per_sample():
    aff = torch.stack([torch.diag(torch.tensor([0.8, 0.8, 2.5, 1.0])), torch.diag(torch.tensor([-1.5, 1.5, 1.5, 1.0]))])
    img = MetaTensor(torch.zeros(2, 1, 4, 4, 4), affine=aff)
    sp = spacing_from_affine(img)
    assert torch.allclose(sp, torch.tensor([[0.8, 0.8, 2.5], [1.5, 1.5, 1.5]], dtype=torch.float64))
    assert torch.allclose(spacing_from_affine(torch.zeros(3, 1, 4, 4, 4)), torch.ones(3, 3, dtype=torch.float64))
