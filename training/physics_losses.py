"""
CT-physics regularisers (model_walkthrough.md, Section 6.3).

HU consistency: voxels predicted as organ c should have this scan's HU for organ c. The per-scan, per-organ HU
statistics come only from the confident core of the atlas prior (P_c >= core_p and w >= core_w), never from the
target label, so the term is usable at inference and is not a label leak. Organ HU shifts between scans with
contrast phase and scanner, which is why the statistics are estimated per scan instead of fixed per organ.

    mu_c = median(I over core_c),  sigma_c = max(1.4826 * MAD(I over core_c), sigma_min)
    L_HU = mean over (scan, organ) with a big enough core of
           sum_v yhat_c(v) * huber_delta((I(v) - mu_c) / sigma_c)  /  (sum_v yhat_c(v) + eps)

Edge-aware: organ boundaries sit where attenuation changes, i.e. on CT intensity edges. Prediction changes in flat
CT regions are penalised; on strong edges they are nearly free.

    L_edge = mean_v sum_{c>=1} |grad yhat_c(v)| * exp(-|grad I_s(v)| / kappa)

with I_s a Gaussian-smoothed CT, gradients in physical units (per mm, from each sample's voxel spacing) and kappa the
median CT gradient magnitude inside the body of that scan.

Both are regularisers to be weighted small next to Dice + CE: the HU term can still be lowered by predicting only
the best-fitting voxels, and the edge term also penalises real low-contrast boundaries (pancreas vs duodenum).
Inputs follow the training pipeline: image (B, 1, X, Y, Z) scaled from [-1000, 1000] HU to [0, 1], logits
(B, 13, X, Y, Z), prob (B, 13, ...), conf (B, 1, ...). Everything is computed in float32, also under AMP.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

HU_MIN, HU_MAX = -1000.0, 1000.0


def to_hu(image: torch.Tensor) -> torch.Tensor:
    """Inverse of ScaleIntensityRanged(a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True)."""
    return image.float() * (HU_MAX - HU_MIN) + HU_MIN


def spacing_from_affine(image) -> torch.Tensor:
    """(B, 3) voxel spacing in mm for the three spatial axes, from a MONAI MetaTensor's affine; ones without meta."""
    b = image.shape[0]
    affine = getattr(image, "affine", None)
    if affine is None:
        return torch.ones(b, 3, dtype=torch.float64)
    a = torch.as_tensor(affine, dtype=torch.float64).cpu()
    if a.ndim == 2:
        a = a.unsqueeze(0).expand(b, -1, -1)
    return torch.linalg.norm(a[:, :3, :3], dim=1)  # column norms, same convention as data/transforms._voxel_spacing


@torch.no_grad()
def organ_hu_stats(hu: torch.Tensor, prob: torch.Tensor, conf: torch.Tensor, core_p: float = 0.9,
                   core_w: float = 0.8, min_voxels: int = 50, sigma_min: float = 10.0):
    """
    hu (B, 1, ...), prob (B, C, ...), conf (B, 1, ...).
    Returns mu, sigma (B, C) in HU and valid (B, C) bool. Background (c = 0) is never valid.
    """
    b, c = prob.shape[:2]
    mu = torch.zeros(b, c, device=hu.device)
    sigma = torch.ones(b, c, device=hu.device)
    valid = torch.zeros(b, c, dtype=torch.bool, device=hu.device)
    confident = conf[:, 0] >= core_w
    for i in range(b):
        for k in range(1, c):
            core = (prob[i, k] >= core_p) & confident[i]
            n = int(core.sum())
            if n < min_voxels:
                continue
            vals = hu[i, 0][core]
            med = vals.median()
            mad = (vals - med).abs().median()
            mu[i, k] = med
            sigma[i, k] = torch.clamp(1.4826 * mad, min=sigma_min)
            valid[i, k] = True
    return mu, sigma, valid


def huber(z: torch.Tensor, delta: float) -> torch.Tensor:
    a = z.abs()
    return torch.where(a <= delta, 0.5 * z * z, delta * (a - 0.5 * delta))


def hu_consistency_loss(logits: torch.Tensor, image: torch.Tensor, prob: torch.Tensor, conf: torch.Tensor,
                        delta: float = 2.0, core_p: float = 0.9, core_w: float = 0.8, min_voxels: int = 50,
                        sigma_min: float = 10.0, eps: float = 1.0):
    """Returns (loss, number of (scan, organ) pairs that contributed). Zero, with a gradient path, if none did."""
    hu = to_hu(image)
    mu, sigma, valid = organ_hu_stats(hu, prob.float(), conf.float(), core_p, core_w, min_voxels, sigma_min)
    yhat = torch.softmax(logits.float(), dim=1)
    terms = []
    for i, k in valid.nonzero().tolist():
        z = (hu[i, 0] - mu[i, k]) / sigma[i, k]
        p = yhat[i, k]
        terms.append((p * huber(z, delta)).sum() / (p.sum() + eps))
    if not terms:
        return yhat.sum() * 0.0, 0
    return torch.stack(terms).mean(), len(terms)


def _gaussian_kernel1d(sigma_vox: float, device) -> torch.Tensor:
    radius = max(1, int(math.ceil(2.5 * sigma_vox)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma_vox) ** 2)
    return k / k.sum()


@torch.no_grad()
def gaussian_blur3d(x: torch.Tensor, sigma_vox: float = 1.0) -> torch.Tensor:
    """Separable Gaussian blur of (B, 1, X, Y, Z) with replicate padding; sigma in voxels."""
    if sigma_vox <= 0:
        return x
    k = _gaussian_kernel1d(sigma_vox, x.device)
    r = k.numel() // 2
    out = x.float()
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = k.numel()
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - axis)] = pad[2 * (2 - axis) + 1] = r  # F.pad orders the last dimension first
        out = F.conv3d(F.pad(out, pad, mode="replicate"), k.view(shape))
    return out


def edge_aware_loss(logits: torch.Tensor, image: torch.Tensor, spacing: torch.Tensor | None = None,
                    sigma_vox: float = 1.0, body_hu: float = -500.0, kappa_min: float = 1e-4) -> torch.Tensor:
    """spacing: (B, 3) mm per voxel (see spacing_from_affine); None means 1 mm."""
    b = logits.shape[0]
    if spacing is None:
        spacing = torch.ones(b, 3, dtype=torch.float64)
    yhat = torch.softmax(logits.float(), dim=1)[:, 1:]  # organs only
    smooth = gaussian_blur3d(image.float(), sigma_vox)
    hu = to_hu(image)
    losses = []
    for i in range(b):
        sp = [float(s) for s in spacing[i]]
        with torch.no_grad():
            gi = torch.gradient(smooth[i, 0], spacing=sp, dim=(0, 1, 2))
            mag_i = torch.sqrt(gi[0] ** 2 + gi[1] ** 2 + gi[2] ** 2)
            body = hu[i, 0] > body_hu
            ref = mag_i[body] if bool(body.any()) else mag_i.flatten()
            kappa = torch.clamp(ref.median(), min=kappa_min)
            weight = torch.exp(-mag_i / kappa)
        gp = torch.gradient(yhat[i], spacing=sp, dim=(1, 2, 3))
        mag_p = torch.sqrt(gp[0] ** 2 + gp[1] ** 2 + gp[2] ** 2 + 1e-12)  # finite gradient at 0, negligible floor
        losses.append((mag_p.sum(dim=0) * weight).mean())
    return torch.stack(losses).mean()
