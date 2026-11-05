"""
nnU-Net v2 with the atlas prior: nnU-Net's own planning, preprocessing, network, training and inference, plus the
atlas-prior paths of models/swin_atlas.py that make sense for a CNN (the anatomical cross-attention tokens are left out).

Input (written by scripts/nnunet_make_datasets.py), one file per channel:
  channel 0          CT, nnU-Net's "CT" normalisation
  channels 1..K      P_k, atlas probability of organ k                       } uint8 0..250 = value x 250,
  channels K+1..2K   N_k = 1 - D_k, nearness to organ k (D clipped at 50 mm) } "noNorm" (stored as is)
  last channel       w, atlas confidence (1 - normalised entropy)            }
  ("pw" layout: P_1..P_K and w only.)
  Zero means "no organ, far from every organ, no confidence", so the zero padding of nnU-Net's crops and spatial
  augmentation is a neutral prior. P_background = 1 - sum_k P_k.

Network: the plain nnU-Net network sees [CT, priors / 250]; every output (all deep-supervision scales) gets
  logits_c + gamma_c * w * log(P_c + 1e-3),   gamma init 1, so the untrained network starts at the atlas vote.

Training: nnU-Net's intensity augmentations (noise, blur, brightness, contrast, low resolution, gamma) act on the CT
only; the prior channels are set aside before them and put back after (spatial transforms and mirroring still move CT,
label and priors together). Prior dropout: with probability 0.3 a sample gets "no atlas information"
(P_k = 1/C, N_k = 0, w = 0), so the network keeps its own image pathway.

Image-only baseline: the stock nnUNetTrainer (same number of epochs) on the same cases, CT channel only, same plans.

This file must sit inside the installed nnunetv2 package (nnU-Net finds trainers by name there, also at inference):
    python nnunet_ext/install.py
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

PRIOR_SCALE = 250.0
LOG_PRIOR_EPS = 1e-3
INTENSITY_TRANSFORMS = (GaussianNoiseTransform, GaussianBlurTransform, MultiplicativeBrightnessTransform,
                        ContrastTransform, SimulateLowResolutionTransform, GammaTransform)


def prior_layout(num_input_channels: int, num_classes: int) -> str:
    k, n = num_classes - 1, num_input_channels - 1
    if n == 2 * k + 1:
        return "pnw"
    if n == k + 1:
        return "pw"
    raise ValueError(f"{num_input_channels} input channels fit neither CT + P,N,w ({2 * k + 2}) nor CT + P,w ({k + 2}) "
                     f"for {num_classes} classes")


PAUSE_EXIT_CODE = 75


def pause_file() -> str:
    """<nnunet root>/PAUSE: while it exists, a run stops after the epoch it is in (scripts/amos_pause.py)."""
    return os.path.join(os.path.dirname(os.environ.get("nnUNet_results", "")), "PAUSE")


class ResumableMixin:
    """checkpoint_latest.pth after every epoch (stock nnU-Net: every 50), and a clean stop when the PAUSE file exists.
    Resume = rerun with --c: weights, optimizer, scaler, LR schedule position and the logs continue from that epoch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_every = 1

    def on_epoch_end(self):
        super().on_epoch_end()  # saves checkpoint_latest.pth and advances current_epoch
        if os.path.exists(pause_file()):
            self.print_to_log_file(f"PAUSE requested: stopped cleanly; resume continues at epoch {self.current_epoch}")
            os._exit(PAUSE_EXIT_CODE)  # no interpreter shutdown: it crashes with the data loader workers alive


class AtlasPriorFusion(nn.Module):
    """nnU-Net network + log-prior output fusion. Exposes `.decoder` for nnU-Net's deep-supervision switch."""

    def __init__(self, net: nn.Module, num_input_channels: int, num_classes: int, gamma_init: float = 1.0):
        super().__init__()
        self.net = net
        self.layout = prior_layout(num_input_channels, num_classes)
        self.num_organs = num_classes - 1
        self.gamma = nn.Parameter(torch.full((num_classes,), float(gamma_init)))

    @property
    def decoder(self):
        return self.net.decoder

    def log_prior(self, prior: torch.Tensor) -> torch.Tensor:
        """w * log(P + eps) for all classes, from priors scaled to [0, 1]."""
        p_org = prior[:, :self.num_organs]
        p_bg = (1.0 - p_org.sum(1, keepdim=True)).clamp(0.0, 1.0)
        return prior[:, -1:] * torch.log(torch.cat([p_bg, p_org], 1) + LOG_PRIOR_EPS)

    def forward(self, x: torch.Tensor):
        prior = (x[:, 1:].float() / PRIOR_SCALE).clamp(0.0, 1.0)  # resampling can overshoot slightly
        out = self.net(torch.cat([x[:, :1], prior.to(x.dtype)], 1))
        with torch.no_grad():
            lp = self.log_prior(prior)
            if torch.is_autocast_enabled():  # the 4 GB GPU has no room for a float32 copy of the log prior
                lp = lp.half()
        del prior

        def fuse(o):
            lpo = lp if o.shape[2:] == lp.shape[2:] else F.interpolate(lp, size=o.shape[2:], mode="area")
            g = self.gamma.to(lpo.dtype).view(1, -1, *([1] * (o.ndim - 2)))
            return o + (g * lpo).to(o.dtype)

        return [fuse(o) for o in out] if isinstance(out, (list, tuple)) else fuse(out)


class SetAsidePriors(BasicTransform):
    """image (1 + n, ...) -> image (1, ...); the prior channels wait in data_dict['_atlas_priors']."""

    def apply(self, data_dict, **params):
        img = data_dict["image"]
        data_dict["_atlas_priors"], data_dict["image"] = img[1:], img[:1]
        return data_dict


class PutBackPriors(BasicTransform):
    def apply(self, data_dict, **params):
        pr = data_dict.pop("_atlas_priors")
        data_dict["image"] = torch.cat([data_dict["image"], pr.to(data_dict["image"].dtype)], 0)
        return data_dict


class PriorDropout(BasicTransform):
    """With probability p, replace the priors by 'no atlas information': P_k = 1/C, N_k = 0, w = 0 (raw uint8 units)."""

    def __init__(self, p: float, num_classes: int):
        super().__init__()
        self.p, self.num_classes = p, num_classes

    def get_parameters(self, **data_dict) -> dict:
        return {"drop": torch.rand(1).item() < self.p}

    def apply(self, data_dict, **params):
        if params["drop"]:
            img = data_dict["image"].clone()
            k = self.num_classes - 1
            img[1:1 + k] = PRIOR_SCALE / self.num_classes
            img[1 + k:] = 0.0
            data_dict["image"] = img
        return data_dict


def _is_intensity(t) -> bool:
    inner = t.transform if isinstance(t, RandomTransform) else t
    return isinstance(inner, INTENSITY_TRANSFORMS)


def protect_priors(transforms: ComposeTransforms, num_classes: int, prior_dropout: float) -> ComposeTransforms:
    """Wrap nnU-Net's intensity augmentations so they see only the CT; add prior dropout after them."""
    ts = list(transforms.transforms)
    idx = [i for i, t in enumerate(ts) if _is_intensity(t)]
    if not idx:
        raise RuntimeError("no intensity augmentation found in nnU-Net's training transforms; check the nnU-Net version")
    first, last = idx[0], idx[-1]
    if any(not _is_intensity(t) for t in ts[first:last + 1]):
        raise RuntimeError("nnU-Net's intensity augmentations are no longer contiguous; revisit protect_priors")
    new = ts[:first] + [SetAsidePriors()] + ts[first:last + 1] + [PutBackPriors()]
    if prior_dropout > 0:
        new.append(PriorDropout(prior_dropout, num_classes))
    return ComposeTransforms(new + ts[last + 1:])


class nnUNetTrainerAtlasPrior(ResumableMixin, nnUNetTrainer):
    PRIOR_DROPOUT = 0.3
    GAMMA_INIT = 1.0

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels, num_output_channels,
                                   enable_deep_supervision=True):
        # also called by nnU-Net's predictor, so inference gets the same fusion
        net = nnUNetTrainer.build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                                       num_output_channels, enable_deep_supervision)
        return AtlasPriorFusion(net, num_input_channels, num_output_channels, nnUNetTrainerAtlasPrior.GAMMA_INIT)

    def get_training_transforms(self, *args, **kwargs):  # an instance method here: it needs the class count
        base = nnUNetTrainer.get_training_transforms(*args, **kwargs)
        return protect_priors(base, self.label_manager.num_segmentation_heads, self.PRIOR_DROPOUT)


class nnUNetTrainerAtlasPrior_50epochs(nnUNetTrainerAtlasPrior):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainerAtlasPrior_100epochs(nnUNetTrainerAtlasPrior):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerAtlasPrior_250epochs(nnUNetTrainerAtlasPrior):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerResumable(ResumableMixin, nnUNetTrainer):
    """The stock trainer (image-only baseline) with per-epoch checkpoints and the pause switch; training is unchanged."""


class nnUNetTrainerResumable_100epochs(nnUNetTrainerResumable):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100


class nnUNetTrainerResumable_50epochs(nnUNetTrainerResumable):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50


class nnUNetTrainerResumable_250epochs(nnUNetTrainerResumable):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250


class nnUNetTrainerResumable_smoke(nnUNetTrainerResumable):
    """4 epochs of 10 iterations: tests pause and resume, nothing more."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs, self.num_iterations_per_epoch, self.num_val_iterations_per_epoch = 4, 10, 5


class nnUNetTrainerAtlasPrior_smoke(nnUNetTrainerAtlasPrior):
    """1 epoch of 10 iterations: checks the whole train / validate / predict path, nothing more."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs, self.num_iterations_per_epoch, self.num_val_iterations_per_epoch = 1, 10, 5


class nnUNetTrainer_smoke(nnUNetTrainer):
    """The stock trainer (image-only baseline) for 1 epoch of 10 iterations: a pipeline check."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs, self.num_iterations_per_epoch, self.num_val_iterations_per_epoch = 1, 10, 5
