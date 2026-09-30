"""
Shared MONAI transform pieces, pulled out of the notebook so they have one
definition instead of being copy-pasted between the notebook and scripts/.
"""
from __future__ import annotations

import monai.transforms as mt
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


class CustomSpatialDistanceTransformd(mt.MapTransform):
    """
    Per-organ Euclidean distance map, D_c(v) = min_{y in O_c} ||v - y||.

    BUGFIX (see research_audit.md, Section G): the original implementation had
    `to_closest_foreground=True` calling `distance_transform_edt(binary_mask)`.
    scipy's `distance_transform_edt` measures, for every NONZERO element, the
    distance to the nearest ZERO element -- i.e. it gave the distance from
    inside the organ to its own boundary, and 0 everywhere in the background.
    That is the opposite of what the research report specifies: D_c(v) should
    be 0 *inside* the organ and grow with distance *outside* it. The correct
    call is `distance_transform_edt(~binary_mask)`.

    `legacy_buggy_distance=True` reproduces the ORIGINAL (incorrect) behaviour
    bit-for-bit. It exists only so `scripts/evaluate_checkpoint.py` can
    reproduce the epoch-101 validation Dice (~0.6505) against `best_model.pth`,
    which was trained on the buggy distance maps. Do not use it for new
    training runs -- new runs should leave this at the default (False).
    """

    def __init__(
        self,
        keys,
        to_closest_foreground: bool = True,
        include_background: bool = False,
        output_postfix: str = "_distance",
        allow_missing_keys: bool = False,
        legacy_buggy_distance: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.to_closest_foreground = to_closest_foreground
        self.include_background = include_background
        self.output_postfix = output_postfix
        self.legacy_buggy_distance = legacy_buggy_distance

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label_one_hot = d[key]  # torch.Tensor, (C, D, H, W)
            if not isinstance(label_one_hot, torch.Tensor):
                raise TypeError(f"input label must be torch.Tensor, got {type(label_one_hot)}")
            if label_one_hot.ndim < 4:
                raise ValueError(f"input label must have at least 4 dimensions (C, D, H, W), got {label_one_hot.ndim}")

            label_np = label_one_hot.cpu().numpy()
            num_channels = label_np.shape[0]
            spatial_dims = label_np.shape[1:]

            distance_maps_list = []
            start_channel_idx = 0 if self.include_background else 1

            for c in range(start_channel_idx, num_channels):
                binary_mask = label_np[c].astype(bool)

                if self.legacy_buggy_distance:
                    dt = distance_transform_edt(binary_mask) if self.to_closest_foreground else distance_transform_edt(~binary_mask)
                elif self.to_closest_foreground:
                    # Correct: distance from every voxel to the nearest foreground (organ) voxel.
                    dt = distance_transform_edt(~binary_mask)
                else:
                    # Distance from every voxel to the nearest background voxel.
                    dt = distance_transform_edt(binary_mask)

                distance_maps_list.append(dt)

            if not distance_maps_list and not self.include_background:
                d[key + self.output_postfix] = torch.zeros((0,) + spatial_dims, dtype=torch.float32)
            elif not distance_maps_list and self.include_background:
                d[key + self.output_postfix] = torch.zeros((num_channels,) + spatial_dims, dtype=torch.float32)
            else:
                d[key + self.output_postfix] = torch.from_numpy(np.stack(distance_maps_list, axis=0)).float()

        return d


def _voxel_spacing(t) -> tuple:
    """Voxel spacing (mm) from a MetaTensor's affine; 1.0 if there is no meta."""
    affine = getattr(t, "affine", None)
    if affine is None:
        return (1.0,) * (t.ndim - 1)
    A = np.asarray(affine, dtype=np.float64)[:3, :3]
    return tuple(float(s) for s in np.linalg.norm(A, axis=0))


class AtlasPriorsd(mt.MapTransform):
    """
    Builds the atlas-derived anatomical priors from a stack of warped atlas
    labels (N_atlas, X, Y, Z) that is already on the same grid as the image:

        P_c(v) = (1/N) sum_i 1[L'_i(v) = c]                 -> "prob"  (C, X, Y, Z)
        w(v)   = 1 - H(v) / log C,  H = -sum_c P_c log P_c  -> "conf"  (1, X, Y, Z)
        D_c(v) = min_{y in O_c} ||v - y||, in mm            -> "dist"  (C-1, X, Y, Z)

    Nothing here reads the target's own label.

    O_c (the atlas-derived region of organ c) is {v : P_c(v) >= r * max P_c},
    with r = relative_threshold (default 0.5). For a well-registered organ
    (max P_c = 1) this is the majority region. For a small, poorly agreeing
    organ (e.g. max P_c = 0.3) a fixed 0.5 cut would leave O_c empty and D_c
    undefined; the relative cut keeps "where the atlases agree most" instead.
    An organ that no atlas places in view at all gets D_c = 1 everywhere.

    D is clipped at dist_clip_mm and divided by it, so it lies in [0, 1] like
    P. Raw mm distances (0-300) would otherwise dominate P inside the
    AnatomicalPriorGenerator MLP. Distances use the true voxel spacing (from
    the affine), because the 96^3 grid is anisotropic after resizing.

    Must run AFTER all spatial augmentation, so the priors match the augmented CT.
    """

    def __init__(
        self,
        keys=("atlas_labels",),
        num_classes: int = 13,
        dist_clip_mm: float = 50.0,
        relative_threshold: float = 0.5,
        prob_key: str = "prob",
        dist_key: str = "dist",
        conf_key: str = "conf",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.num_classes = num_classes
        self.dist_clip_mm = dist_clip_mm
        self.relative_threshold = relative_threshold
        self.prob_key, self.dist_key, self.conf_key = prob_key, dist_key, conf_key

    def __call__(self, data):
        d = dict(data)
        key = self.keys[0]
        stack = d.pop(key)
        spacing = _voxel_spacing(stack)
        lab = np.rint(np.asarray(stack.detach().cpu() if torch.is_tensor(stack) else stack)).astype(np.int64)

        C = self.num_classes
        P = np.stack([(lab == c).mean(axis=0) for c in range(C)]).astype(np.float32)
        H = -(P * np.log(np.clip(P, 1e-8, None))).sum(axis=0)
        conf = (1.0 - H / np.log(C)).astype(np.float32)

        dist = np.ones((C - 1,) + P.shape[1:], dtype=np.float32)
        for c in range(1, C):
            peak = P[c].max()
            if peak <= 0:
                continue
            region = P[c] >= self.relative_threshold * peak
            dt = distance_transform_edt(~region, sampling=spacing)
            dist[c - 1] = np.minimum(dt, self.dist_clip_mm) / self.dist_clip_mm

        d[self.prob_key] = torch.from_numpy(P)
        d[self.dist_key] = torch.from_numpy(dist)
        d[self.conf_key] = torch.from_numpy(conf[None])
        return d


PRIOR_KEYS = {
    # prior_source -> (probability-map key, distance-map key) fed to the model
    "gt": ("label", "label_distance"),  # original prototype: priors from the case's OWN label (leaks GT)
    "atlas": ("prob", "dist"),  # atlas-derived priors, no target GT
}


def build_transforms(
    img_size,
    num_classes: int,
    train: bool,
    legacy_buggy_distance: bool = False,
    prior_source: str = "gt",
    rotate_range=(-15, 15),
    dist_clip_mm: float = 50.0,
):
    """
    prior_source="gt" reconstructs the exact train/val Compose from the
    notebook (cell 14): priors come from the case's own ground truth.

    prior_source="atlas" loads the cached warped-atlas stack (key
    "atlas_labels", see data/splits.attach_atlas_priors) and resamples it
    directly onto the final 96^3 image grid with ResampleToMatchd. MONAI
    tracks the image affine through orientation, crop, spacing and resize,
    so no intermediate full-resolution copy of the stack is needed. It then
    goes through the same random flips/rotations as the image, and
    AtlasPriorsd builds P, D, w at the very end. The label is still loaded,
    but only as the supervised-loss target, never as a prior.

    rotate_range: RandRotated ranges are in RADIANS in MONAI. The notebook's
    (-15, 15) is therefore +-15 rad, i.e. arbitrary rotations, not +-15
    degrees. It is kept as the default so experiments stay comparable with
    the epoch-101 baseline. np.deg2rad(15) ~= 0.26 is the presumably intended
    value, to be tested as a separate ablation.
    """
    if prior_source not in PRIOR_KEYS:
        raise ValueError(f"unknown prior_source {prior_source!r}")
    atlas = prior_source == "atlas"
    spatial = ["image", "label"] + (["atlas_labels"] if atlas else [])
    interp = ("bilinear", "nearest") + (("nearest",) if atlas else ())

    common_pre = [
        mt.LoadImaged(keys=spatial),
        mt.EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
    ]
    if atlas:
        common_pre.append(mt.EnsureChannelFirstd(keys=["atlas_labels"], channel_dim=-1))
    common_pre += [
        mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
        mt.CropForegroundd(keys=["image", "label"], source_key="image"),
        mt.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        mt.ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        mt.Resized(keys=["image", "label"], spatial_size=img_size, mode=("area", "nearest")),
    ]
    if atlas:
        common_pre.append(mt.ResampleToMatchd(keys=["atlas_labels"], key_dst="image", mode="nearest", padding_mode="zeros"))

    if train:
        aug = [
            mt.RandSpatialCropd(keys=spatial, roi_size=img_size, random_size=False, random_center=True),
            mt.RandFlipd(keys=spatial, prob=0.5, spatial_axis=0),
            mt.RandFlipd(keys=spatial, prob=0.5, spatial_axis=1),
            mt.RandFlipd(keys=spatial, prob=0.5, spatial_axis=2),
            mt.RandRotated(keys=spatial, range_x=rotate_range, range_y=rotate_range, range_z=rotate_range, prob=0.5, mode=interp),
        ]
    else:
        aug = []

    post = [
        mt.CastToTyped(keys=["image"], dtype=torch.float32),
        mt.CastToTyped(keys=["label"], dtype=torch.long),
        mt.AsDiscreted(keys=["label"], to_onehot=num_classes),
    ]
    if atlas:
        post += [
            AtlasPriorsd(keys=["atlas_labels"], num_classes=num_classes, dist_clip_mm=dist_clip_mm),
            mt.ToTensord(keys=["image", "label", "prob", "dist", "conf"]),
        ]
    else:
        post += [
            CustomSpatialDistanceTransformd(
                keys=["label"],
                to_closest_foreground=True,
                include_background=False,
                output_postfix="_distance",
                legacy_buggy_distance=legacy_buggy_distance,
            ),
            mt.ToTensord(keys=["image", "label", "label_distance"]),
        ]

    return mt.Compose(common_pre + aug + post)
