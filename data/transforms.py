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
        stack = d.pop(self.keys[0])
        P, dist, conf = atlas_prior_maps(stack, self.num_classes, self.dist_clip_mm, self.relative_threshold)
        d[self.prob_key] = torch.from_numpy(P)
        d[self.dist_key] = torch.from_numpy(dist)
        d[self.conf_key] = torch.from_numpy(conf)
        return d


def atlas_prior_maps(stack, num_classes: int, dist_clip_mm: float = 50.0, relative_threshold: float = 0.5,
                     spacing=None):
    """
    P (C, ...), D (C-1, ...), w (1, ...) as float32 arrays from a warped-atlas label stack (N, X, Y, Z); see AtlasPriorsd.
    `spacing` (mm per spatial axis, in array order) defaults to the stack's affine.
    """
    spacing = tuple(spacing) if spacing is not None else _voxel_spacing(stack)
    lab = np.rint(np.asarray(stack.detach().cpu() if torch.is_tensor(stack) else stack)).astype(np.int64)

    C = num_classes
    P = np.stack([(lab == c).mean(axis=0) for c in range(C)]).astype(np.float32)
    H = -(P * np.log(np.clip(P, 1e-8, None))).sum(axis=0)
    conf = (1.0 - H / np.log(C)).astype(np.float32)

    dist = np.ones((C - 1,) + P.shape[1:], dtype=np.float32)
    for c in range(1, C):
        peak = P[c].max()
        if peak <= 0:
            continue
        region = P[c] >= relative_threshold * peak
        dt = distance_transform_edt(~region, sampling=spacing)
        dist[c - 1] = np.minimum(dt, dist_clip_mm) / dist_clip_mm
    return P, dist, conf[None]


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


# ---------------------------------------------------------------------------------------------------------------------
# Patch pipeline: training on fixed-spacing patches, evaluation on the whole volume by sliding window.
#
# build_transforms above resizes every cropped volume to 96^3, so a voxel is 3-5 mm and an adrenal gland is 1-3 voxels
# thick. Here the volume keeps a fixed spacing (1.5 mm by default) and the network sees 96^3 patches of it, as in UNETR
# and Swin UNETR. The atlas priors are built ONCE per case on the warped-atlas grid (2 mm, the registration grid), over
# the whole volume, so D_c is the distance to the organ even when the organ lies outside the patch; each patch then
# samples them onto its own grid.
# ---------------------------------------------------------------------------------------------------------------------

PRIOR_QUANT = 250.0  # priors are stored as uint8 round(x * 250): exact for P with 10 atlases, 0.2 mm steps for D


class AtlasPriorVolumed(mt.MapTransform):
    """
    The warped-atlas stack (N, X, Y, Z) -> one MetaTensor `out_key` of 2C channels [P (C), D (C-1), w (1)], on the
    stack's own grid and with its affine, so a later ResampleToMatchd can sample it onto any image grid or patch.
    """

    def __init__(self, keys=("atlas_labels",), out_key: str = "priors", num_classes: int = 13, dist_clip_mm: float = 50.0,
                 relative_threshold: float = 0.5, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.out_key, self.num_classes = out_key, num_classes
        self.dist_clip_mm, self.relative_threshold = dist_clip_mm, relative_threshold

    def __call__(self, data):
        from monai.data import MetaTensor

        d = dict(data)
        stack = d.pop(self.keys[0])
        P, dist, conf = atlas_prior_maps(stack, self.num_classes, self.dist_clip_mm, self.relative_threshold)
        meta = dict(stack.meta) if isinstance(stack, MetaTensor) else {}  # includes the stack's affine
        d[self.out_key] = MetaTensor(torch.from_numpy(np.concatenate([P, dist, conf])), meta=meta)
        return d


class QuantizePriorsd(mt.MapTransform):
    """[0, 1] priors <-> uint8 round(x * PRIOR_QUANT), for the on-disk cache (inverse=True decodes to float32)."""

    def __init__(self, keys=("priors",), inverse: bool = False, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.inverse = inverse

    def __call__(self, data):
        d = dict(data)
        for k in self.key_iterator(d):
            d[k] = d[k].float() / PRIOR_QUANT if self.inverse else (d[k] * PRIOR_QUANT).round().clamp(0, 255).to(torch.uint8)
        return d


class ComplementPriorsd(mt.MapTransform):
    """
    x -> 1 - x on the channels P_background, D and w of `priors` (self-inverse). Wrapped around a rotation with zero
    padding, the rotated-in corners then read P_bg = 1, P_organ = 0, D = 1, w = 1: "the atlases say background here",
    which is what build_transforms gives (its zero-padded atlas labels are background) and matches the label (also
    background there). Interpolation is linear, so the complement commutes with it everywhere else.
    """

    def __init__(self, keys=("priors",), num_classes: int = 13, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.channels = [0] + list(range(num_classes, 2 * num_classes))

    def __call__(self, data):
        d = dict(data)
        for k in self.key_iterator(d):
            x = d[k].clone()
            x[self.channels] = 1.0 - x[self.channels]
            d[k] = x
        return d


class SplitPriorsd(mt.MapTransform):
    """`priors` (2C channels) -> plain tensors prob (C), dist (C-1), conf (1) in `dtype`."""

    def __init__(self, keys=("priors",), num_classes: int = 13, dtype=torch.float32, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.num_classes, self.dtype = num_classes, dtype

    def __call__(self, data):
        d = dict(data)
        pr = d.pop(self.keys[0])
        # every prior lies in [0, 1]; interpolation and the complement leave ~1e-7 rounding outside it
        pr = (pr.as_tensor() if hasattr(pr, "as_tensor") else torch.as_tensor(pr)).clamp(0, 1).to(self.dtype)
        C = self.num_classes
        d["prob"], d["dist"], d["conf"] = (pr[:C].contiguous(), pr[C:2 * C - 1].contiguous(), pr[2 * C - 1:].contiguous())
        return d


def cache_boundary(d):
    # A plain callable (not a monai Transform): PersistentDataset caches everything before it and nothing after.
    return d


def build_patch_transforms(
    patch_size,
    spacing,
    num_classes: int,
    train: bool,
    rotate_range=(-0.26, 0.26),
    dist_clip_mm: float = 50.0,
    max_samples_per_class: int = 10000,
):
    """
    Cached part (identical for training and evaluation, so a case's cache file serves both):
      load; image and label to RAS, crop to the body (HU > 0, as build_transforms), resample to `spacing` mm, scale
      HU [-1000, 1000] -> [0, 1], pad to at least one patch; priors P, D, w from the warped-atlas stack on its own grid;
      per-class voxel indices for the class-balanced crop (at most `max_samples_per_class` per class). Stored compactly:
      image float16 (~1 HU steps), label uint8, priors uint8 (PRIOR_QUANT).

    Training: one patch per sample, centred on a voxel of a class drawn uniformly from the classes present (background
    included), so small organs are seen as often as the liver; priors sampled onto the patch grid (trilinear, border
    padding: outside the atlas grid the nearest atlas value is used); flips and rotations applied to image, label and
    priors together, with the corners a rotation brings in read as air / background / "atlas: background"
    (ComplementPriorsd). Output: image (1, *patch), label one-hot (C, *patch), prob, dist, conf as in build_transforms.

    Evaluation: the whole resampled volume; priors sampled onto it; label stays an index map (1, ...) uint8 and image and
    priors stay float16 (a 1.5 mm volume is ~13 M voxels; one-hot labels and float32 priors would need several GB).
    """
    spatial = ["image", "label"]
    patch_size = tuple(int(s) for s in patch_size)
    spacing = tuple(float(s) for s in spacing)
    cached = [
        mt.LoadImaged(keys=spatial + ["atlas_labels"]),
        mt.EnsureChannelFirstd(keys=spatial, channel_dim="no_channel"),
        mt.EnsureChannelFirstd(keys=["atlas_labels"], channel_dim=-1),
        mt.Orientationd(keys=spatial, axcodes="RAS"),
        mt.CropForegroundd(keys=spatial, source_key="image"),
        mt.Spacingd(keys=spatial, pixdim=spacing, mode=("bilinear", "nearest")),
        mt.ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
        mt.SpatialPadd(keys=spatial, spatial_size=patch_size),
        AtlasPriorVolumed(keys=["atlas_labels"], num_classes=num_classes, dist_clip_mm=dist_clip_mm),
        mt.ClassesToIndicesd(keys=["label"], num_classes=num_classes, max_samples_per_class=max_samples_per_class),
        mt.CastToTyped(keys=spatial, dtype=(torch.float16, torch.uint8)),
        QuantizePriorsd(keys=["priors"]),
        cache_boundary,
        QuantizePriorsd(keys=["priors"], inverse=True),
    ]
    to_grid = mt.ResampleToMatchd(keys=["priors"], key_dst="image", mode="bilinear", padding_mode="border",
                                  dtype=torch.float32)
    if train:
        keys = spatial + ["priors"]
        rest = [
            mt.CastToTyped(keys=["image"], dtype=torch.float32),
            mt.RandCropByLabelClassesd(keys=spatial, label_key="label", spatial_size=patch_size, num_classes=num_classes,
                                       num_samples=1, indices_key="label_cls_indices", warn=False),
            mt.DeleteItemsd(keys=["label_cls_indices"]),
            to_grid,
            mt.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            mt.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            mt.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            ComplementPriorsd(keys=["priors"], num_classes=num_classes),
            mt.RandRotated(keys=keys, range_x=rotate_range, range_y=rotate_range, range_z=rotate_range, prob=0.5,
                           mode=("bilinear", "nearest", "bilinear"), padding_mode="zeros"),
            ComplementPriorsd(keys=["priors"], num_classes=num_classes),
            mt.CastToTyped(keys=["label"], dtype=torch.long),
            mt.AsDiscreted(keys=["label"], to_onehot=num_classes),
            SplitPriorsd(keys=["priors"], num_classes=num_classes),
            mt.ToTensord(keys=["image", "label", "prob", "dist", "conf"]),
        ]
    else:
        rest = [
            mt.DeleteItemsd(keys=["label_cls_indices"]),
            to_grid,
            SplitPriorsd(keys=["priors"], num_classes=num_classes, dtype=torch.float16),
            mt.ToTensord(keys=["image", "label", "prob", "dist", "conf"]),
        ]
    return mt.Compose(cached + rest)
