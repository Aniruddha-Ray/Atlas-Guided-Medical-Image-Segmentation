"""
Atlas-to-target registration, label warping, and multi-atlas anatomical priors.

This is the real (non-ground-truth) P(v)/D(v) pipeline described in the
research report, Sections 1.4-1.7. It was drafted directly in the notebook
(AtlasGuidedUNETR_fixed.ipynb) but was never made importable, never had its
SimpleITK dependency declared, and is not called anywhere -- training still
uses `mt.AsDiscreted`/`CustomSpatialDistanceTransformd` on ground-truth labels
(see research_audit.md, Sections C and G). Moving it here removes those three
blockers so it can actually be imported and unit-tested; the logic itself is
unchanged from the notebook draft.

NOT YET WIRED INTO TRAINING. `AtlasGuidedViTUNETR` is still trained with
GT-derived priors. Swapping the training/validation dataloaders over to this
pipeline is the next milestone (Experiment 1 in the research plan) and should
be done deliberately, with caching (Section 30 of the brief), since
`register_atlas_to_target` runs two full image-registration optimizations per
atlas per target volume and is far too slow to call inside a training loop.
"""
from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import distance_transform_edt

try:
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover - exercised only when SimpleITK is missing
    raise ImportError(
        "SimpleITK is required for atlas.atlas_pipeline (atlas-to-target "
        "registration). Install it with `pip install -r requirements.txt`."
    ) from exc


def register_atlas_to_target(atlas_img_np: np.ndarray, target_img_np: np.ndarray) -> "sitk.Transform":
    """
    Atlas-to-target registration: coarse affine (RANSAC-affine equivalent)
    followed by B-spline non-rigid refinement (TPS/NiftyReg equivalent).
    Returns a single composite transform T(x) = phi(A(x)).

    Both inputs are expected as numpy arrays in (D, H, W) order, already
    preprocessed the same way the training pipeline preprocesses CT
    (orientation, spacing, intensity scaling) so atlas and target are in a
    comparable intensity range.

    NOTE: this substitutes intensity-based (Mattes mutual information)
    registration for the report's feature-correspondence + RANSAC pipeline
    (golden transformation, reliable feature scoring, Top-K selection --
    research_audit.md gap table, rows 1-3). That machinery does not exist yet
    anywhere in this project.
    """
    atlas_img = sitk.Cast(sitk.GetImageFromArray(atlas_img_np), sitk.sitkFloat32)
    target_img = sitk.Cast(sitk.GetImageFromArray(target_img_np), sitk.sitkFloat32)

    # --- Stage 1: affine, coarse alignment ---
    affine_reg = sitk.ImageRegistrationMethod()
    affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    affine_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200
    )
    affine_reg.SetInterpolator(sitk.sitkLinear)
    init_tx = sitk.CenteredTransformInitializer(target_img, atlas_img, sitk.AffineTransform(3))
    affine_reg.SetInitialTransform(init_tx, inPlace=False)
    affine_tx = affine_reg.Execute(target_img, atlas_img)

    # resample atlas into target space with the affine result before refining
    atlas_affine_aligned = sitk.Resample(
        atlas_img, target_img, affine_tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )

    # --- Stage 2: B-spline non-rigid refinement ---
    mesh_size = [4] * 3  # increase for finer deformation, at higher registration cost
    bspline_tx = sitk.BSplineTransformInitializer(target_img, mesh_size)
    bspline_reg = sitk.ImageRegistrationMethod()
    bspline_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    bspline_reg.SetOptimizerAsLBFGSB(numberOfIterations=100)
    bspline_reg.SetInterpolator(sitk.sitkLinear)
    bspline_reg.SetInitialTransform(bspline_tx, inPlace=False)
    bspline_final = bspline_reg.Execute(target_img, atlas_affine_aligned)

    # full transform = bspline composed with affine
    return sitk.CompositeTransform([bspline_final, affine_tx])


def warp_label(atlas_label_np: np.ndarray, transform: "sitk.Transform", ref_img_np: np.ndarray) -> np.ndarray:
    """Warp an atlas's integer label map into the target's coordinate space (nearest-neighbor)."""
    atlas_label = sitk.GetImageFromArray(atlas_label_np.astype(np.uint8))
    ref_img = sitk.GetImageFromArray(ref_img_np.astype(np.float32))
    warped = sitk.Resample(atlas_label, ref_img, transform, sitk.sitkNearestNeighbor, 0, atlas_label.GetPixelID())
    return sitk.GetArrayFromImage(warped)


def build_atlas_priors(target_img_np: np.ndarray, atlas_list, num_classes: int):
    """
    atlas_list: list of (atlas_img_np, atlas_label_np) pairs.

    Returns:
        P:          (C, D, H, W) float32 -- multi-atlas voted probability map
        D_map:      (C, D, H, W) float32 -- per-organ Euclidean distance map,
                    computed from the voted atlas region, NOT from ground truth
        confidence: (D, H, W)    float32 -- entropy-based atlas agreement, in [0, 1]
    """
    warped_labels = []
    for atlas_img_np, atlas_label_np in atlas_list:
        tx = register_atlas_to_target(atlas_img_np, target_img_np)
        warped_labels.append(warp_label(atlas_label_np, tx, target_img_np))

    stacked = np.stack(warped_labels, axis=0)  # (N_atlas, D, H, W)
    onehots = np.stack(
        [(stacked == c).astype(np.float32) for c in range(num_classes)], axis=1
    )  # (N_atlas, C, D, H, W)
    P = onehots.mean(axis=0)  # (C, D, H, W) -- Section 11 (probability map)

    # Section 13: per-organ distance transform, computed from the voted region (argmax), not GT.
    # distance_transform_edt(voted_label != c) is 0 inside organ c and grows
    # with distance outside it -- this is the correct direction (see the
    # bugfix note in CustomSpatialDistanceTransformd in the notebook).
    voted_label = np.argmax(P, axis=0)
    D_map = np.stack(
        [distance_transform_edt(voted_label != c).astype(np.float32) for c in range(num_classes)],
        axis=0,
    )

    # Section 12: entropy-based confidence w(v)
    eps = 1e-8
    entropy = -(P * np.log(P + eps)).sum(axis=0)
    confidence = 1.0 - (entropy / np.log(num_classes))

    return P.astype(np.float32), D_map, confidence.astype(np.float32)


def select_atlas_library(all_cases, atlas_count: int, seed: int = 42):
    """
    Deterministically splits a list of {"image": ..., "label": ...} case dicts
    (the same shape the notebook builds in its data-preparation cell) into an
    atlas library and a remaining target pool.

    This exists because there is currently no other definition anywhere in the
    project of which labeled cases act as "atlases" (Ai = (Ii, Li), used only
    to build P(v)/D(v)) versus "targets" (evaluated normally, priors built by
    registering atlases onto them). `Dataset060_Merged_Def` is 100% labeled,
    so this split is a configuration choice, not something inferable from the
    data. Given the same `all_cases`, `atlas_count`, and `seed`, this always
    returns the same split.
    """
    if atlas_count <= 0:
        raise ValueError("atlas_count must be a positive integer")
    if atlas_count >= len(all_cases):
        raise ValueError(
            f"atlas_count ({atlas_count}) must be smaller than the number of "
            f"available cases ({len(all_cases)}) so at least one target remains"
        )
    rng = random.Random(seed)
    shuffled = list(all_cases)
    rng.shuffle(shuffled)
    atlas_library = shuffled[:atlas_count]
    targets = shuffled[atlas_count:]
    return atlas_library, targets
