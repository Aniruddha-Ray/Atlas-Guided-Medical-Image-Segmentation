"""
Label warping (research report Section 1.5): L'_i(v) = L_i(T_i^{-1}(v)).

Registration.map_target_to_atlas already IS T_i^{-1} (target -> atlas), so
warping is a pull-back resample: for every target voxel, sample the atlas
label at the mapped atlas point with nearest neighbour.

The TPS is evaluated on a coarse grid (default 6 mm) and turned into a
SimpleITK displacement field, which SimpleITK interpolates linearly while
resampling. A TPS is smooth, so this is accurate, and it is orders of
magnitude cheaper than evaluating the spline at every output voxel.

Warped labels of all atlases for one target are stored together as a single
4D uint8 NIfTI (X, Y, Z, N_atlas) on the target's 2 mm grid. P(v), D(v) and
w(v) are NOT stored: they are cheap to compute from this stack and must be
computed AFTER the training pipeline's crop/resize/augmentation so they stay
aligned with the augmented CT (see research_audit.md / discussion log).
"""
from __future__ import annotations

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from atlas.geometry import grid_physical_points, make_reference_grid, sitk_index_to_ras_affine, to_xyz_array


def displacement_transform(mapping, reference: sitk.Image, field_spacing_mm: float = 6.0) -> sitk.DisplacementFieldTransform:
    """
    mapping: callable (N, 3) target LPS -> (N, 3) atlas LPS.
    reference: the output grid (the displacement field covers its extent).
    """
    field_grid = make_reference_grid(reference, field_spacing_mm, pad_voxels=2)
    pts = grid_physical_points(field_grid)
    disp = mapping(pts) - pts
    sx, sy, sz = field_grid.GetSize()
    disp_img = sitk.GetImageFromArray(disp.reshape(sz, sy, sx, 3).astype(np.float64), isVector=True)
    disp_img.CopyInformation(field_grid)
    return sitk.DisplacementFieldTransform(disp_img)


def warp_label(atlas_label: sitk.Image, reference: sitk.Image, transform) -> sitk.Image:
    return sitk.Resample(atlas_label, reference, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)


def warp_ct(atlas_ct: sitk.Image, reference: sitk.Image, transform) -> sitk.Image:
    """For visual QA only."""
    return sitk.Resample(atlas_ct, reference, transform, sitk.sitkLinear, -1000.0, sitk.sitkFloat32)


def vote_counts(stack_xyzn: np.ndarray, num_classes: int) -> np.ndarray:
    """(X, Y, Z, N) uint8 labels -> (C, X, Y, Z) uint8 per-class vote counts."""
    return np.stack([(stack_xyzn == c).sum(axis=-1).astype(np.uint8) for c in range(num_classes)])


def save_label_stack(warped: list, reference: sitk.Image, path: str):
    """warped: list of sitk uint8 images on `reference`'s grid -> 4D NIfTI (X, Y, Z, N)."""
    stack = np.stack([to_xyz_array(img) for img in warped], axis=-1).astype(np.uint8)
    img = nib.Nifti1Image(stack, sitk_index_to_ras_affine(reference))
    img.header.set_data_dtype(np.uint8)
    nib.save(img, str(path))
    return stack
