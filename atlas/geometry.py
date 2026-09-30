"""
Image-geometry helpers for the atlas registration pipeline.

Every point in the registration code is in SimpleITK physical space (LPS, mm).
nibabel uses RAS instead. The two differ by a sign flip on x and y, so mixing
them silently mirrors the anatomy left-right and front-back. To avoid that,
landmarks, transforms and warps are all computed from SimpleITK-loaded images.
RAS appears only when a result is written with nibabel (sitk_index_to_ras_affine).

Index convention: a SimpleITK index is (x, y, z), but sitk.GetArrayFromImage
returns arrays indexed [z, y, x]. to_xyz_array / from_xyz_array convert between
the two so that array[i, j, k] matches SimpleITK index (i, j, k).
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def read_ct(path: str, clip=(-1000.0, 1000.0)) -> sitk.Image:
    img = sitk.Cast(sitk.ReadImage(str(path)), sitk.sitkFloat32)
    if clip is not None:
        img = sitk.Clamp(img, sitk.sitkFloat32, float(clip[0]), float(clip[1]))
    return img


def read_label(path: str) -> sitk.Image:
    return sitk.Cast(sitk.ReadImage(str(path)), sitk.sitkUInt8)


def sitk_index_to_lps_affine(image: sitk.Image) -> np.ndarray:
    """4x4 matrix mapping homogeneous (x, y, z, 1) index -> LPS physical point (mm)."""
    direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.array(image.GetSpacing(), dtype=np.float64)
    origin = np.array(image.GetOrigin(), dtype=np.float64)
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


def sitk_index_to_ras_affine(image: sitk.Image) -> np.ndarray:
    """Same as sitk_index_to_lps_affine but in RAS, i.e. a nibabel-compatible affine."""
    return _LPS_TO_RAS @ sitk_index_to_lps_affine(image)


def index_to_physical(image: sitk.Image, index_xyz: np.ndarray) -> np.ndarray:
    """Vectorised TransformContinuousIndexToPhysicalPoint. (N, 3) -> (N, 3)."""
    affine = sitk_index_to_lps_affine(image)
    index_xyz = np.asarray(index_xyz, dtype=np.float64)
    return index_xyz @ affine[:3, :3].T + affine[:3, 3]


def physical_to_index(image: sitk.Image, points_lps: np.ndarray) -> np.ndarray:
    """Vectorised TransformPhysicalPointToContinuousIndex. (N, 3) -> (N, 3) continuous (x, y, z)."""
    affine = sitk_index_to_lps_affine(image)
    inv = np.linalg.inv(affine)
    points_lps = np.asarray(points_lps, dtype=np.float64)
    return points_lps @ inv[:3, :3].T + inv[:3, 3]


def to_xyz_array(image: sitk.Image) -> np.ndarray:
    """Array indexed [x, y, z], matching SimpleITK index order."""
    return np.transpose(sitk.GetArrayFromImage(image), (2, 1, 0))


def make_reference_grid(like: sitk.Image, spacing_mm: float, pad_voxels: int = 1) -> sitk.Image:
    """
    An empty image on an isotropic grid covering at least the same physical
    extent as `like` (same origin and direction), for use as a resampling
    reference. `pad_voxels` extra voxels on the far side keep the last partial
    cell inside the grid.
    """
    size = np.array(like.GetSize())
    spacing = np.array(like.GetSpacing())
    new_size = np.ceil(size * spacing / spacing_mm).astype(int) + pad_voxels
    ref = sitk.Image([int(s) for s in new_size], sitk.sitkUInt8)
    ref.SetOrigin(like.GetOrigin())
    ref.SetSpacing((float(spacing_mm),) * 3)
    ref.SetDirection(like.GetDirection())
    return ref


def resample_to_grid(image: sitk.Image, reference: sitk.Image, interpolator, default_value: float, transform=None) -> sitk.Image:
    transform = transform if transform is not None else sitk.Transform()
    return sitk.Resample(image, reference, transform, interpolator, float(default_value), image.GetPixelID())


def resample_isotropic(image: sitk.Image, spacing_mm: float, interpolator, default_value: float) -> sitk.Image:
    return resample_to_grid(image, make_reference_grid(image, spacing_mm, pad_voxels=0), interpolator, default_value)


def grid_physical_points(reference: sitk.Image) -> np.ndarray:
    """
    Physical (LPS) coordinates of every voxel of `reference`, ordered to match
    sitk.GetArrayFromImage (z slowest, x fastest). Returns (Z*Y*X, 3).
    """
    sx, sy, sz = reference.GetSize()
    zz, yy, xx = np.meshgrid(np.arange(sz), np.arange(sy), np.arange(sx), indexing="ij")
    index_xyz = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float64)
    return index_to_physical(reference, index_xyz)


def affine_matrix_to_sitk(matrix_4x4: np.ndarray) -> sitk.AffineTransform:
    """4x4 homogeneous matrix (point -> point) -> equivalent sitk.AffineTransform."""
    tx = sitk.AffineTransform(3)
    tx.SetMatrix(matrix_4x4[:3, :3].ravel().tolist())
    tx.SetTranslation(matrix_4x4[:3, 3].tolist())
    tx.SetCenter((0.0, 0.0, 0.0))
    return tx


def sitk_affine_to_matrix(tx: sitk.AffineTransform) -> np.ndarray:
    """sitk.AffineTransform (with arbitrary center) -> 4x4 homogeneous matrix."""
    M = np.array(tx.GetMatrix(), dtype=np.float64).reshape(3, 3)
    c = np.array(tx.GetCenter(), dtype=np.float64)
    t = np.array(tx.GetTranslation(), dtype=np.float64)
    out = np.eye(4)
    out[:3, :3] = M
    out[:3, 3] = c + t - M @ c
    return out


def apply_affine(matrix_4x4: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix_4x4[:3, :3].T + matrix_4x4[:3, 3]
