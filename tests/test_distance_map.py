"""
Verifies the CustomSpatialDistanceTransformd bugfix (research_audit.md,
Section G) and the required shape/value invariants from Section 32 of the
project brief: D.shape matches the label, and D >= 0 everywhere.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("monai")

from data.transforms import CustomSpatialDistanceTransformd  # noqa: E402


def _one_hot_label_with_single_voxel_organ(size=9, organ_channel=1, num_channels=3):
    """(C, D, H, W) one-hot label: channel 0 = background, one organ voxel at the center."""
    label = torch.zeros((num_channels, size, size, size), dtype=torch.float32)
    label[0] = 1.0
    center = size // 2
    label[0, center, center, center] = 0.0
    label[organ_channel, center, center, center] = 1.0
    return label, center


def test_distance_is_zero_inside_organ_and_grows_outside():
    label, center = _one_hot_label_with_single_voxel_organ()
    transform = CustomSpatialDistanceTransformd(
        keys=["label"], to_closest_foreground=True, include_background=False, output_postfix="_distance"
    )
    out = transform({"label": label})
    dist = out["label_distance"]  # (num_channels - 1, D, H, W); channel 0 here == organ_channel 1

    organ_dist = dist[0]
    assert organ_dist.shape == label.shape[1:]
    assert (organ_dist >= 0).all()

    # 0 at the organ voxel itself.
    assert organ_dist[center, center, center].item() == 0.0

    # A neighboring voxel must be strictly farther than the organ voxel, and a
    # far corner must be farther still -- i.e. distance grows with distance
    # from the organ, which is the whole point of D_c(v).
    neighbor_dist = organ_dist[center, center, center + 1].item()
    corner_dist = organ_dist[0, 0, 0].item()
    assert neighbor_dist > 0.0
    assert corner_dist > neighbor_dist


def _one_hot_label_with_solid_organ_block(size=9, half_width=2, organ_channel=1, num_channels=3):
    """(C, D, H, W) one-hot label: channel 0 = background, a solid 5^3 organ block centered in the volume."""
    label = torch.zeros((num_channels, size, size, size), dtype=torch.float32)
    label[0] = 1.0
    c = size // 2
    lo, hi = c - half_width, c + half_width + 1
    label[0, lo:hi, lo:hi, lo:hi] = 0.0
    label[organ_channel, lo:hi, lo:hi, lo:hi] = 1.0
    return label, c


def test_legacy_buggy_distance_reproduces_original_inverted_behavior():
    """
    Documents the original bug so it can't silently regress back in.

    With legacy_buggy_distance=True, distance_transform_edt is called
    directly on the organ mask, which measures depth *inward* from the
    organ's own boundary (0 at the boundary, growing toward the center) and
    is exactly 0 for every background voxel, no matter how far it is from
    the organ. That is backwards from D_c(v) = min_{y in O_c} ||v-y||, which
    should be 0 *inside* the organ and grow *outside* it -- verified below by
    comparing against the fixed (default) semantics on the same input.
    """
    label, center = _one_hot_label_with_solid_organ_block(half_width=2)
    boundary_offset = 2  # half_width used above; block spans [center-2, center+2]

    buggy_transform = CustomSpatialDistanceTransformd(
        keys=["label"],
        to_closest_foreground=True,
        include_background=False,
        output_postfix="_distance",
        legacy_buggy_distance=True,
    )
    buggy_dist = buggy_transform({"label": label})["label_distance"][0]

    # Every background voxel reads exactly 0 in the buggy version, whether it
    # is adjacent to the organ or on the far side of the volume -- no
    # positional signal at all outside the organ.
    assert buggy_dist[0, 0, 0].item() == 0.0
    assert buggy_dist[center, center, center + boundary_offset + 1].item() == 0.0
    # Inside the organ, the buggy version instead grows with depth (wrong
    # direction): 1 at the organ's own boundary, more at its center.
    assert buggy_dist[center, center, center - boundary_offset].item() == 1.0
    assert buggy_dist[center, center, center].item() == boundary_offset + 1

    fixed_transform = CustomSpatialDistanceTransformd(
        keys=["label"], to_closest_foreground=True, include_background=False, output_postfix="_distance"
    )
    fixed_dist = fixed_transform({"label": label})["label_distance"][0]

    # Fixed version: exactly the opposite -- 0 throughout the organ interior,
    # and background grows the farther it is from the organ.
    assert fixed_dist[center, center, center].item() == 0.0
    near_bg = fixed_dist[center, center, center + boundary_offset + 1].item()
    far_bg = fixed_dist[0, 0, 0].item()
    assert near_bg > 0.0
    assert far_bg > near_bg


def test_distance_map_channel_count_excludes_background():
    label, _ = _one_hot_label_with_single_voxel_organ(num_channels=13)
    transform = CustomSpatialDistanceTransformd(
        keys=["label"], to_closest_foreground=True, include_background=False, output_postfix="_distance"
    )
    out = transform({"label": label})
    assert out["label_distance"].shape[0] == 12  # 13 classes minus background
