"""
3D thin-plate spline (TPS), regularized ("smoothing spline").

Neither SimpleITK nor scipy ship a 3D TPS (SimpleITK's registration methods
are all dense/intensity-based; scipy.interpolate.RBFInterpolator can be
configured to behave like one but doesn't expose the classical
affine+warp decomposition or a jacobian-determinant check for folding), so
this is a small from-scratch implementation of the classical formulation:

    T(x) = A x + b + sum_k w_k * phi(||x - c_k||)

where c_k are the control points (correspondences), phi(r) = r is the
fundamental radial basis solution used for 3D thin-plate splines (in 2D the
usual choice is r^2 log(r); in 3D it's r), and the weights w_k are
constrained (sum w_k = 0, sum w_k c_k = 0) so the spline doesn't reintroduce
its own affine component on top of A, b.

`regularization > 0` turns this from an exact interpolant (every control
point maps exactly onto its target) into a smoothing spline: noisy/outlier
correspondences get partially absorbed into the smooth affine+low-frequency
part of the field instead of forcing a sharp local bend. This matters here
because atlas-to-target correspondences from intensity-patch matching WILL
contain some bad matches even after RANSAC.

Sign of the regularisation term: with phi(r) = r the kernel matrix K is
conditionally NEGATIVE definite (distance matrices are of negative type), so
the smoothing system is (K - lambda*I), not (K + lambda*I). Minimising
sum |f(x_i) - v_i|^2 + lambda * bending_energy, with bending energy
-w'Kw >= 0, gives M[(K - lambda*I) w + P c - v] = 0. The opposite sign is
near-singular whenever lambda matches an eigenvalue of -K, which produced
erratic fits (residuals of 100-4000 mm, weights in the hundreds, 40-55 % of
control points folding) in an earlier version of this file. With the correct
sign the fit moves smoothly and monotonically from exact interpolation
(lambda = 0) to the plain affine least-squares fit (lambda -> infinity).

lambda is in units of mm of kernel distance, so useful values are large
(tens to thousands for control points spread over ~100s of mm), not O(1).
"""
from __future__ import annotations

import numpy as np


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,3), b: (M,3) -> (N,M) Euclidean distances."""
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=-1)


class ThinPlateSpline:
    def __init__(self, source_points: np.ndarray, target_points: np.ndarray, regularization: float = 0.0):
        """
        Fits T such that T(source_points[k]) ~= target_points[k].

        source_points, target_points: (N, 3), N >= 4 (need at least 4 non-
        coplanar points to fix the affine part in 3D).
        regularization: smoothing weight lambda (see module docstring for its
        scale); 0.0 = exact interpolation, large = the affine least-squares fit.
        """
        source_points = np.asarray(source_points, dtype=np.float64)
        target_points = np.asarray(target_points, dtype=np.float64)
        if source_points.shape != target_points.shape:
            raise ValueError("source_points and target_points must have the same shape")
        n = source_points.shape[0]
        if n < 4:
            raise ValueError(f"need at least 4 correspondences to fit a 3D TPS, got {n}")

        self.control_points = source_points

        K = _pairwise_distances(source_points, source_points)  # phi(r) = r
        if regularization > 0:
            K = K - regularization * np.eye(n)  # minus: K is conditionally negative definite, see module docstring

        P = np.concatenate([np.ones((n, 1)), source_points], axis=1)  # (N, 4): [1, x, y, z]

        top = np.concatenate([K, P], axis=1)  # (N, N+4)
        bottom = np.concatenate([P.T, np.zeros((4, 4))], axis=1)  # (4, N+4)
        L = np.concatenate([top, bottom], axis=0)  # (N+4, N+4)

        rhs = np.concatenate([target_points, np.zeros((4, 3))], axis=0)  # (N+4, 3)

        solution, *_ = np.linalg.lstsq(L, rhs, rcond=None)
        self.weights = solution[:n]  # (N, 3)
        affine_params = solution[n:]  # (4, 3): [b; A_row_for_x; A_row_for_y; A_row_for_z]... see transform()
        self.translation = affine_params[0]  # (3,)
        self.affine_matrix = affine_params[1:]  # (3, 3), applied as source_points @ affine_matrix

    def transform(self, points: np.ndarray, chunk_size: int = 16384) -> np.ndarray:
        """
        points: (M, 3) in source space -> (M, 3) in target space.

        Evaluated in chunks so a dense grid (hundreds of thousands of points)
        never needs an (M, N) distance matrix in memory at once.
        """
        points = np.asarray(points, dtype=np.float64)
        out = np.empty_like(points)
        ctrl = self.control_points
        ctrl_sq = (ctrl**2).sum(axis=1)
        for start in range(0, points.shape[0], chunk_size):
            chunk = points[start : start + chunk_size]
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2ab avoids an (M, N, 3) intermediate.
            sq = (chunk**2).sum(axis=1)[:, None] + ctrl_sq[None, :] - 2.0 * chunk @ ctrl.T
            R = np.sqrt(np.maximum(sq, 0.0))
            out[start : start + chunk_size] = chunk @ self.affine_matrix + self.translation + R @ self.weights
        return out

    def jacobian_determinant(self, points: np.ndarray, eps: float = 1e-2) -> np.ndarray:
        """
        Numerically estimates det(J) of the transform at each point via
        central differences. det(J) <= 0 means the deformation folds
        (locally non-invertible) at that point -- a registration QA flag,
        not something to silently accept.
        """
        points = np.asarray(points, dtype=np.float64)
        m = points.shape[0]
        jac = np.zeros((m, 3, 3))
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = eps
            plus = self.transform(points + offset)
            minus = self.transform(points - offset)
            jac[:, :, axis] = (plus - minus) / (2 * eps)
        return np.linalg.det(jac)
