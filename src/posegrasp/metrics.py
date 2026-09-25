"""Pose-error functions and recall as defined by the BOP benchmark (Hodaň et al., ECCV 2020).

Errors are computed on the model vertices. Symmetric objects are handled by taking the minimum
over the object's symmetry transformations, discretised like bop_toolkit
(`misc.get_symmetry_transformations`, max_sym_disc_step = 0.01).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from posegrasp.data.bop import ObjectModel, Pose
from posegrasp.geometry import project

FloatArray = NDArray[np.float64]

MSSD_THRESHOLDS = np.arange(0.05, 0.51, 0.05)  # x object diameter
MSPD_THRESHOLDS = np.arange(5, 51, 5)  # x image width / 640, pixels


def symmetry_transforms(model: ObjectModel, max_step: float = 0.01) -> list[FloatArray]:
    """4x4 model-frame transforms mapping the object onto itself, identity first."""
    discrete = [np.eye(4), *[np.asarray(s, dtype=float) for s in model.symmetries_discrete]]
    continuous: list[FloatArray] = [np.eye(4)]
    for sym in model.symmetries_continuous:
        axis = np.asarray(sym["axis"], dtype=float)
        axis /= np.linalg.norm(axis)
        offset = np.asarray(sym["offset"], dtype=float)
        steps = math.ceil(math.pi / max_step)
        for i in range(1, steps):
            R = _axis_angle(axis, 2 * math.pi * i / steps)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, -R @ offset + offset
            continuous.append(T)
    return [c @ d for d in discrete for c in continuous]


def _axis_angle(axis: FloatArray, angle: float) -> FloatArray:
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    out: FloatArray = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)
    return out


def _apply(pose: Pose, points: FloatArray) -> FloatArray:
    out: FloatArray = points @ pose.R.T + pose.t
    return out


def _gt_variants(gt: Pose, symmetries: Sequence[FloatArray]) -> list[Pose]:
    T = gt.matrix()
    return [Pose.from_matrix(T @ S) for S in symmetries]


def mssd(est: Pose, gt: Pose, vertices: FloatArray, symmetries: Sequence[FloatArray]) -> float:
    """Maximum Symmetry-aware Surface Distance, millimetres."""
    e = _apply(est, vertices)
    return min(
        float(np.linalg.norm(e - _apply(g, vertices), axis=1).max())
        for g in _gt_variants(gt, symmetries)
    )


def mspd(
    est: Pose, gt: Pose, vertices: FloatArray, symmetries: Sequence[FloatArray], K: FloatArray
) -> float:
    """Maximum Symmetry-aware Projection Distance, pixels."""
    e = project(_apply(est, vertices), K)
    return min(
        float(np.linalg.norm(e - project(_apply(g, vertices), K), axis=1).max())
        for g in _gt_variants(gt, symmetries)
    )


def add(est: Pose, gt: Pose, vertices: FloatArray) -> float:
    """Average distance of corresponding model points (ADD), millimetres."""
    return float(np.linalg.norm(_apply(est, vertices) - _apply(gt, vertices), axis=1).mean())


def adi(est: Pose, gt: Pose, vertices: FloatArray) -> float:
    """Average distance to the closest model point (ADD-S / ADI), millimetres."""
    distances, _ = cKDTree(_apply(gt, vertices)).query(_apply(est, vertices), k=1)
    return float(np.mean(distances))


def recall_curve(errors: Sequence[float], thresholds: Sequence[float]) -> float:
    """Mean over thresholds of the fraction of errors below the threshold (misses: inf)."""
    e = np.asarray(errors, dtype=float)
    if e.size == 0:
        return 0.0
    return float(np.mean([np.mean(e < th) for th in thresholds]))


def pose_errors(
    est: Pose | None,
    gt: Pose,
    *,
    vertices: FloatArray,
    K: FloatArray,
    symmetries: Sequence[FloatArray],
) -> dict[str, float]:
    """All errors of one estimate; a missing estimate gets infinite errors."""
    if est is None:
        return {"mssd": math.inf, "mspd": math.inf, "add": math.inf, "adi": math.inf}
    return {
        "mssd": mssd(est, gt, vertices, symmetries),
        "mspd": mspd(est, gt, vertices, symmetries, K),
        "add": add(est, gt, vertices),
        "adi": adi(est, gt, vertices),
    }


def summarize(
    rows: Sequence[dict[str, float]], diameters: Sequence[float], image_width: int = 640
) -> dict[str, float]:
    """AR_MSSD, AR_MSPD (BOP) and ADD(-S) accuracy at 10 % of the diameter.

    `rows` carry mssd, mspd, add, adi and "symmetric" (1.0/0.0) per target instance.
    """
    d = np.asarray(diameters, dtype=float)
    mssd_norm = np.asarray([r["mssd"] for r in rows]) / d
    mspd_norm = np.asarray([r["mspd"] for r in rows]) * 640 / image_width
    add_s = np.asarray([r["adi"] if r["symmetric"] else r["add"] for r in rows]) / d
    ar_mssd = recall_curve(mssd_norm.tolist(), MSSD_THRESHOLDS.tolist())
    ar_mspd = recall_curve(mspd_norm.tolist(), MSPD_THRESHOLDS.tolist())
    return {
        "instances": float(len(rows)),
        "ar_mssd": ar_mssd,
        "ar_mspd": ar_mspd,
        "add_s_01d": float(np.mean(add_s < 0.1)) if len(rows) else 0.0,
    }
