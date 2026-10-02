"""Pose-error functions and recall as defined by the BOP benchmark (Hodaň et al., ECCV 2020).

Errors are computed on the vertices of the BOP evaluation models (`models_eval/`), as the official
bop_toolkit does. Symmetric objects are handled by taking the minimum over the object's symmetry
transformations, discretised like bop_toolkit (`misc.get_symmetry_transformations`,
max_sym_disc_step = 0.01). VSD follows `pose_error.vsd` with the BOP'19 visibility rule and the
"step" pixel cost; tests/test_metrics.py checks every error against bop_toolkit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from posegrasp.data.bop import ObjectModel, Pose
from posegrasp.geometry import project
from posegrasp.render import Roi, projected_roi

FloatArray = NDArray[np.float64]

MSSD_THRESHOLDS = np.arange(0.05, 0.51, 0.05)  # x object diameter
MSPD_THRESHOLDS = np.arange(5, 51, 5)  # x image width / 640, pixels
VSD_DELTA = 15.0  # mm, tolerance of the visibility test
VSD_TAUS = np.arange(0.05, 0.51, 0.05)  # misalignment tolerance, x object diameter
VSD_THRESHOLDS = np.arange(0.05, 0.51, 0.05)  # correctness threshold on the VSD error

# Depth of the model at a pose over a pixel window (posegrasp.render.render_depth, bound to a mesh).
DepthRenderer = Callable[[Pose, Roi], FloatArray]


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
    """Average distance to the closest model point (ADD-S / ADI), millimetres: from each vertex in
    the true pose to the nearest vertex in the estimated pose, as bop_toolkit computes it."""
    distances, _ = cKDTree(_apply(est, vertices)).query(_apply(gt, vertices), k=1)
    return float(np.mean(distances))


def distance_image(depth: FloatArray, K: FloatArray, roi: Roi) -> FloatArray:
    """Distance from the camera centre per pixel, from a depth window (0 stays 0)."""
    x0, y0, x1, y1 = roi
    u, v = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    scale = np.sqrt(1.0 + ((u - K[0, 2]) / K[0, 0]) ** 2 + ((v - K[1, 2]) / K[1, 1]) ** 2)
    out: FloatArray = depth * scale
    return out


def _visible(d_test: FloatArray, d_model: FloatArray, delta: float) -> NDArray[np.bool_]:
    """BOP'19 visibility: the model surface is visible where it is not behind the measured
    surface by more than delta, or where the sensor has no depth."""
    diff = d_model.astype(np.float32) - d_test.astype(np.float32)
    visible: NDArray[np.bool_] = ((diff <= delta) | (d_test == 0)) & (d_model > 0)
    return visible


def vsd(
    est: Pose,
    gt: Pose,
    *,
    depth_test: FloatArray,
    K: FloatArray,
    vertices: FloatArray,
    diameter: float,
    render: DepthRenderer,
    delta: float = VSD_DELTA,
    taus: Sequence[float] = tuple(VSD_TAUS),
) -> list[float]:
    """Visible Surface Discrepancy, one error per misalignment tolerance in `taus`.

    Only the window covering both rendered silhouettes is rendered; pixels outside it are empty
    in both renderings and do not enter the error.
    """
    h, w = depth_test.shape
    boxes = [projected_roi(vertices, p, K, w, h) for p in (est, gt)]
    roi = (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )
    x0, y0, x1, y1 = roi
    if x1 <= x0 or y1 <= y0:
        return [1.0] * len(taus)
    dist_test = distance_image(depth_test[y0:y1, x0:x1], K, roi)
    dist_est = distance_image(render(est, roi), K, roi)
    dist_gt = distance_image(render(gt, roi), K, roi)
    visib_gt = _visible(dist_test, dist_gt, delta)
    visib_est = _visible(dist_test, dist_est, delta) | (visib_gt & (dist_est > 0))
    inter = visib_gt & visib_est
    union_count = int((visib_gt | visib_est).sum())
    if union_count == 0:
        return [1.0] * len(taus)
    complement = union_count - int(inter.sum())
    dists = np.abs(dist_gt[inter] - dist_est[inter]) / diameter
    return [float((np.sum(dists >= tau) + complement) / union_count) for tau in taus]


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
    rows: Sequence[dict[str, Any]], diameters: Sequence[float], image_width: int = 640
) -> dict[str, float]:
    """AR_VSD, AR_MSSD, AR_MSPD and their mean AR (the BOP score), and ADD(-S) accuracy at 10 %
    of the diameter.

    `rows` carry mssd, mspd, add, adi (inf for a miss), "symmetric" (1.0/0.0) and "vsd" (one error
    per tau; inf each for a miss; None if VSD was not computed, which makes AR_VSD and AR nan).
    """
    d = np.asarray(diameters, dtype=float)
    mssd_norm = np.asarray([r["mssd"] for r in rows]) / d
    mspd_norm = np.asarray([r["mspd"] for r in rows]) * 640 / image_width
    add_s = np.asarray([r["adi"] if r["symmetric"] else r["add"] for r in rows]) / d
    ar_mssd = recall_curve(mssd_norm.tolist(), MSSD_THRESHOLDS.tolist())
    ar_mspd = recall_curve(mspd_norm.tolist(), MSPD_THRESHOLDS.tolist())
    ar_vsd = math.nan
    vsd_rows = [r.get("vsd") for r in rows]
    if not rows:
        ar_vsd = 0.0
    elif all(v is not None for v in vsd_rows):
        e = np.asarray(vsd_rows, dtype=float)  # (instances, taus)
        ar_vsd = float(
            np.mean([np.mean(e[:, i] < th) for i in range(e.shape[1]) for th in VSD_THRESHOLDS])
        )
    return {
        "instances": float(len(rows)),
        "ar": (ar_vsd + ar_mssd + ar_mspd) / 3,
        "ar_vsd": ar_vsd,
        "ar_mssd": ar_mssd,
        "ar_mspd": ar_mspd,
        "add_s_01d": float(np.mean(add_s < 0.1)) if len(rows) else 0.0,
    }
