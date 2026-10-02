"""Evaluation pipeline: target -> segmented scene points -> pose estimate -> BOP errors.

Detection conditions (see docs/EVAL_PROTOCOL.md):
- `gt`      the ground-truth visible mask of the instance (upper bound for pose estimation)
- `gdrnpp`  BOP'23 default boxes of a detector trained on the objects (no mask)
- `cnos`    BOP'23 default masks of a zero-shot detector for unseen objects
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from posegrasp import metrics
from posegrasp.data import detections as det
from posegrasp.data.bop import Dataset, Frame, ObjectModel, Pose, Target
from posegrasp.estimators.base import Estimate, PoseEstimator, best
from posegrasp.geometry import (
    FloatArray,
    ModelCloud,
    backproject,
    load_model,
    load_vertices,
    remove_outliers,
)
from posegrasp.render import depth_renderer
from posegrasp.verify import DepthVerifier

CONDITIONS = {
    "gt": None,
    "gdrnpp": "det_gdrnppdet-pbr_lmo.json",
    "cnos": "det_cnos-fastsam_lmo.json",
}
MAX_POINTS = 20_000


@dataclass(frozen=True, slots=True)
class Segment:
    points: FloatArray
    mask: NDArray[np.bool_]  # pixels of the kept points (what verification must explain)
    detection_score: float
    detection_seconds: float


def points_mask(points: FloatArray, K: FloatArray, height: int, width: int) -> NDArray[np.bool_]:
    mask = np.zeros((height, width), dtype=bool)
    if len(points) == 0:
        return mask
    u = np.round(points[:, 0] * K[0, 0] / points[:, 2] + K[0, 2]).astype(int)
    v = np.round(points[:, 1] * K[1, 1] / points[:, 2] + K[1, 2]).astype(int)
    ok = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    mask[v[ok], u[ok]] = True
    return mask


def crop_box_segment(points: FloatArray, box_center: FloatArray, diameter: float) -> FloatArray:
    """Box-only detections: keep the points within about one object radius of the surface
    point at the box centre, which drops most background and far occluders."""
    if len(points) == 0:
        return points
    near = np.linalg.norm(points[:, :2] - box_center[:2], axis=1)
    core = points[near <= np.quantile(near, 0.25)]
    center = np.median(core, axis=0)
    center[2] += 0.25 * diameter  # the box centre sees the front surface; move toward the middle
    out: FloatArray = points[np.linalg.norm(points - center, axis=1) <= 0.6 * diameter]
    return out


def segment(
    frame: Frame,
    depth: FloatArray,
    *,
    obj_id: int,
    model: ObjectModel,
    condition: str,
    detection: det.Detection | None,
) -> Segment | None:
    K, h, w = frame.camera.K, frame.camera.height, frame.camera.width
    if condition == "gt":
        instance = next(g for g in frame.gt if g.obj_id == obj_id)
        mask: NDArray[np.bool_] = frame.gt_mask(instance)
        score, seconds = 1.0, 0.0
    else:
        if detection is None:
            return None
        mask = detection.mask(h, w)
        score, seconds = detection.score, detection.seconds
    points = backproject(depth, mask, K, max_points=MAX_POINTS)
    if condition != "gt" and detection is not None and not detection.has_mask:
        x, y, bw, bh = detection.bbox
        u, v = x + bw / 2, y + bh / 2
        center_ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        z = np.median(points[:, 2]) if len(points) else 0.0
        points = crop_box_segment(points, center_ray * z, model.diameter)
    points = remove_outliers(points)
    kept = mask if condition == "gt" or (detection is not None and detection.has_mask) else None
    if kept is None:
        kept = points_mask(points, K, h, w)
    return Segment(points, kept & (depth > 0), score, seconds)


def gt_pose(frame: Frame, obj_id: int) -> tuple[Pose, float]:
    instance = next(g for g in frame.gt if g.obj_id == obj_id)
    return instance.pose, instance.visib_fract


@dataclass(frozen=True, slots=True)
class InstanceResult:
    segment: Segment | None  # None: no detection for this instance
    hypotheses: int
    estimate: Estimate | None  # None: no pose
    seconds: float  # segmentation + hypotheses + ICP + verification


def estimate_instance(
    frame: Frame,
    depth: FloatArray,
    *,
    model: ObjectModel,
    estimator: PoseEstimator,
    condition: str,
    detection: det.Detection | None,
    selection: str = "score",
    verifier: DepthVerifier | None = None,
) -> InstanceResult:
    """Pose of one object instance: segment the depth, propose and refine hypotheses, select one.

    selection: "score" keeps the hypothesis with the best ICP inlier fraction, "verify" the one
    that best explains the measured depth (posegrasp.verify).
    """
    start = time.perf_counter()
    seg = segment(
        frame, depth, obj_id=model.obj_id, model=model, condition=condition, detection=detection
    )
    if seg is None or len(seg.points) < 10:
        return InstanceResult(seg, 0, None, time.perf_counter() - start)
    cloud: ModelCloud = load_model(model.mesh_path, model.diameter)
    hyps = [h for h in estimator.hypotheses(model.obj_id, cloud, seg.points) if h.pose]
    if not hyps:
        return InstanceResult(seg, 0, None, time.perf_counter() - start)
    if selection == "verify":
        verifier = verifier or DepthVerifier()
        hyps = [
            Estimate(
                h.pose,
                verifier.score(
                    model.mesh_path,
                    model.diameter,
                    h.pose,  # type: ignore[arg-type]
                    depth_mm=depth,
                    mask=seg.mask,
                    K=frame.camera.K,
                ),
                h.seconds,
            )
            for h in hyps
        ]
    elif selection != "score":
        raise ValueError(f"unknown selection '{selection}' (score | verify)")
    return InstanceResult(seg, len(hyps), best(hyps), time.perf_counter() - start)


def evaluate_targets(
    dataset: Dataset,
    targets: Sequence[Target],
    *,
    estimator: PoseEstimator,
    condition: str,
    detections_dir: Path,
    selection: str = "score",
) -> Iterator[dict[str, Any]]:
    """One result row per target instance, in target order, with all BOP errors."""
    verifier = DepthVerifier()
    models = dataset.models()
    chosen: dict[tuple[int, int, int], list[det.Detection]] = {}
    if CONDITIONS[condition] is not None:
        chosen = det.for_targets(det.load(detections_dir / str(CONDITIONS[condition])), targets)
    by_image: dict[tuple[int, int], list[Target]] = {}
    for t in targets:
        by_image.setdefault((t.scene_id, t.im_id), []).append(t)
    symmetries = {i: metrics.symmetry_transforms(m) for i, m in models.items()}
    for frame in dataset.frames(targets):
        depth = frame.depth_mm()
        K = frame.camera.K
        for t in by_image[frame.key]:
            model = models[t.obj_id]
            vertices = load_vertices(model.eval_mesh)
            gt, visib = gt_pose(frame, t.obj_id)
            candidates = chosen.get((t.scene_id, t.im_id, t.obj_id), [])
            for k in range(t.inst_count):
                result = estimate_instance(
                    frame,
                    depth,
                    model=model,
                    estimator=estimator,
                    condition=condition,
                    detection=candidates[k] if k < len(candidates) else None,
                    selection=selection,
                    verifier=verifier,
                )
                est, seg = result.estimate, result.segment
                pose = est.pose if est is not None else None
                errors = metrics.pose_errors(
                    pose, gt, vertices=vertices, K=K, symmetries=symmetries[t.obj_id]
                )
                vsd = None
                if pose is not None:
                    vsd = metrics.vsd(
                        pose,
                        gt,
                        depth_test=depth,
                        K=K,
                        vertices=vertices,
                        diameter=model.diameter,
                        render=depth_renderer(model.eval_mesh, K),
                    )
                yield {
                    "scene_id": t.scene_id,
                    "im_id": t.im_id,
                    "obj_id": t.obj_id,
                    "condition": condition,
                    "estimator": estimator.name,
                    "selection": selection,
                    "hypotheses": result.hypotheses,
                    "visib_fract": visib,
                    "diameter": model.diameter,
                    "symmetric": float(model.is_symmetric),
                    "detected": seg is not None,
                    "points": 0 if seg is None else len(seg.points),
                    "detection_score": None if seg is None else seg.detection_score,
                    "detection_seconds": None if seg is None else seg.detection_seconds,
                    "score": None if est is None else est.score,
                    "seconds": result.seconds,
                    "R": None if pose is None else pose.R.reshape(-1).tolist(),
                    "t": None if pose is None else pose.t.tolist(),
                    **{k2: (None if math.isinf(v) else v) for k2, v in errors.items()},
                    "vsd": vsd,
                }


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _vsd_or_miss(row: dict[str, Any]) -> list[float] | None:
    if "vsd" not in row:  # rows written before VSD was computed
        return None
    vsd: list[float] | None = row["vsd"]
    return vsd if vsd is not None else [math.inf] * len(metrics.VSD_TAUS)


def summarize_rows(rows: Sequence[dict[str, Any]], image_width: int = 640) -> dict[str, float]:
    inf = math.inf
    prepared = [
        {
            "mssd": inf if r["mssd"] is None else r["mssd"],
            "mspd": inf if r["mspd"] is None else r["mspd"],
            "add": inf if r["add"] is None else r["add"],
            "adi": inf if r["adi"] is None else r["adi"],
            "symmetric": r["symmetric"],
            "vsd": _vsd_or_miss(r),
        }
        for r in rows
    ]
    summary = metrics.summarize(prepared, [r["diameter"] for r in rows], image_width)
    seconds = [r["seconds"] for r in rows if r["score"] is not None]
    summary["seconds_p50"] = float(np.median(seconds)) if seconds else math.nan
    return summary


def to_bop_csv(rows: Sequence[dict[str, Any]]) -> str:
    """Results in the BOP submission format (score with the official bop_toolkit)."""
    lines = ["scene_id,im_id,obj_id,score,R,t,time"]
    image_time: dict[tuple[int, int], float] = {}
    for r in rows:
        key = (r["scene_id"], r["im_id"])
        image_time[key] = image_time.get(key, 0.0) + r["seconds"] + (r["detection_seconds"] or 0.0)
    for r in rows:
        if r["R"] is None:
            continue
        R = " ".join(f"{v:.6f}" for v in r["R"])
        t = " ".join(f"{v:.3f}" for v in r["t"])
        score = r["score"] if r["score"] is not None else 0.0
        lines.append(
            f"{r['scene_id']},{r['im_id']},{r['obj_id']},{score:.6f},{R},{t},"
            f"{image_time[(r['scene_id'], r['im_id'])]:.4f}"
        )
    return "\n".join(lines) + "\n"
