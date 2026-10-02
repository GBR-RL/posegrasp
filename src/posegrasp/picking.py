"""The grasp experiment: what do pose errors cost in picks?

For every target instance, a grasp is planned from the *estimated* pose in the observed scene
(posegrasp.grasp.Planner) and judged on the object at its *true* pose (posegrasp.grasp.judge).
The oracle plans from the true pose instead: its failures come from grasp planning and the scene,
not from pose estimation, so it bounds what any estimator can reach.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from posegrasp.data.bop import Dataset, Frame, ObjectModel, Pose
from posegrasp.geometry import FloatArray, backproject, load_model
from posegrasp.grasp import GraspSet, Planner, judge, synthesize

SCENE_STRIDE = 2  # every 2nd pixel in each direction: about 5 mm apart on the objects
ERROR_BINS = (0.05, 0.1, 0.2, 0.5)  # MSSD as a fraction of the diameter


def grasp_sets(
    models: dict[int, ObjectModel], cache_dir: Path, *, log: bool = False
) -> dict[int, GraspSet]:
    """Antipodal grasps per object, synthesised once and cached as .npz."""
    sets = {}
    for obj_id, model in sorted(models.items()):
        path = cache_dir / f"obj_{obj_id:06d}.npz"
        if not path.exists():
            start = time.perf_counter()
            surface = load_model(model.mesh_path, model.diameter).points
            synthesize(model.mesh_path, surface=surface).save(path)
            if log:
                print(
                    f"grasps for object {obj_id}: {time.perf_counter() - start:.0f} s", flush=True
                )
        sets[obj_id] = GraspSet.load(path)
    return sets


def scene_points(depth: FloatArray, K: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Camera-frame points of the whole image at SCENE_STRIDE, and the sampled-pixel mask."""
    sampled = np.zeros(depth.shape, dtype=bool)
    sampled[::SCENE_STRIDE, ::SCENE_STRIDE] = True
    return backproject(depth, sampled, K), sampled


def _pose(row: dict[str, Any]) -> FloatArray | None:
    if row.get("R") is None:
        return None
    return Pose(np.asarray(row["R"], dtype=float).reshape(3, 3), np.asarray(row["t"])).matrix()


def evaluate_grasps(
    dataset: Dataset,
    rows: Sequence[dict[str, Any]],
    *,
    grasps: dict[int, GraspSet],
    planner: Planner | None = None,
    oracle: bool = False,
) -> Iterator[dict[str, Any]]:
    """One grasp result per pose row (scene_id, im_id, obj_id, R, t), in row order."""
    planner = planner or Planner()
    models = dataset.models()
    by_image: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for r in rows:
        by_image.setdefault((r["scene_id"], r["im_id"]), []).append(r)
    for (scene_id, im_id), image_rows in by_image.items():
        frame: Frame = dataset.frame(scene_id, im_id)
        depth = frame.depth_mm()
        scene, sampled = scene_points(depth, frame.camera.K)
        valid = sampled & (depth > 0)
        for row in image_rows:
            model = models[row["obj_id"]]
            instance = next(g for g in frame.gt if g.obj_id == row["obj_id"])
            true_pose = instance.pose.matrix()
            target_pixels = frame.gt_mask(instance)[valid]
            obstacles = scene[~target_pixels]
            surface = load_model(model.mesh_path, model.diameter).points
            pose = true_pose if oracle else _pose(row)
            start = time.perf_counter()
            result: dict[str, Any] = {
                "scene_id": scene_id,
                "im_id": im_id,
                "obj_id": row["obj_id"],
                "visib_fract": instance.visib_fract,
                "diameter": model.diameter,
                "mssd": 0.0 if oracle else row.get("mssd"),
                "add": 0.0 if oracle else row.get("add"),
                "planned": False,
                "success": False,
                "reason": "no_pose",
                "candidates": 0,
                "feasible": 0,
                "grasp": None,
                "width": None,
            }
            if pose is not None:
                plan = planner.plan(
                    grasps[row["obj_id"]],
                    pose,
                    surface=surface,
                    scene=scene,
                    diameter=model.diameter,
                )
                result.update(candidates=plan.candidates, feasible=plan.feasible)
                result["reason"] = "no_grasp"
                if plan.grasp is not None:
                    outcome = judge(
                        plan.grasp,
                        true_pose,
                        mesh_path=model.mesh_path,
                        surface=surface,
                        obstacles=obstacles,
                        gripper=planner.gripper,
                    )
                    result.update(
                        planned=True,
                        success=outcome.success,
                        reason=outcome.reason,
                        grasp=plan.grasp.tolist(),
                        width=None if math.isnan(outcome.width) else outcome.width,
                    )
            result["seconds"] = time.perf_counter() - start
            yield result


def error_bin(row: dict[str, Any]) -> str:
    """MSSD of the pose the grasp was planned from, binned by the object diameter."""
    if row.get("mssd") is None:
        return "no pose"
    ratio = row["mssd"] / row["diameter"]
    lower = 0.0
    for upper in ERROR_BINS:
        if ratio < upper:
            return f"{lower:g}-{upper:g}"
        lower = upper
    return f">{ERROR_BINS[-1]:g}"


def summarize_grasps(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    planned = [r for r in rows if r["planned"]]
    bins: dict[str, list[bool]] = {}
    for r in rows:
        bins.setdefault(error_bin(r), []).append(bool(r["success"]))
    return {
        "instances": n,
        "success_rate": sum(r["success"] for r in rows) / n if n else 0.0,
        "planned_rate": len(planned) / n if n else 0.0,
        "success_when_planned": (
            sum(r["success"] for r in planned) / len(planned) if planned else 0.0
        ),
        "reasons": dict(Counter(r["reason"] for r in rows)),
        "by_mssd": {k: {"n": len(v), "success_rate": sum(v) / len(v)} for k, v in bins.items()},
        "seconds_p50": float(np.median([r["seconds"] for r in rows])) if rows else math.nan,
    }
