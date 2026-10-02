"""The estimator interface shared by all methods.

An estimator proposes coarse pose hypotheses; every hypothesis is refined by the same ICP
(`Refiner`), so methods differ only in how they search for the pose. The pipeline then keeps the
best hypothesis, either by the method's own score or by depth verification (posegrasp.verify).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from posegrasp.data.bop import Pose
from posegrasp.geometry import ModelCloud, estimate_normals, to_open3d

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Estimate:
    pose: Pose | None  # None: the method found no pose
    score: float  # method-specific confidence (after ICP: inlier fraction), higher is better
    seconds: float


class PoseEstimator(Protocol):
    name: str

    def prepare(self, obj_id: int, model: ModelCloud) -> None:
        """Offline work per object (model features, training); excluded from the latency."""
        ...

    def hypotheses(self, obj_id: int, model: ModelCloud, scene: FloatArray) -> list[Estimate]:
        """Refined pose hypotheses (model -> camera) from the segmented scene points, mm."""
        ...


def best(estimates: list[Estimate]) -> Estimate:
    if not estimates:
        return Estimate(None, 0.0, 0.0)
    return max(estimates, key=lambda e: e.score)


@dataclass(frozen=True, slots=True)
class Refiner:
    """Coarse-to-fine point-to-plane ICP of the scene segment onto the complete model.

    Registering the partial scene onto the full model keeps every correspondence genuine under
    occlusion (each visible point has a model counterpart; back faces are never forced to match).
    """

    voxel_ratio: float = 0.02  # of the object diameter
    scales: tuple[float, ...] = (3.0, 1.5, 0.75)  # correspondence distances, in voxels
    iterations: int = 30

    def prepare(self, model: ModelCloud, scene: FloatArray) -> tuple[Any, Any, float]:
        voxel = self.voxel_ratio * model.diameter
        model_pcd = to_open3d(model.points, model.normals).voxel_down_sample(voxel)  # type: ignore[attr-defined]
        scene_pcd = to_open3d(scene).voxel_down_sample(voxel)  # type: ignore[attr-defined]
        if len(scene_pcd.points) >= 3:
            import open3d as o3d

            scene_pcd.normals = o3d.utility.Vector3dVector(
                estimate_normals(np.asarray(scene_pcd.points), radius=3 * voxel)
            )
        return model_pcd, scene_pcd, voxel

    def refine(self, prepared: tuple[Any, Any, float], model_to_cam: FloatArray) -> Estimate:
        import open3d as o3d

        reg = o3d.pipelines.registration
        model_pcd, scene_pcd, voxel = prepared
        if len(scene_pcd.points) < 3:
            return Estimate(Pose.from_matrix(model_to_cam), 0.0, 0.0)
        T = np.linalg.inv(model_to_cam)  # scene (camera) -> model
        fitness = 0.0
        for scale in self.scales:
            result = reg.registration_icp(
                scene_pcd,
                model_pcd,
                scale * voxel,
                T,
                reg.TransformationEstimationPointToPlane(),
                reg.ICPConvergenceCriteria(max_iteration=self.iterations),
            )
            T, fitness = np.asarray(result.transformation, dtype=float), float(result.fitness)
        return Estimate(Pose.from_matrix(np.linalg.inv(T)), fitness, 0.0)


def create_estimator(name: str, **kwargs: Any) -> PoseEstimator:
    if name == "fpfh":
        from posegrasp.estimators.fpfh import FpfhEstimator

        return FpfhEstimator(**kwargs)
    if name == "ppf":
        from posegrasp.estimators.ppf import PpfEstimator

        return PpfEstimator(**kwargs)
    raise ValueError(f"unknown estimator '{name}' (fpfh | ppf)")
