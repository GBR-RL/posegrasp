"""Global registration with FPFH features and RANSAC (Rusu et al., ICRA 2009), via Open3D.

Each restart runs RANSAC over mutual FPFH matches with a different seed and yields one coarse
hypothesis, which the shared ICP refines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from posegrasp.estimators.base import Estimate, FloatArray, Refiner
from posegrasp.geometry import ModelCloud, estimate_normals, to_open3d


@dataclass
class FpfhEstimator:
    name: str = "fpfh"
    voxel_ratio: float = 0.04  # voxel size as a fraction of the object diameter
    feature_radius: float = 5.0  # FPFH radius in voxels
    restarts: int = 1  # RANSAC runs with different seeds, one hypothesis each
    ransac_iterations: int = 100_000
    ransac_confidence: float = 0.999
    refiner: Refiner = field(default_factory=Refiner)
    _models: dict[tuple[int, float], Any] = field(default_factory=dict, repr=False)

    def _features(self, pcd: Any, voxel: float) -> Any:
        import open3d as o3d

        return o3d.pipelines.registration.compute_fpfh_feature(  # type: ignore[call-arg]
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.feature_radius * voxel, max_nn=100),
        )

    def hypotheses(self, obj_id: int, model: ModelCloud, scene: FloatArray) -> list[Estimate]:
        import open3d as o3d

        reg = o3d.pipelines.registration
        start = time.perf_counter()
        voxel = self.voxel_ratio * model.diameter
        key = (obj_id, voxel)
        if key not in self._models:
            pcd = to_open3d(model.points, model.normals).voxel_down_sample(voxel)  # type: ignore[attr-defined]
            self._models[key] = (pcd, self._features(pcd, voxel))
        model_pcd, model_feat = self._models[key]
        scene_pcd = to_open3d(scene).voxel_down_sample(voxel)  # type: ignore[attr-defined]
        if len(scene_pcd.points) < 10:
            return []
        scene_pcd.normals = o3d.utility.Vector3dVector(
            estimate_normals(np.asarray(scene_pcd.points), radius=2.5 * voxel)
        )
        scene_feat = self._features(scene_pcd, voxel)
        prepared = self.refiner.prepare(model, scene)
        out = []
        for seed in range(self.restarts):
            o3d.utility.random.seed(seed)
            coarse = reg.registration_ransac_based_on_feature_matching(
                scene_pcd,
                model_pcd,
                scene_feat,
                model_feat,
                mutual_filter=True,
                max_correspondence_distance=1.5 * voxel,
                estimation_method=reg.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    reg.CorrespondenceCheckerBasedOnDistance(1.5 * voxel),
                ],
                criteria=reg.RANSACConvergenceCriteria(
                    self.ransac_iterations, self.ransac_confidence
                ),
            )
            model_to_cam = np.linalg.inv(np.asarray(coarse.transformation, dtype=float))
            out.append(self.refiner.refine(prepared, model_to_cam))
        seconds = time.perf_counter() - start
        return [Estimate(e.pose, e.score, seconds) for e in out]
