"""Point Pair Features (Drost et al., CVPR 2010) via OpenCV surface_matching.

PPF votes for poses with oriented point pairs hashed from the model; a variant of it was the
strongest method of the first BOP Challenge (Vidal et al., 2018). The stock OpenCV detector is
trained once per object; its best pose clusters are refined by the shared ICP.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from posegrasp.estimators.base import Estimate, FloatArray, Refiner
from posegrasp.geometry import ModelCloud, estimate_normals, to_open3d, voxel_down


@dataclass
class PpfEstimator:
    name: str = "ppf"
    sampling_step: float = 0.04  # model and scene sampling, relative to the model diameter
    distance_step: float = 0.05  # pair-distance quantisation, relative to the diameter
    scene_sample_step: float = 0.2  # fraction of scene points used as reference points
    scene_distance: float = 0.05  # pose clustering distance, relative to the diameter
    hypotheses_kept: int = 5  # best-voted pose clusters refined by ICP
    refiner: Refiner = field(default_factory=Refiner)
    _detectors: dict[int, Any] = field(default_factory=dict, repr=False)

    def _detector(self, obj_id: int, model: ModelCloud) -> Any:
        import cv2

        if obj_id not in self._detectors:
            detector = cv2.ppf_match_3d_PPF3DDetector(  # type: ignore[attr-defined]
                self.sampling_step, self.distance_step
            )
            # Training is quadratic in the model points: sample the surface at the PPF step first
            # (OpenCV's own sampling keeps far more points and makes training take minutes).
            pcd = to_open3d(model.points, model.normals).voxel_down_sample(  # type: ignore[attr-defined]
                self.sampling_step * model.diameter
            )
            normals = np.asarray(pcd.normals, dtype=np.float64)
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            detector.trainModel(np.hstack([np.asarray(pcd.points), normals]).astype(np.float32))
            self._detectors[obj_id] = detector
        return self._detectors[obj_id]

    def hypotheses(self, obj_id: int, model: ModelCloud, scene: FloatArray) -> list[Estimate]:
        start = time.perf_counter()
        detector = self._detector(obj_id, model)
        voxel = self.sampling_step * model.diameter
        points = voxel_down(scene, voxel)
        if len(points) < 10:
            return []
        normals = estimate_normals(points, radius=2.5 * voxel)
        scene_pc = np.hstack([points, normals]).astype(np.float32)
        clusters = list(detector.match(scene_pc, self.scene_sample_step, self.scene_distance))
        prepared = self.refiner.prepare(model, scene)
        out = [
            self.refiner.refine(prepared, np.asarray(c.pose, dtype=float))
            for c in clusters[: self.hypotheses_kept]
        ]
        seconds = time.perf_counter() - start
        return [Estimate(e.pose, e.score, seconds) for e in out]
