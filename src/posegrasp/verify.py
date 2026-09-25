"""Hypothesis verification: which candidate pose explains the observed depth best?

The model is rendered at each candidate pose by ray casting the CAD mesh (Open3D, CPU) over the
pixels around the detection, and compared with the measured depth:

- explained: segmented object pixels where rendered and measured depth agree within `tau`
- violation: pixels where the model would be in front of free space (measured depth is behind the
  rendered surface by more than `tau`), i.e. the camera sees through the hypothesised object
- occlusion (measured depth in front of the model) is not penalised: other objects may cover it

score = explained fraction of the segment x (1 - violation fraction of the rendered silhouette)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from posegrasp.data.bop import Pose

FloatArray = NDArray[np.float64]


@lru_cache(maxsize=32)
def _raycasting_scene(mesh_path: Path) -> Any:
    import open3d as o3d

    mesh = o3d.t.io.read_triangle_mesh(str(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def render_depth(
    mesh_path: Path, pose: Pose, K: FloatArray, roi: tuple[int, int, int, int]
) -> FloatArray:
    """Depth (mm) of the posed mesh over the pixel window roi = (x0, y0, x1, y1); 0 = no hit."""
    import open3d as o3d

    x0, y0, x1, y1 = roi
    u, v = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    rays_cam = np.stack(
        [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1
    )
    # Cast in the model frame: origin = camera centre, directions rotated by R^T.
    origin = -pose.R.T @ pose.t
    dirs = rays_cam.reshape(-1, 3) @ pose.R  # (R^T d)^T = d^T R
    rays = np.hstack([np.broadcast_to(origin, dirs.shape), dirs]).astype(np.float32)
    hits = _raycasting_scene(mesh_path).cast_rays(o3d.core.Tensor(rays))  # type: ignore[call-overload]
    t_hit = hits["t_hit"].numpy().reshape(u.shape).astype(np.float64)
    depth: FloatArray = np.where(np.isfinite(t_hit), t_hit, 0.0)  # z = t since ray z-component = 1
    return depth


@dataclass(frozen=True, slots=True)
class DepthVerifier:
    tau_ratio: float = 0.05  # depth agreement, as a fraction of the object diameter
    margin: int = 16  # pixels around the segment's box

    def score(
        self,
        mesh_path: Path,
        diameter: float,
        pose: Pose,
        *,
        depth_mm: FloatArray,
        mask: NDArray[np.bool_],
        K: FloatArray,
    ) -> float:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0 or pose.t[2] <= 0:
            return 0.0
        h, w = depth_mm.shape
        roi = (
            max(0, int(xs.min()) - self.margin),
            max(0, int(ys.min()) - self.margin),
            min(w, int(xs.max()) + 1 + self.margin),
            min(h, int(ys.max()) + 1 + self.margin),
        )
        x0, y0, x1, y1 = roi
        rendered = render_depth(mesh_path, pose, K, roi)
        observed = depth_mm[y0:y1, x0:x1]
        segment = mask[y0:y1, x0:x1] & (observed > 0)
        tau = self.tau_ratio * diameter
        valid = (rendered > 0) & (observed > 0)
        agree = valid & (np.abs(rendered - observed) < tau)
        violation = valid & (observed > rendered + tau)
        explained = (agree & segment).sum() / max(1, segment.sum())
        silhouette = valid.sum()
        free_space = violation.sum() / silhouette if silhouette else 1.0
        return float(explained * (1.0 - free_space))
