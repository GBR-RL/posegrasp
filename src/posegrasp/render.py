"""Depth rendering of a posed CAD mesh by ray casting (Open3D, CPU).

Pixel (u, v) is the ray through the image point (u, v), so integer coordinates are pixel centres:
the convention of the BOP toolkit and of `geometry.backproject`.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from posegrasp.data.bop import Pose

FloatArray = NDArray[np.float64]
Roi = tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)


@lru_cache(maxsize=32)
def raycasting_scene(mesh_path: Path) -> Any:
    import open3d as o3d

    mesh = o3d.t.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise FileNotFoundError(mesh_path)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene


def render_depth(mesh_path: Path, pose: Pose, K: FloatArray, roi: Roi) -> FloatArray:
    """Depth (mm) of the posed mesh over the pixel window `roi`; 0 where no surface is hit."""
    import open3d as o3d

    x0, y0, x1, y1 = roi
    if x1 <= x0 or y1 <= y0:
        return np.zeros((max(0, y1 - y0), max(0, x1 - x0)))
    u, v = np.meshgrid(np.arange(x0, x1, dtype=float), np.arange(y0, y1, dtype=float))
    rays_cam = np.stack(
        [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1
    )
    # Cast in the model frame: origin = camera centre, directions rotated by R^T.
    origin = -pose.R.T @ pose.t
    dirs = rays_cam.reshape(-1, 3) @ pose.R  # (R^T d)^T = d^T R
    rays = np.hstack([np.broadcast_to(origin, dirs.shape), dirs]).astype(np.float32)
    hits = raycasting_scene(mesh_path).cast_rays(o3d.core.Tensor(rays))  # type: ignore[call-overload]
    t_hit = hits["t_hit"].numpy().reshape(u.shape).astype(np.float64)
    # The ray's z-component is 1, so the hit distance along it is the depth.
    depth: FloatArray = np.where(np.isfinite(t_hit) & (t_hit > 0), t_hit, 0.0)
    return depth


def depth_renderer(mesh_path: Path, K: FloatArray) -> Callable[[Pose, Roi], FloatArray]:
    """render_depth bound to one mesh and camera."""

    def render(pose: Pose, roi: Roi) -> FloatArray:
        return render_depth(mesh_path, pose, K, roi)

    return render


def projected_roi(vertices: FloatArray, pose: Pose, K: FloatArray, width: int, height: int) -> Roi:
    """Pixel window that contains the projection of the posed mesh (the whole image when part
    of the mesh is behind the camera)."""
    pts = vertices @ pose.R.T + pose.t
    if (pts[:, 2] <= 1e-6).any():
        return 0, 0, width, height
    uv = pts[:, :2] / pts[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    x0, y0 = np.floor(uv.min(axis=0)).astype(int)
    x1, y1 = np.ceil(uv.max(axis=0)).astype(int) + 1
    return (
        int(np.clip(x0, 0, width)),
        int(np.clip(y0, 0, height)),
        int(np.clip(x1, 0, width)),
        int(np.clip(y1, 0, height)),
    )
