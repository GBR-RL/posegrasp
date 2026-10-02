"""Point clouds from depth, model sampling, and rigid-transform helpers (millimetres)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def backproject(
    depth_mm: FloatArray, mask: NDArray[np.bool_], K: FloatArray, *, max_points: int | None = None
) -> FloatArray:
    """Camera-frame points (N, 3) of the masked pixels with valid depth."""
    v, u = np.nonzero(mask & (depth_mm > 0))
    if max_points is not None and len(u) > max_points:
        keep = np.linspace(0, len(u) - 1, max_points).astype(int)
        u, v = u[keep], v[keep]
    z = depth_mm[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1)


def transform(points: FloatArray, T: FloatArray) -> FloatArray:
    out: FloatArray = points @ T[:3, :3].T + T[:3, 3]
    return out


def project(points: FloatArray, K: FloatArray) -> FloatArray:
    """Pixel coordinates (N, 2) of camera-frame points."""
    uvw = points @ K.T
    out: FloatArray = uvw[:, :2] / uvw[:, 2:3]
    return out


@dataclass(frozen=True, slots=True)
class ModelCloud:
    """Surface samples of a CAD model with normals."""

    points: FloatArray  # (N, 3) mm, uniform surface samples
    normals: FloatArray  # (N, 3) unit, outward
    diameter: float


@lru_cache(maxsize=32)
def load_model(mesh_path: Path, diameter: float, n_samples: int = 6000) -> ModelCloud:
    """Samples the mesh surface evenly (Poisson disk) so point density does not follow the
    triangle layout; normals come from the mesh."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise FileNotFoundError(mesh_path)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(n_samples, init_factor=3)
    # Normals are interpolated from the vertex normals, so they are shorter than 1 where the
    # surface bends between vertices.
    normals = np.asarray(pcd.normals, dtype=float)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return ModelCloud(
        points=np.asarray(pcd.points, dtype=float),
        normals=normals,
        diameter=diameter,
    )


@lru_cache(maxsize=32)
def load_vertices(mesh_path: Path) -> FloatArray:
    """Mesh vertices (mm); for the evaluation models these are what BOP errors are computed on."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if mesh.is_empty():
        raise FileNotFoundError(mesh_path)
    return np.asarray(mesh.vertices, dtype=float)


def to_open3d(points: FloatArray, normals: FloatArray | None = None) -> object:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def estimate_normals(
    points: FloatArray, radius: float, toward: FloatArray | None = None
) -> FloatArray:
    """Normals from local neighbourhoods, oriented toward a viewpoint (the camera: origin)."""
    import open3d as o3d

    pcd = to_open3d(points)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))  # type: ignore[attr-defined]
    pcd.orient_normals_towards_camera_location(  # type: ignore[attr-defined]
        np.zeros(3) if toward is None else toward
    )
    normals: FloatArray = np.asarray(pcd.normals, dtype=np.float64)  # type: ignore[attr-defined]
    return normals


def remove_outliers(
    points: FloatArray, *, neighbours: int = 20, std_ratio: float = 2.0
) -> FloatArray:
    """Statistical outlier removal: drops mixed pixels at depth edges and sensor noise."""
    if len(points) < neighbours + 1:
        return points
    pcd = to_open3d(points)
    _, keep = pcd.remove_statistical_outlier(neighbours, std_ratio)  # type: ignore[attr-defined]
    kept: FloatArray = points[np.asarray(keep, dtype=int)]
    return kept


def voxel_down(points: FloatArray, voxel: float) -> FloatArray:
    pcd = to_open3d(points).voxel_down_sample(voxel)  # type: ignore[attr-defined]
    return np.asarray(pcd.points, dtype=float)
