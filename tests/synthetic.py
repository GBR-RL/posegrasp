"""Synthetic test data: an irregular object mesh, a camera and rendered views (no dataset)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation

from posegrasp.data.bop import Pose
from posegrasp.geometry import FloatArray, backproject
from posegrasp.render import render_depth

# LM-O (Primesense) intrinsics
K_LMO = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])
WIDTH, HEIGHT = 640, 480


@dataclass(frozen=True)
class SyntheticObject:
    mesh_path: Path
    vertices: FloatArray
    diameter: float


def make_blob_object(folder: Path) -> SyntheticObject:
    """A smooth, asymmetric blob (~130 mm): an ellipsoid with a few bumps of different sizes, so
    local surface shape tells the views apart, as on the real objects."""
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    directions = np.asarray(mesh.vertices, dtype=float)
    rng = np.random.default_rng(7)
    centres = rng.normal(size=(6, 3))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    heights = np.array([0.45, 0.35, 0.3, 0.25, 0.2, -0.15])
    widths = np.array([0.15, 0.25, 0.1, 0.3, 0.12, 0.2])
    sq = ((directions[:, None, :] - centres[None]) ** 2).sum(-1)
    radius = 1.0 + (heights * np.exp(-sq / widths)).sum(axis=1)
    mesh.vertices = directions * radius[:, None] * np.array([50.0, 35.0, 25.0])
    path = folder / "blob.ply"
    mesh.export(path)
    vertices = np.asarray(mesh.vertices, dtype=float)
    return SyntheticObject(path, vertices, float(pdist(vertices).max()))


def pose_at(rotvec: tuple[float, float, float], t: tuple[float, float, float]) -> Pose:
    return Pose(R=Rotation.from_rotvec(rotvec).as_matrix(), t=np.asarray(t, dtype=float))


@dataclass(frozen=True)
class SyntheticView:
    pose: Pose
    depth: FloatArray  # mm, object on a background plane
    mask: np.ndarray  # object pixels
    points: FloatArray  # camera-frame points of the object pixels


def render_view(obj: SyntheticObject, pose: Pose, noise_mm: float = 0.0) -> SyntheticView:
    depth = render_depth(obj.mesh_path, pose, K_LMO, (0, 0, WIDTH, HEIGHT))
    mask = depth > 0
    background = np.full_like(depth, pose.t[2] + obj.diameter)
    scene = np.where(mask, depth, background)
    if noise_mm:
        scene = scene + np.random.default_rng(0).normal(0.0, noise_mm, scene.shape)
    return SyntheticView(pose, scene, mask, backproject(scene, mask, K_LMO))
