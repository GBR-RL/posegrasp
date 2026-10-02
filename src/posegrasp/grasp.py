"""Parallel-jaw grasps: antipodal synthesis on the CAD model, planning in the scene from an
estimated pose, and judging a planned grasp against the true pose.

Grasp frame (as the Franka hand's TCP): the origin lies midway between the finger pad centres,
y is the closing axis, z the approach direction (from the palm towards the object), x = y cross z.
All lengths are millimetres.

A grasp *succeeds* on an object at a pose when (docs/EVAL_PROTOCOL.md, "Grasp success"):
1. the open gripper collides neither with the object nor with the rest of the scene,
2. closing the fingers along y makes both pads touch the object, and
3. both contact normals lie inside the friction cone around the closing axis.
The same test validates the synthesised grasps on the model, so a grasp planned from the true pose
fails only through the scene.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from posegrasp.render import cast_rays

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class Gripper:
    """The Franka Hand as three boxes: two fingers and the palm."""

    max_width: float = 80.0  # stroke: opening between the finger pads
    finger_length: float = 50.0  # z extent, fingertip to palm
    finger_width: float = 20.0  # x extent
    finger_thickness: float = 12.0  # y extent
    pad_offset: float = 9.0  # pad centre (the contact) behind the fingertip, along z
    pad_size: float = 18.0  # square pad, x and z extent
    palm: tuple[float, float, float] = (60.0, 200.0, 70.0)  # x, y, z extents
    friction: float = 0.5  # Coulomb coefficient; cone half-angle atan(0.5) = 26.6 degrees
    clearance: float = 5.0  # free opening left beside the object on each side

    @property
    def cone_cos(self) -> float:
        return math.cos(math.atan(self.friction))

    def boxes(self, opening: float | None = None) -> tuple[FloatArray, FloatArray]:
        """Centres and half extents (3, 3) of the -y finger, the +y finger and the palm."""
        opening = self.max_width if opening is None else opening
        finger_z = self.pad_offset - self.finger_length / 2
        finger_y = opening / 2 + self.finger_thickness / 2
        palm_z = self.pad_offset - self.finger_length - self.palm[2] / 2
        centres = np.array([[0.0, -finger_y, finger_z], [0.0, finger_y, finger_z], [0, 0, palm_z]])
        finger_half = [self.finger_width / 2, self.finger_thickness / 2, self.finger_length / 2]
        halves = np.array([finger_half, finger_half, np.asarray(self.palm) / 2])
        return centres, halves

    def pad_grid(self, steps: int = 3) -> FloatArray:
        """(steps², 2) x, z offsets over a pad, inset from its edges."""
        s = np.linspace(-0.4, 0.4, steps) * self.pad_size
        xs, zs = np.meshgrid(s, s)
        return np.column_stack([xs.ravel(), zs.ravel()])


def box_distance(points: FloatArray, centres: FloatArray, halves: FloatArray) -> FloatArray:
    """Signed distance (N, B) of grasp-frame points to each box; negative inside."""
    q = np.abs(points[:, None, :] - centres[None]) - halves[None]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    out: FloatArray = outside + inside
    return out


def to_frame(points: FloatArray, T: FloatArray) -> FloatArray:
    """Points expressed in the frame whose pose (frame -> world) is T."""
    out: FloatArray = (points - T[:3, 3]) @ T[:3, :3]
    return out


# ---- judging a grasp on an object --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Outcome:
    success: bool
    reason: str  # "ok", "collision_object", "collision_scene", "no_contact", "slip"
    width: float = math.nan  # opening after closing, mm


def close_on_many(grasps: FloatArray, mesh_path: Path, gripper: Gripper) -> list[Outcome]:
    """Closes the fingers of each grasp (K, 4, 4; grasp -> model frame) on the mesh, by ray
    casting from the pad grid of each open finger along the closing axis. The first hit per
    finger is its contact."""
    grid = gripper.pad_grid()
    n = len(grid)
    half = gripper.max_width / 2
    # finger at -y closes along +y, the one at +y along -y
    local = np.vstack(
        [np.column_stack([grid[:, 0], np.full(n, s * half), grid[:, 1]]) for s in (-1.0, 1.0)]
    )
    directions = np.vstack([np.tile([0.0, -s, 0.0], (n, 1)) for s in (-1.0, 1.0)])
    R, t = grasps[:, :3, :3], grasps[:, :3, 3]
    origins = np.einsum("kij,nj->kni", R, local) + t[:, None, :]
    dirs = np.einsum("kij,nj->kni", R, directions)
    t_hit, hit_normals = cast_rays(mesh_path, np.concatenate([origins, dirs], -1).reshape(-1, 6))
    distance = t_hit.reshape(len(grasps), 2, n)
    normals = hit_normals.reshape(len(grasps), 2, n, 3)
    first = distance.argmin(axis=2)
    d = np.take_along_axis(distance, first[..., None], axis=2)[..., 0]
    contact_normals = np.take_along_axis(normals, first[..., None, None], axis=2)[:, :, 0]
    y = R[:, :, 1]
    # outward normals must face their finger: -y at the -y finger, +y at the other
    cos_low = -np.einsum("ki,ki->k", contact_normals[:, 0], y)
    cos_high = np.einsum("ki,ki->k", contact_normals[:, 1], y)
    outcomes = []
    for k in range(len(grasps)):
        width = gripper.max_width - float(d[k].sum())
        if not np.isfinite(d[k]).all() or width <= 0:
            outcomes.append(Outcome(False, "no_contact"))
        elif min(cos_low[k], cos_high[k]) < gripper.cone_cos:
            outcomes.append(Outcome(False, "slip", width))
        else:
            outcomes.append(Outcome(True, "ok", width))
    return outcomes


def close_on(grasp: FloatArray, mesh_path: Path, gripper: Gripper) -> Outcome:
    return close_on_many(grasp[None], mesh_path, gripper)[0]


def collides_many(
    grasps: FloatArray, points: FloatArray, gripper: Gripper, *, tolerance: int = 0
) -> BoolArray:
    """For each grasp (K, 4, 4), whether more than `tolerance` points (in the grasps' parent
    frame) lie inside the open gripper."""
    if len(points) == 0:
        return np.zeros(len(grasps), dtype=bool)
    centres, halves = gripper.boxes()
    R, t = grasps[:, :3, :3], grasps[:, :3, 3]
    local = (points[None] - t[:, None, :]) @ R  # (K, N, 3) in each grasp frame
    inside = np.zeros(local.shape[:2], dtype=bool)
    for centre, half in zip(centres, halves, strict=True):
        inside |= (np.abs(local - centre) < half).all(axis=-1)
    out: BoolArray = inside.sum(axis=1) > tolerance
    return out


def collides(
    grasp: FloatArray, points: FloatArray, gripper: Gripper, *, tolerance: int = 0
) -> bool:
    """Whether more than `tolerance` points (in the grasp's parent frame) lie inside the open
    gripper."""
    return bool(collides_many(grasp[None], points, gripper, tolerance=tolerance)[0])


def judge(
    grasp_cam: FloatArray,
    true_pose: FloatArray,
    *,
    mesh_path: Path,
    surface: FloatArray,
    obstacles: FloatArray,
    gripper: Gripper,
) -> Outcome:
    """Grasp (grasp -> camera) judged on the object at its true pose (model -> camera).

    surface: model-frame samples of the object surface; obstacles: camera-frame points of the rest
    of the scene. Three obstacle points inside the gripper are tolerated as depth noise.
    """
    grasp_model = np.linalg.inv(true_pose) @ grasp_cam
    if collides(grasp_model, surface, gripper):
        return Outcome(False, "collision_object")
    if collides(grasp_cam, obstacles, gripper, tolerance=3):
        return Outcome(False, "collision_scene")
    return close_on(grasp_model, mesh_path, gripper)


# ---- synthesis on the model ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GraspSet:
    poses: FloatArray  # (M, 4, 4) grasp -> model frame
    widths: FloatArray  # (M,) object width between the contacts
    quality: FloatArray  # (M,) friction margin: 1 at the cone axis, 0 at its border
    robustness: FloatArray | None = None  # (M,) success rate under small pose errors

    def __len__(self) -> int:
        return len(self.poses)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        robustness = np.full(len(self), np.nan) if self.robustness is None else self.robustness
        np.savez_compressed(
            path, poses=self.poses, widths=self.widths, quality=self.quality, robustness=robustness
        )

    @staticmethod
    def load(path: Path) -> GraspSet:
        with np.load(path) as data:
            robustness = data["robustness"] if "robustness" in data.files else None
            if robustness is not None and np.isnan(robustness).all():
                robustness = None
            return GraspSet(data["poses"], data["widths"], data["quality"], robustness)


def perturbations(
    n: int, *, translation: float = 3.0, rotation_deg: float = 3.0, seed: int = 0
) -> FloatArray:
    """n small rigid motions (4x4): Gaussian translation (mm) and rotation (degrees) per axis."""
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    T = np.tile(np.eye(4), (n, 1, 1))
    rotvecs = np.deg2rad(rotation_deg) * rng.normal(size=(n, 3))
    T[:, :3, :3] = Rotation.from_rotvec(rotvecs).as_matrix()
    T[:, :3, 3] = translation * rng.normal(size=(n, 3))
    return T


def robustness(
    poses: FloatArray,
    mesh_path: Path,
    surface: FloatArray,
    gripper: Gripper,
    *,
    samples: int = 16,
    translation: float = 3.0,
    rotation_deg: float = 3.0,
    seed: int = 0,
) -> FloatArray:
    """Share of small object pose errors under which each grasp (grasp -> model) still succeeds.

    The object is moved by `samples` rigid motions about its origin while the grasp stays where it
    was planned: the situation of a grasp planned from a slightly wrong pose estimate.
    """
    motions = perturbations(samples, translation=translation, rotation_deg=rotation_deg, seed=seed)
    deltas = np.linalg.inv(motions)
    out = np.zeros(len(poses))
    for i, grasp in enumerate(poses):
        moved = deltas @ grasp  # the grasp relative to each moved object
        free = moved[~collides_many(moved, surface, gripper)]
        held = sum(o.success for o in close_on_many(free, mesh_path, gripper)) if len(free) else 0
        out[i] = held / samples
    return out


def _frames(centres: FloatArray, axes: FloatArray, approaches: int) -> FloatArray:
    """Grasp frames (N * approaches, 4, 4) around each closing axis, approach every 360/k deg."""
    y = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    helper = np.where(np.abs(y[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    u = np.cross(y, helper)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.cross(y, u)
    frames = []
    for k in range(approaches):
        a = 2 * math.pi * k / approaches
        z = math.cos(a) * u + math.sin(a) * w
        T = np.tile(np.eye(4), (len(y), 1, 1))
        T[:, :3, 0], T[:, :3, 1], T[:, :3, 2], T[:, :3, 3] = np.cross(y, z), y, z, centres
        frames.append(T)
    return np.stack(frames, axis=1).reshape(-1, 4, 4)


def synthesize(
    mesh_path: Path,
    *,
    gripper: Gripper | None = None,
    samples: int = 1500,
    approaches: int = 12,
    max_grasps: int = 3000,
    seed: int = 0,
    surface: FloatArray | None = None,
    robustness_samples: int = 16,
) -> GraspSet:
    """Antipodal grasps on a CAD model.

    From surface samples, a ray into the object along the inward normal finds the opposite wall;
    pairs that fit the stroke with clearance and lie inside the friction cone get `approaches`
    approach directions around their closing axis. A candidate is kept when the same test that
    judges planned grasps (`judge` without a scene) succeeds on the model; pass the `surface`
    samples the judge will use, so that a grasp planned from the true pose never collides with
    the object. Each kept grasp is then scored for robustness to small pose errors.
    """
    import open3d as o3d

    gripper = gripper or Gripper()
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_triangle_normals()
    o3d.utility.random.seed(seed)  # the sampling draws from Open3D's global generator
    pcd = mesh.sample_points_poisson_disk(samples, init_factor=3, use_triangle_normal=True)
    points = np.asarray(pcd.points, dtype=float)
    normals = np.asarray(pcd.normals, dtype=float)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    origins = points - 0.5 * normals  # just under the surface
    t_hit, far_normals = cast_rays(mesh_path, np.hstack([origins, -normals]))
    hit = np.isfinite(t_hit)
    far = origins - np.where(hit, t_hit, 0.0)[:, None] * normals
    width = np.linalg.norm(far - points, axis=1)
    ok = hit & (width >= 2.0) & (width <= gripper.max_width - 2 * gripper.clearance)
    axis = (far - points) / np.maximum(width, 1e-9)[:, None]
    cos_near = np.einsum("ij,ij->i", -normals, axis)
    cos_far = np.einsum("ij,ij->i", far_normals, axis)
    ok &= np.minimum(cos_near, cos_far) >= gripper.cone_cos
    centres = (points + far) / 2
    margin = (np.minimum(cos_near, cos_far) - gripper.cone_cos) / (1 - gripper.cone_cos)
    frames = _frames(centres[ok], axis[ok], approaches)
    widths = np.repeat(width[ok], approaches)
    quality = np.repeat(np.clip(margin[ok], 0, 1), approaches)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frames))
    if surface is None:
        surface = np.asarray(mesh.sample_points_uniformly(4000).points, dtype=float)
    keep: list[int] = []
    for chunk in np.array_split(order, max(1, len(order) // 64)):
        if len(keep) >= max_grasps:
            break
        free = chunk[~collides_many(frames[chunk], surface, gripper)]
        if len(free):
            held = close_on_many(frames[free], mesh_path, gripper)
            keep += [int(i) for i, o in zip(free, held, strict=True) if o.success]
    idx = np.asarray(keep[:max_grasps], dtype=int)
    robust = None
    if robustness_samples:
        robust = robustness(
            frames[idx], mesh_path, surface, gripper, samples=robustness_samples, seed=seed
        )
    return GraspSet(frames[idx], widths[idx], quality[idx], robust)


# ---- planning in the scene ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Plan:
    grasp: FloatArray | None  # grasp -> camera; None: no feasible grasp
    candidates: int  # grasps facing the camera
    feasible: int  # of those, collision-free in the scene
    score: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    ranked: FloatArray = field(default_factory=lambda: np.empty((0, 4, 4)))  # best first
    widths: FloatArray = field(default_factory=lambda: np.empty(0))  # of the ranked grasps


@dataclass(frozen=True, slots=True)
class Planner:
    gripper: Gripper = field(default_factory=Gripper)
    max_approach_angle: float = 60.0  # degrees between approach and the camera ray
    surface_margin: float = 8.0  # scene points this close to the posed model belong to it
    reach: float = 160.0  # obstacles further than this beyond the object radius are ignored
    keep: int = 10  # feasible grasps returned in rank order (a robot tries them in turn)
    ranking: str = "robust"  # "robust" | "nominal"

    def plan(
        self,
        grasps: GraspSet,
        pose: FloatArray,
        *,
        surface: FloatArray,
        scene: FloatArray,
        diameter: float,
    ) -> Plan:
        """Best grasp on the object at `pose` (model -> camera) in the scene.

        surface: model-frame surface samples; scene: camera-frame points of the whole depth image.
        Scene points away from the posed surface are obstacles. Grasps must approach from the
        camera's side. Ranking "robust" orders the feasible grasps by robustness to pose errors,
        approach angle and clearance to the obstacles (weights 0.6, 0.2, 0.2); "nominal" by
        approach angle, clearance and friction margin (0.5, 0.3, 0.2), ignoring pose errors.
        """
        if self.ranking not in {"robust", "nominal"}:
            raise ValueError(f"unknown ranking '{self.ranking}' (robust | nominal)")
        if len(grasps) == 0:
            return Plan(None, 0, 0)
        frames = pose @ grasps.poses
        centre = pose[:3, 3]
        ray = frames[:, :3, 3] / np.linalg.norm(frames[:, :3, 3], axis=1, keepdims=True)
        alignment = np.einsum("ij,ij->i", frames[:, :3, 2], ray)
        facing = np.flatnonzero(alignment >= math.cos(math.radians(self.max_approach_angle)))
        if len(facing) == 0:
            return Plan(None, 0, 0)
        near = scene[np.linalg.norm(scene - centre, axis=1) < diameter / 2 + self.reach]
        posed_surface = surface @ pose[:3, :3].T + pose[:3, 3]
        dist, _ = cKDTree(posed_surface).query(near, k=1)
        obstacles = near[dist > self.surface_margin]
        centres, halves = self.gripper.boxes()
        clearance = np.full(len(facing), 50.0)
        free = np.ones(len(facing), dtype=bool)
        if len(obstacles):
            for j, i in enumerate(facing):
                d = box_distance(to_frame(obstacles, frames[i]), centres, halves)
                free[j] = int((d < 0).any(axis=1).sum()) <= 3
                clearance[j] = min(50.0, float(d.min()))
        if not free.any():
            return Plan(None, len(facing), 0)
        free_space = np.clip(clearance, 0, 20) / 20
        if self.ranking == "robust" and grasps.robustness is not None:
            score = 0.6 * grasps.robustness[facing] + 0.2 * alignment[facing] + 0.2 * free_space
        else:
            score = 0.5 * alignment[facing] + 0.3 * free_space + 0.2 * grasps.quality[facing]
        score = np.where(free, score, -np.inf)
        order = np.argsort(-score, kind="stable")[: min(self.keep, int(free.sum()))]
        best = int(order[0])
        return Plan(
            frames[facing[best]],
            len(facing),
            int(free.sum()),
            float(score[best]),
            {"width": float(grasps.widths[facing[best]]), "clearance": float(clearance[best])},
            frames[facing[order]],
            grasps.widths[facing[order]],
        )
