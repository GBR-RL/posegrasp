"""Grasp synthesis, judging and planning on the synthetic blob."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from posegrasp.grasp import (
    GraspSet,
    Gripper,
    Planner,
    box_distance,
    close_on,
    collides,
    judge,
    synthesize,
)
from synthetic import SyntheticObject, pose_at, render_view

GRIPPER = Gripper()


@pytest.fixture(scope="module")
def grasps(blob: SyntheticObject) -> GraspSet:
    return synthesize(blob.mesh_path, samples=300, approaches=6, max_grasps=300)


@pytest.fixture(scope="module")
def surface(blob: SyntheticObject) -> np.ndarray:
    import trimesh

    mesh = trimesh.load(blob.mesh_path)
    return np.asarray(mesh.sample(3000, seed=0), dtype=float)


def test_box_distance_is_signed() -> None:
    centres = np.zeros((1, 3))
    halves = np.array([[1.0, 2.0, 3.0]])
    points = np.array([[0.0, 0, 0], [0, 0, 5], [4, 6, 3]])
    d = box_distance(points, centres, halves)[:, 0]
    assert d[0] == pytest.approx(-1.0)  # inside: minus the distance to the nearest face
    assert d[1] == pytest.approx(2.0)
    assert d[2] == pytest.approx(5.0)  # from the corner (1, 2, 3): (3, 4, 0)


def test_open_gripper_volume() -> None:
    grasp = np.eye(4)
    between_the_fingers = np.array([[0.0, 30.0, 0.0]])
    in_a_finger = np.array([[0.0, GRIPPER.max_width / 2 + 3, 0.0]])
    in_the_palm = np.array([[0.0, 0.0, GRIPPER.pad_offset - GRIPPER.finger_length - 10]])
    beyond_the_tips = np.array([[0.0, 45.0, GRIPPER.pad_offset + 1]])
    assert not collides(grasp, between_the_fingers, GRIPPER)
    assert collides(grasp, in_a_finger, GRIPPER)
    assert collides(grasp, in_the_palm, GRIPPER)
    assert not collides(grasp, beyond_the_tips, GRIPPER)
    assert not collides(grasp, np.repeat(in_a_finger, 3, axis=0), GRIPPER, tolerance=3)


def test_synthesised_grasps_are_valid_on_the_model(
    blob: SyntheticObject, grasps: GraspSet, surface: np.ndarray
) -> None:
    assert len(grasps) >= 50
    assert np.allclose(np.linalg.det(grasps.poses[:, :3, :3]), 1.0)
    assert (grasps.widths <= GRIPPER.max_width - 2 * GRIPPER.clearance).all()
    assert ((grasps.quality >= 0) & (grasps.quality <= 1)).all()
    for grasp in grasps.poses[:20]:
        outcome = judge(
            grasp,
            np.eye(4),
            mesh_path=blob.mesh_path,
            surface=surface,
            obstacles=np.empty((0, 3)),
            gripper=GRIPPER,
        )
        assert outcome.success, outcome.reason
        assert 0 < outcome.width < GRIPPER.max_width


def test_save_and_load(tmp_path: Path, grasps: GraspSet) -> None:
    path = tmp_path / "g.npz"
    grasps.save(path)
    loaded = GraspSet.load(path)
    assert np.allclose(loaded.poses, grasps.poses)
    assert len(loaded) == len(grasps)


def test_pose_error_turns_a_grasp_into_a_failure(
    blob: SyntheticObject, grasps: GraspSet, surface: np.ndarray
) -> None:
    true_pose = pose_at((0.3, -0.5, 0.2), (30, -20, 700)).matrix()
    grasp_cam = true_pose @ grasps.poses[0]
    ok = judge(
        grasp_cam,
        true_pose,
        mesh_path=blob.mesh_path,
        surface=surface,
        obstacles=np.empty((0, 3)),
        gripper=GRIPPER,
    )
    assert ok.success
    # planned on an estimate 60 mm off along the approach: the fingers close on air
    wrong = true_pose.copy()
    wrong[:3, 3] -= 60 * grasp_cam[:3, 2]
    miss = judge(
        wrong @ grasps.poses[0],
        true_pose,
        mesh_path=blob.mesh_path,
        surface=surface,
        obstacles=np.empty((0, 3)),
        gripper=GRIPPER,
    )
    assert not miss.success
    assert miss.reason in {"no_contact", "slip", "collision_object"}


def test_closing_on_a_cube_across_faces_holds_and_along_a_diagonal_slips(tmp_path: Path) -> None:
    import trimesh

    path = tmp_path / "cube.ply"
    trimesh.creation.box(extents=(40, 40, 40)).export(path)
    across = np.eye(4)  # top-down approach, closing along y onto two opposite faces
    held = close_on(across, path, GRIPPER)
    assert held.success
    assert held.width == pytest.approx(40.0, abs=1e-3)
    diagonal = np.eye(4)
    diagonal[:3, :3] = Rotation.from_rotvec([0.0, 0.0, np.pi / 4]).as_matrix()
    # the contact normals are 45 degrees off the closing axis, outside the 26.6-degree cone
    assert close_on(diagonal, path, GRIPPER).reason == "slip"


def test_planner_finds_a_camera_facing_grasp_that_works(
    blob: SyntheticObject, grasps: GraspSet, surface: np.ndarray
) -> None:
    true = pose_at((0.3, -0.5, 0.2), (30, -20, 700))
    view = render_view(blob, true)
    from posegrasp.geometry import backproject
    from synthetic import K_LMO

    scene = backproject(view.depth, view.depth > 0, K_LMO)[::7]  # object and background plane
    plan = Planner().plan(
        grasps, true.matrix(), surface=surface, scene=scene, diameter=blob.diameter
    )
    assert plan.grasp is not None
    assert plan.feasible <= plan.candidates <= len(grasps)
    approach = plan.grasp[:3, 2]
    ray = plan.grasp[:3, 3] / np.linalg.norm(plan.grasp[:3, 3])
    assert approach @ ray >= np.cos(np.radians(60)) - 1e-9
    obstacles = scene[~view.mask[view.depth > 0][::7]]
    outcome = judge(
        plan.grasp,
        true.matrix(),
        mesh_path=blob.mesh_path,
        surface=surface,
        obstacles=obstacles,
        gripper=GRIPPER,
    )
    assert outcome.success, outcome.reason
