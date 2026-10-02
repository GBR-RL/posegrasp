"""MuJoCo pick trials on the synthetic blob (needs the `sim` extra and network for the hand)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synthetic import SyntheticObject

pytest.importorskip("mujoco")

from posegrasp.grasp import GraspSet, synthesize
from posegrasp.sim import PickTrial, SceneObject, build_model_xml, fetch_hand


@pytest.fixture(scope="module")
def hand(tmp_path_factory: pytest.TempPathFactory) -> Path:
    try:
        return fetch_hand(tmp_path_factory.mktemp("sim"))
    except OSError as error:  # offline
        pytest.skip(f"MuJoCo Menagerie not reachable: {error}")


@pytest.fixture(scope="module")
def blob_grasps(blob: SyntheticObject) -> GraspSet:
    return synthesize(blob.mesh_path, samples=300, approaches=8, max_grasps=400)


def _scene(blob: SyntheticObject, tmp_path: Path) -> tuple[np.ndarray, Path]:
    import trimesh

    mesh = trimesh.load(blob.mesh_path)
    mesh.apply_scale(0.001)
    part = tmp_path / "blob.obj"
    mesh.export(part)
    pose = np.eye(4)
    pose[:3, 3] = [0.5, 0.0, -mesh.vertices[:, 2].min() + 0.001]  # resting on the table
    return pose, part


def _top_down_grasp(grasps: GraspSet, pose: np.ndarray) -> np.ndarray:
    """The most robust grasp approaching from above, in the world (metres)."""
    approach_down = -grasps.poses[:, 2, 2]  # z axis of the grasp, pointing down
    assert grasps.robustness is not None
    score = grasps.robustness + (approach_down > 0.9)
    G = grasps.poses[int(np.argmax(score))].copy()
    G[:3, 3] /= 1000.0
    return pose @ G


def _trial(blob: SyntheticObject, hand: Path, tmp_path: Path, grasp: np.ndarray) -> PickTrial:
    pose, part = _scene(blob, tmp_path)
    hand_pose = grasp.copy()
    hand_pose[:3, 3] -= grasp[:3, 2] * (0.10 + 0.1034)
    xml = build_model_xml(hand, [SceneObject(1, pose, (part,), free=True)], hand_pose=hand_pose)
    return PickTrial(xml, grasp)


def test_a_planned_top_down_grasp_lifts_the_object(
    blob: SyntheticObject, blob_grasps: GraspSet, hand: Path, tmp_path: Path
) -> None:
    pose, _ = _scene(blob, tmp_path)
    grasp = _top_down_grasp(blob_grasps, pose)
    result = _trial(blob, hand, tmp_path, grasp).run()
    assert result.success, result.reason
    assert result.rise > 0.05


def test_a_grasp_beside_the_object_lifts_nothing(
    blob: SyntheticObject, blob_grasps: GraspSet, hand: Path, tmp_path: Path
) -> None:
    pose, _ = _scene(blob, tmp_path)
    grasp = _top_down_grasp(blob_grasps, pose)
    grasp[:3, 3] += 0.08 * grasp[:3, 0]  # 8 cm off along the pad's width axis
    result = _trial(blob, hand, tmp_path, grasp).run()
    assert not result.success
    assert result.rise < 0.05
