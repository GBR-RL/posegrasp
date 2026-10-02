"""Estimators and verification on synthetic views with a known pose (no dataset needed)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from posegrasp import metrics
from posegrasp.data.bop import Pose
from posegrasp.estimators import create_estimator
from posegrasp.estimators.base import Estimate, Refiner, best
from posegrasp.geometry import load_model, remove_outliers
from posegrasp.verify import DepthVerifier
from synthetic import K_LMO, SyntheticObject, pose_at, render_view

POSES = [
    pose_at((0.3, -0.5, 0.2), (30, -20, 700)),
    pose_at((-1.2, 0.4, 2.0), (-60, 40, 900)),
]


def _add_ratio(est: Pose | None, gt: Pose, obj: SyntheticObject) -> float:
    assert est is not None
    return metrics.add(est, gt, obj.vertices) / obj.diameter


@pytest.mark.parametrize("name", ["fpfh", "ppf"])
@pytest.mark.parametrize("pose_index", [0, 1])
def test_estimator_recovers_a_known_pose(blob: SyntheticObject, name: str, pose_index: int) -> None:
    gt = POSES[pose_index]
    view = render_view(blob, gt, noise_mm=0.5)
    model = load_model(blob.mesh_path, blob.diameter)
    # a coarser PPF model than the benchmark's keeps training to seconds
    kwargs = {"restarts": 2} if name == "fpfh" else {"sampling_step": 0.06}
    estimator = create_estimator(name, **kwargs)
    hyps = estimator.hypotheses(1, model, remove_outliers(view.points))
    assert hyps
    assert all(h.seconds > 0 for h in hyps)
    assert _add_ratio(best(hyps).pose, gt, blob) < 0.05


def test_refiner_converges_from_a_nearby_pose(blob: SyntheticObject) -> None:
    gt = POSES[0]
    view = render_view(blob, gt, noise_mm=0.5)
    model = load_model(blob.mesh_path, blob.diameter)
    refiner = Refiner()
    prepared = refiner.prepare(model, view.points)
    start = gt.matrix()
    start[:3, :3] = start[:3, :3] @ Rotation.from_rotvec([0.0, 0.12, 0.05]).as_matrix()
    start[:3, 3] += [6.0, -4.0, 5.0]
    refined = refiner.refine(prepared, start)
    assert _add_ratio(Pose.from_matrix(start), gt, blob) > 0.05
    assert _add_ratio(refined.pose, gt, blob) < 0.02
    assert refined.score > 0.8  # ICP inlier fraction


def test_depth_verifier_prefers_the_true_pose(blob: SyntheticObject) -> None:
    gt = POSES[0]
    view = render_view(blob, gt)
    verifier = DepthVerifier()

    def score(pose: Pose) -> float:
        return verifier.score(
            blob.mesh_path,
            blob.diameter,
            pose,
            depth_mm=view.depth,
            mask=view.mask,
            K=K_LMO,
        )

    flipped = Pose(gt.R @ Rotation.from_rotvec([np.pi, 0, 0]).as_matrix(), gt.t)
    shifted = Pose(gt.R, gt.t + np.array([0.0, 0.0, 30.0]))
    assert score(gt) > 0.95
    assert score(gt) > score(flipped) + 0.2
    assert score(gt) > score(shifted) + 0.2
    assert score(Pose(gt.R, gt.t * np.array([1, 1, -1]))) == 0.0  # behind the camera


def test_best_and_unknown_estimator() -> None:
    pose = Pose(np.eye(3), np.zeros(3))
    assert best([]).pose is None
    assert best([Estimate(pose, 0.2, 0.1), Estimate(pose, 0.7, 0.1)]).score == 0.7
    with pytest.raises(ValueError, match="unknown estimator"):
        create_estimator("nope")
