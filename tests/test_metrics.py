"""BOP pose errors: unit checks, and agreement with the official bop_toolkit where installed."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from posegrasp import metrics
from posegrasp.data.bop import ObjectModel, Pose
from posegrasp.render import depth_renderer, render_depth
from synthetic import HEIGHT, K_LMO, WIDTH, SyntheticObject, pose_at

SYMMETRY_CASES: dict[str, dict[str, Any]] = {
    "none": {},
    "discrete": {
        # 180 degrees about z through a point off the origin
        "symmetries_discrete": [[-1, 0, 0, 4.0, 0, -1, 0, -2.0, 0, 0, 1, 0, 0, 0, 0, 1]]
    },
    "continuous": {"symmetries_continuous": [{"axis": [0, 0, 1], "offset": [1.0, -2.0, 0]}]},
}


def _model(info: dict[str, Any], diameter: float = 120.0) -> ObjectModel:
    return ObjectModel(
        obj_id=1,
        mesh_path=Path("unused.ply"),
        diameter=diameter,
        symmetries_discrete=tuple(
            np.asarray(s, dtype=float).reshape(4, 4) for s in info.get("symmetries_discrete", [])
        ),
        symmetries_continuous=tuple(info.get("symmetries_continuous", [])),
    )


def _random_pose_pair(rng: np.random.Generator, angle_deg: float) -> tuple[Pose, Pose]:
    gt = Pose(
        R=Rotation.random(random_state=rng).as_matrix(),
        t=np.array([rng.uniform(-80, 80), rng.uniform(-60, 60), rng.uniform(500, 1000)]),
    )
    axis = Rotation.random(random_state=rng).as_rotvec()
    delta = Rotation.from_rotvec(np.deg2rad(angle_deg) * axis / np.linalg.norm(axis))
    est = Pose(R=gt.R @ delta.as_matrix(), t=gt.t + rng.normal(0, 8, 3))
    return est, gt


def test_symmetry_transforms_count_and_identity() -> None:
    assert len(metrics.symmetry_transforms(_model(SYMMETRY_CASES["none"]))) == 1
    assert len(metrics.symmetry_transforms(_model(SYMMETRY_CASES["discrete"]))) == 2
    continuous = metrics.symmetry_transforms(_model(SYMMETRY_CASES["continuous"]))
    assert len(continuous) == math.ceil(math.pi / 0.01)
    assert np.allclose(continuous[0], np.eye(4))
    # every transform maps the symmetry axis (through the offset) onto itself
    for T in continuous[::50]:
        point_on_axis = np.array([1.0, -2.0, 30.0, 1.0])
        assert np.allclose(T @ point_on_axis, point_on_axis)


def test_errors_of_the_true_pose_are_zero_and_symmetry_is_honoured() -> None:
    rng = np.random.default_rng(1)
    vertices = rng.normal(size=(300, 3)) * 40
    _, gt = _random_pose_pair(rng, 0)
    sym = metrics.symmetry_transforms(_model(SYMMETRY_CASES["discrete"]))
    assert metrics.mssd(gt, gt, vertices, sym) == pytest.approx(0, abs=1e-9)
    assert metrics.mspd(gt, gt, vertices, sym, K_LMO) == pytest.approx(0, abs=1e-9)
    flipped = Pose.from_matrix(gt.matrix() @ sym[1])  # the symmetric twin of the true pose
    assert metrics.mssd(flipped, gt, vertices, sym) == pytest.approx(0, abs=1e-9)
    assert metrics.add(flipped, gt, vertices) > 10  # ADD ignores symmetry


def test_summary_recall_and_misses() -> None:
    inf = math.inf
    rows = [
        # correct at every threshold
        {"mssd": 1.0, "mspd": 1.0, "add": 1.0, "adi": 1.0, "symmetric": 0.0, "vsd": [0.0] * 10},
        # wrong at every threshold
        {"mssd": 1e4, "mspd": 1e4, "add": 1e4, "adi": 1e4, "symmetric": 0.0, "vsd": [1.0] * 10},
        # a miss
        {"mssd": inf, "mspd": inf, "add": inf, "adi": inf, "symmetric": 1.0, "vsd": [inf] * 10},
        # MSSD 0.12 d: below 8 of the 10 thresholds 0.05..0.5 (0.15 and up)
        {"mssd": 12.0, "mspd": 1e4, "add": 1e4, "adi": 5.0, "symmetric": 1.0, "vsd": [1.0] * 10},
    ]
    s = metrics.summarize(rows, [100.0] * 4)
    assert s["ar_mssd"] == pytest.approx((1 + 0.8) / 4)
    assert s["ar_mspd"] == pytest.approx(1 / 4)
    assert s["ar_vsd"] == pytest.approx(1 / 4)
    assert s["ar"] == pytest.approx((s["ar_vsd"] + s["ar_mssd"] + s["ar_mspd"]) / 3)
    assert s["add_s_01d"] == pytest.approx(2 / 4)  # row 4 is symmetric: ADD-S 5 mm < 10 mm
    rows[0] = {**rows[0], "vsd": None}  # VSD not computed: no AR
    assert math.isnan(metrics.summarize(rows, [100.0] * 4)["ar"])


def test_render_depth_of_a_box_face(tmp_path: Path) -> None:
    import trimesh

    path = tmp_path / "box.ply"
    trimesh.creation.box(extents=(100, 100, 100)).export(path)
    pose = Pose(R=np.eye(3), t=np.array([0.0, 0.0, 600.0]))
    cx, cy = int(K_LMO[0, 2]), int(K_LMO[1, 2])
    depth = render_depth(path, pose, K_LMO, (cx - 5, cy - 5, cx + 5, cy + 5))
    assert np.allclose(depth, 550.0)  # the face towards the camera
    far = render_depth(path, pose, K_LMO, (0, 0, 10, 10))
    assert (far == 0).all()  # image corner: no hit


def test_projected_roi_contains_the_rendering(blob: SyntheticObject) -> None:
    from posegrasp.render import projected_roi

    pose = pose_at((0.3, -0.2, 0.5), (40, -20, 700))
    full = render_depth(blob.mesh_path, pose, K_LMO, (0, 0, WIDTH, HEIGHT))
    x0, y0, x1, y1 = projected_roi(blob.vertices, pose, K_LMO, WIDTH, HEIGHT)
    inside = np.zeros_like(full, dtype=bool)
    inside[y0:y1, x0:x1] = True
    assert (full > 0).sum() > 1000
    assert not ((full > 0) & ~inside).any()


# ---- cross-checks against the official BOP toolkit -------------------------------------------


@pytest.fixture(scope="module")
def bop() -> Any:
    pytest.importorskip("bop_toolkit_lib")
    from bop_toolkit_lib import misc, pose_error

    return pose_error, misc


@pytest.mark.parametrize("case", sorted(SYMMETRY_CASES))
def test_mssd_mspd_add_adi_match_bop_toolkit(bop: Any, case: str) -> None:
    pose_error, misc = bop
    rng = np.random.default_rng(len(case))
    vertices = rng.normal(size=(400, 3)) * np.array([50.0, 30.0, 20.0])
    ours_sym = metrics.symmetry_transforms(_model(SYMMETRY_CASES[case]))
    bop_sym = misc.get_symmetry_transformations(SYMMETRY_CASES[case], max_sym_disc_step=0.01)
    assert len(ours_sym) == len(bop_sym)
    for angle in (2.0, 15.0, 90.0):
        est, gt = _random_pose_pair(rng, angle)
        args = (est.R, est.t.reshape(3, 1), gt.R, gt.t.reshape(3, 1))
        assert metrics.mssd(est, gt, vertices, ours_sym) == pytest.approx(
            pose_error.mssd(*args, vertices, bop_sym), rel=1e-9
        )
        assert metrics.mspd(est, gt, vertices, ours_sym, K_LMO) == pytest.approx(
            pose_error.mspd(*args, K_LMO, vertices, bop_sym), rel=1e-9
        )
        assert metrics.add(est, gt, vertices) == pytest.approx(
            pose_error.add(*args, vertices), rel=1e-9
        )
        assert metrics.adi(est, gt, vertices) == pytest.approx(
            pose_error.adi(*args, vertices), rel=1e-9
        )


class _BopRenderer:
    """bop_toolkit's renderer interface, backed by posegrasp.render (full image)."""

    def __init__(self, mesh_path: Path) -> None:
        self.mesh_path = mesh_path

    def render_object(  # noqa: PLR0917 (the toolkit calls it positionally)
        self, obj_id: int, R: Any, t: Any, fx: float, fy: float, cx: float, cy: float
    ) -> dict[str, Any]:
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        pose = Pose(np.asarray(R, dtype=float), np.asarray(t, dtype=float).reshape(3))
        return {"depth": render_depth(self.mesh_path, pose, K, (0, 0, WIDTH, HEIGHT))}


def test_vsd_matches_bop_toolkit(bop: Any, blob: SyntheticObject) -> None:
    pose_error, _ = bop
    gt = pose_at((0.4, 0.1, -0.3), (20, 10, 650))
    depth = render_depth(blob.mesh_path, gt, K_LMO, (0, 0, WIDTH, HEIGHT))
    scene = np.where(depth > 0, depth, 900.0)  # a background plane behind the object
    scene[200:240, 300:330] -= 40.0  # an occluder in front of part of the object
    scene[250:256, :] = 0.0  # a band of missing depth
    renderer = _BopRenderer(blob.mesh_path)
    render = depth_renderer(blob.mesh_path, K_LMO)
    rng = np.random.default_rng(3)
    for angle, shift in ((0.0, 0.0), (3.0, 4.0), (12.0, 10.0), (60.0, 40.0)):
        delta = Rotation.from_rotvec(np.deg2rad(angle) * np.array([0.6, 0.0, 0.8]))
        est = Pose(gt.R @ delta.as_matrix(), gt.t + shift * rng.normal(size=3))
        ours = metrics.vsd(
            est,
            gt,
            depth_test=scene,
            K=K_LMO,
            vertices=blob.vertices,
            diameter=blob.diameter,
            render=render,
        )
        reference = pose_error.vsd(
            est.R,
            est.t.reshape(3, 1),
            gt.R,
            gt.t.reshape(3, 1),
            scene,
            K_LMO,
            metrics.VSD_DELTA,
            list(metrics.VSD_TAUS),
            True,
            blob.diameter,
            renderer,
            1,
            cost_type="step",
        )
        assert ours == pytest.approx(reference, abs=1e-12)
    assert ours[0] > 0.5  # the 60-degree error is large
