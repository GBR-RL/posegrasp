"""Evaluation pipeline helpers, and the full scoring path on the real data."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from posegrasp import pipeline
from posegrasp.data.bop import Dataset
from synthetic import K_LMO

DATA = Path(__file__).resolve().parents[1] / "data"
HAS_LMO = (DATA / "lmo" / "test").exists()


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scene_id": 2,
        "im_id": 3,
        "obj_id": 1,
        "diameter": 100.0,
        "symmetric": 0.0,
        "mssd": 1.0,
        "mspd": 1.0,
        "add": 1.0,
        "adi": 1.0,
        "vsd": [0.0] * 10,
        "score": 0.9,
        "seconds": 0.5,
        "detection_seconds": 0.25,
        "R": np.eye(3).reshape(-1).tolist(),
        "t": [1.0, 2.0, 3.0],
    }
    row.update(overrides)
    return row


def test_points_mask_marks_projected_pixels() -> None:
    points = np.array([[0.0, 0.0, 500.0], [50.0, 0.0, 500.0], [0.0, 0.0, -1.0]])
    mask = pipeline.points_mask(points[:2], K_LMO, 480, 640)
    assert mask.sum() == 2
    u = round(50 * K_LMO[0, 0] / 500 + K_LMO[0, 2])
    assert mask[round(K_LMO[1, 2]), u]


def test_crop_box_segment_keeps_the_object_not_the_background() -> None:
    rng = np.random.default_rng(0)
    # the camera sees the front half of a sphere of diameter 100 mm centred at z = 600
    xy = rng.uniform(-50, 50, size=(3000, 2))
    xy = xy[np.linalg.norm(xy, axis=1) < 49]
    front = np.column_stack([xy, 600 - np.sqrt(50**2 - (xy**2).sum(axis=1))])
    background = np.column_stack(
        [rng.uniform(-300, 300, 2000), rng.uniform(-300, 300, 2000), np.full(2000, 900.0)]
    )
    box_centre = np.array([0.0, 0.0, 600.0])  # on the ray through the box centre
    kept = pipeline.crop_box_segment(np.vstack([front, background]), box_centre, 100.0)
    assert len(kept) == len(front)
    assert (kept[:, 2] < 800).all()


def test_summarize_rows_counts_misses_as_failures() -> None:
    miss = _row(mssd=None, mspd=None, add=None, adi=None, vsd=None, score=None, R=None, t=None)
    summary = pipeline.summarize_rows([_row(), miss])
    assert summary["ar"] == pytest.approx(0.5)
    assert summary["instances"] == 2
    assert summary["seconds_p50"] == pytest.approx(0.5)  # misses carry no latency
    old = {k: v for k, v in _row().items() if k != "vsd"}
    assert math.isnan(pipeline.summarize_rows([old])["ar"])  # VSD never computed


def test_bop_csv_has_one_line_per_pose_and_image_time() -> None:
    rows = [_row(), _row(obj_id=5, seconds=1.0), _row(obj_id=6, R=None, t=None)]
    lines = pipeline.to_bop_csv(rows).splitlines()
    assert lines[0] == "scene_id,im_id,obj_id,score,R,t,time"
    assert len(lines) == 3  # the row without a pose is left out
    fields = lines[1].split(",")
    assert fields[:3] == ["2", "3", "1"]
    assert len(fields[4].split()) == 9
    assert len(fields[5].split()) == 3
    # time = all pose time in the image + detector time, the same on every line of the image
    assert float(fields[6]) == pytest.approx(0.5 + 1.0 + 0.5 + 3 * 0.25)
    assert lines[2].split(",")[6] == fields[6]


@pytest.mark.dataset
@pytest.mark.skipif(not HAS_LMO, reason="LM-O not downloaded")
def test_true_pose_renders_onto_the_measured_depth_and_scores_zero() -> None:
    from posegrasp import metrics
    from posegrasp.geometry import load_vertices
    from posegrasp.render import depth_renderer

    ds = Dataset(DATA / "lmo")
    target = ds.targets()[0]
    frame = ds.frame(target.scene_id, target.im_id)
    model = ds.models()[target.obj_id]
    gt, _ = pipeline.gt_pose(frame, target.obj_id)
    depth = frame.depth_mm()
    instance = next(g for g in frame.gt if g.obj_id == target.obj_id)
    visible = frame.gt_mask(instance) & (depth > 0)
    rendered = depth_renderer(model.eval_mesh, frame.camera.K)(gt, (0, 0, 640, 480))
    both = visible & (rendered > 0)
    assert both.sum() > 100
    assert np.median(np.abs(rendered[both] - depth[both])) < 5.0  # mm
    vertices = load_vertices(model.eval_mesh)
    vsd = metrics.vsd(
        gt,
        gt,
        depth_test=depth,
        K=frame.camera.K,
        vertices=vertices,
        diameter=model.diameter,
        render=depth_renderer(model.eval_mesh, frame.camera.K),
    )
    assert max(vsd) == 0.0
    assert metrics.mssd(gt, gt, vertices, metrics.symmetry_transforms(model)) == 0.0
