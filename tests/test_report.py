"""The benchmark report on a small, hand-made results tree."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from posegrasp import report


def _pose_row(im_id: int, obj_id: int, correct: bool, visib: float) -> dict[str, Any]:
    err = 1.0 if correct else 1e4
    return {
        "scene_id": 2,
        "im_id": im_id,
        "obj_id": obj_id,
        "diameter": 100.0,
        "symmetric": 0.0,
        "visib_fract": visib,
        "detected": True,
        "mssd": err,
        "mspd": err,
        "add": err,
        "adi": err,
        "vsd": [0.0 if correct else 1.0] * 10,
        "score": 0.9,
        "seconds": 0.5,
        "detection_seconds": 0.1,
        "R": np.eye(3).reshape(-1).tolist(),
        "t": [0.0, 0.0, 500.0],
    }


def _grasp_row(im_id: int, obj_id: int, success: bool, mssd: float | None) -> dict[str, Any]:
    return {
        "scene_id": 2,
        "im_id": im_id,
        "obj_id": obj_id,
        "diameter": 100.0,
        "mssd": mssd,
        "planned": mssd is not None,
        "success": success,
        "reason": "ok" if success else ("no_pose" if mssd is None else "no_contact"),
        "seconds": 0.2,
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def results(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    gt = root / "ppf-score" / "gt"
    # two shards; image 1 appears in both (a resumed run) and must count once
    _write(
        gt / "all-shard-0-of-2.jsonl", [_pose_row(1, 1, True, 0.95), _pose_row(1, 5, False, 0.2)]
    )
    _write(gt / "all-shard-1-of-2.jsonl", [_pose_row(2, 1, True, 0.6), _pose_row(1, 1, True, 0.95)])
    _write(gt / "dev.jsonl", [_pose_row(9, 1, False, 0.5)])  # another subset: ignored
    _write(
        gt / "all-shard-0-of-2.grasp.jsonl",
        [_grasp_row(1, 1, True, 1.0), _grasp_row(1, 5, False, 30.0), _grasp_row(2, 1, True, 2.0)],
    )
    _write(root / "oracle" / "all.grasp.jsonl", [_grasp_row(1, 1, True, 0.0)])
    _write(tmp_path / "latency" / "ppf-score" / "gt" / "dev.jsonl", [_pose_row(9, 1, True, 1.0)])
    return root


def test_report_merges_shards_and_summarises(results: Path) -> None:
    data = report.build(results, subset="all", latency=results.parent / "latency")
    summary = data["poses"]["ppf-score/gt"]["summary"]
    assert summary["instances"] == 3
    assert summary["ar"] == pytest.approx(2 / 3)
    assert data["poses"]["ppf-score/gt"]["by_object"] == {"ape": 1.0, "can": 0.0}
    assert data["poses"]["ppf-score/gt"]["by_visibility"]["0-0.3"]["n"] == 1
    assert data["grasps"]["ppf-score/gt"]["success_rate"] == pytest.approx(2 / 3)
    assert data["grasps"]["oracle"]["success_rate"] == 1.0
    assert data["grasp_success_by_mssd"]["0-0.05"] == {"n": 2, "success_rate": 1.0}
    assert data["grasp_success_by_mssd"]["0.2-0.5"]["n"] == 1
    assert data["latency"]["ppf-score"]["p50"] == pytest.approx(0.5)
    assert "fpfh-verify-r5/gt" not in data["poses"]  # no runs: left out


def test_report_files(results: Path, tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    data = report.build(results, subset="all")
    out = tmp_path / "report"
    written = report.write(data, out)
    md = (out / "results.md").read_text(encoding="utf-8")
    assert "| PPF + ICP | GT masks | **0.667** |" in md
    assert "Oracle (true pose)" in md
    csv = (out / "bop" / "ppf-score-gt_lmo-test.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv) == 4  # header + 3 poses
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["subset"] == "all"
    charts = sorted(p.name for p in written if p.suffix == ".png")
    assert "pose_ar-light.png" in charts
    assert "pose_ar-dark.png" in charts
    assert "success_by_error-dark.png" in charts
