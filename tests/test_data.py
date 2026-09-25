import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from posegrasp.data import detections as det
from posegrasp.data.bop import Dataset, Pose, Target
from posegrasp.data.download import extract

DATA = Path(__file__).resolve().parents[1] / "data"


def test_rle_round_trip_and_column_major_order() -> None:
    rng = np.random.default_rng(0)
    mask = rng.random((7, 5)) > 0.6
    rle = det.encode_rle(mask)
    assert np.array_equal(det.decode_rle(rle), mask)
    # counts run down the columns first: a mask with only the top-left pixel set
    single = np.zeros((3, 2), dtype=bool)
    single[0, 0] = True
    assert det.encode_rle(single)["counts"] == [0, 1, 5]


def test_rle_rejects_wrong_size() -> None:
    with pytest.raises(ValueError, match="RLE covers"):
        det.decode_rle({"counts": [1, 2], "size": [2, 2]})


def test_parse_accepts_string_segmentation_and_box_masks() -> None:
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True
    entries = [
        {
            "scene_id": 2,
            "image_id": 3,
            "category_id": 1,
            "score": 0.9,
            "bbox": [2, 1, 3, 2],
            "time": 0.1,
            "segmentation": str(det.encode_rle(mask)),
        },
        {"scene_id": 2, "image_id": 3, "category_id": 5, "score": 0.8, "bbox": [0.5, 0, 2, 1.5]},
    ]
    with_mask, box_only = det.parse(entries)
    assert np.array_equal(with_mask.mask(4, 6), mask)
    assert not box_only.has_mask
    assert box_only.mask(4, 6)[:2, :3].all()
    assert box_only.mask(4, 6).sum() == 6


def test_for_targets_keeps_top_scores_per_object() -> None:
    def d(obj: int, score: float, im: int = 3) -> det.Detection:
        return det.Detection(2, im, obj, score, (0, 0, 1, 1), 0.0)

    detections = [d(1, 0.2), d(1, 0.9), d(1, 0.5), d(5, 0.7), d(8, 0.99), d(1, 0.95, im=4)]
    chosen = det.for_targets(detections, [Target(2, 3, 1, 2), Target(2, 3, 9, 1)])
    assert [x.score for x in chosen[(2, 3, 1)]] == [0.9, 0.5]
    assert chosen[(2, 3, 9)] == []  # a target without detections stays, empty
    assert len(det.for_targets(detections, [Target(2, 3, 1, 1)], extra=5)[(2, 3, 1)]) == 3


def _write_scene(root: Path) -> None:
    scene = root / "test" / "000002"
    for sub in ("rgb", "depth", "mask_visib"):
        (scene / sub).mkdir(parents=True)
    (root / "camera.json").write_text(json.dumps({"width": 8, "height": 6}))
    K = [500.0, 0, 4, 0, 500, 3, 0, 0, 1]
    (scene / "scene_camera.json").write_text(json.dumps({"7": {"cam_K": K, "depth_scale": 0.1}}))
    R = np.eye(3).reshape(-1).tolist()
    (scene / "scene_gt.json").write_text(
        json.dumps({"7": [{"cam_R_m2c": R, "cam_t_m2c": [1, 2, 900], "obj_id": 5}]})
    )
    (scene / "scene_gt_info.json").write_text(
        json.dumps({"7": [{"visib_fract": 0.4, "bbox_visib": [1, 1, 3, 2]}]})
    )
    cv2.imwrite(str(scene / "rgb" / "000007.png"), np.zeros((6, 8, 3), np.uint8))
    cv2.imwrite(str(scene / "depth" / "000007.png"), np.full((6, 8), 9000, np.uint16))
    mask = np.zeros((6, 8), np.uint8)
    mask[1:3, 1:4] = 255
    cv2.imwrite(str(scene / "mask_visib" / "000007_000000.png"), mask)
    models = root / "models"
    models.mkdir()
    info = {
        "5": {"diameter": 120.5},
        "10": {
            "diameter": 160,
            "symmetries_discrete": [[-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]],
        },
    }
    (models / "models_info.json").write_text(json.dumps(info))
    (root / "test_targets_bop19.json").write_text(
        json.dumps([{"scene_id": 2, "im_id": 7, "obj_id": 5, "inst_count": 1}])
    )


def test_dataset_reads_frames_models_and_targets(tmp_path: Path) -> None:
    _write_scene(tmp_path)
    ds = Dataset(tmp_path)
    (target,) = ds.targets()
    (frame,) = list(ds.frames([target, target]))
    assert frame.key == (2, 7)
    assert frame.camera.K[0, 0] == 500
    assert (frame.camera.width, frame.camera.height) == (8, 6)
    assert np.allclose(frame.depth_mm(), 900.0)  # 9000 * 0.1 mm
    (gt,) = frame.gt
    assert gt.obj_id == 5
    assert gt.visib_fract == 0.4
    assert np.allclose(gt.pose.t, [1, 2, 900])
    assert frame.gt_mask(gt).sum() == 6
    models = ds.models()
    assert not models[5].is_symmetric
    assert models[10].is_symmetric
    assert models[10].symmetries_discrete[0][0, 0] == -1


def test_pose_matrix_round_trip() -> None:
    pose = Pose(R=np.eye(3), t=np.array([1.0, 2.0, 3.0]))
    assert np.allclose(Pose.from_matrix(pose.matrix()).t, [1, 2, 3])


def test_extract_strips_the_dataset_prefix(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("lmo/camera.json", "{}")
        zf.writestr("models/models_info.json", "{}")
    extract(archive, tmp_path / "lmo")
    assert (tmp_path / "lmo" / "camera.json").exists()
    assert (tmp_path / "lmo" / "models" / "models_info.json").exists()


@pytest.mark.dataset
@pytest.mark.skipif(not (DATA / "lmo" / "test").exists(), reason="LM-O not downloaded")
def test_cnos_masks_overlap_ground_truth_on_real_data() -> None:
    ds = Dataset(DATA / "lmo")
    targets = ds.targets()[:40]
    chosen = det.for_targets(det.load(DATA / "detections" / "det_cnos-fastsam_lmo.json"), targets)
    ious = []
    for frame in ds.frames(targets):
        for gt in frame.gt:
            gt_mask = frame.gt_mask(gt)
            for d in chosen.get((frame.scene_id, frame.im_id, gt.obj_id), []):
                m = d.mask(frame.camera.height, frame.camera.width)
                ious.append((m & gt_mask).sum() / max(1, (m | gt_mask).sum()))
    assert ious
    assert np.median(ious) > 0.5
