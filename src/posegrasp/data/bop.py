"""Reader for datasets in the BOP format (bop_toolkit: docs/bop_datasets_format.md).

Units follow BOP: millimetres for translations, depth and models; rotations map model to camera
coordinates (x_cam = R @ x_model + t).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Camera:
    K: FloatArray  # 3x3 intrinsics
    depth_scale: float  # depth image value * depth_scale = millimetres
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class Pose:
    R: FloatArray  # 3x3, model -> camera
    t: FloatArray  # (3,), millimetres

    def matrix(self) -> FloatArray:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.R, self.t
        return T

    @staticmethod
    def from_matrix(T: FloatArray) -> Pose:
        return Pose(R=np.asarray(T[:3, :3], dtype=float), t=np.asarray(T[:3, 3], dtype=float))


@dataclass(frozen=True, slots=True)
class GtInstance:
    obj_id: int
    index: int  # position in scene_gt.json for this image; names the mask files
    pose: Pose
    visib_fract: float
    bbox_visib: tuple[int, int, int, int]  # x, y, w, h


@dataclass(frozen=True, slots=True)
class Frame:
    scene_id: int
    im_id: int
    rgb_path: Path
    depth_path: Path
    camera: Camera
    gt: tuple[GtInstance, ...]
    scene_dir: Path

    @property
    def key(self) -> tuple[int, int]:
        return self.scene_id, self.im_id

    def depth_mm(self) -> FloatArray:
        depth = cv2.imread(str(self.depth_path), cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise FileNotFoundError(self.depth_path)
        scaled: FloatArray = depth.astype(np.float64) * self.camera.depth_scale
        return scaled

    def rgb(self) -> NDArray[np.uint8]:
        image = cv2.imread(str(self.rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(self.rgb_path)
        return np.ascontiguousarray(image[:, :, ::-1])

    def gt_mask(self, instance: GtInstance, *, visible: bool = True) -> NDArray[np.bool_]:
        folder = "mask_visib" if visible else "mask"
        path = self.scene_dir / folder / f"{self.im_id:06d}_{instance.index:06d}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        binary: NDArray[np.bool_] = mask > 0
        return binary


@dataclass(frozen=True, slots=True)
class ObjectModel:
    obj_id: int
    mesh_path: Path
    diameter: float  # millimetres
    symmetries_discrete: tuple[FloatArray, ...] = ()  # 4x4 model-frame transforms
    symmetries_continuous: tuple[dict[str, Any], ...] = ()  # {"axis": [...], "offset": [...]}

    @property
    def is_symmetric(self) -> bool:
        return bool(self.symmetries_discrete or self.symmetries_continuous)


@dataclass(frozen=True, slots=True)
class Target:
    """One BOP evaluation target: estimate `inst_count` instances of `obj_id` in an image."""

    scene_id: int
    im_id: int
    obj_id: int
    inst_count: int


@dataclass(slots=True)
class Dataset:
    root: Path
    split: str = "test"
    _scene_cache: dict[int, dict[str, Any]] = field(default_factory=dict)

    def models(self) -> dict[int, ObjectModel]:
        info = _read_json(self.root / "models" / "models_info.json")
        models = {}
        for key, entry in info.items():
            obj_id = int(key)
            models[obj_id] = ObjectModel(
                obj_id=obj_id,
                mesh_path=self.root / "models" / f"obj_{obj_id:06d}.ply",
                diameter=float(entry["diameter"]),
                symmetries_discrete=tuple(
                    np.asarray(s, dtype=float).reshape(4, 4)
                    for s in entry.get("symmetries_discrete", [])
                ),
                symmetries_continuous=tuple(entry.get("symmetries_continuous", [])),
            )
        return models

    def targets(self, filename: str = "test_targets_bop19.json") -> list[Target]:
        return [
            Target(int(t["scene_id"]), int(t["im_id"]), int(t["obj_id"]), int(t["inst_count"]))
            for t in _read_json(self.root / filename)
        ]

    def image_size(self) -> tuple[int, int]:
        """(width, height) from the dataset's camera.json."""
        if "size" not in self._scene_cache.get(-1, {}):
            cam = _read_json(self.root / "camera.json")
            self._scene_cache[-1] = {"size": (int(cam["width"]), int(cam["height"]))}
        size: tuple[int, int] = self._scene_cache[-1]["size"]
        return size

    def scene_ids(self) -> list[int]:
        return sorted(int(p.name) for p in (self.root / self.split).iterdir() if p.is_dir())

    def _scene(self, scene_id: int) -> dict[str, Any]:
        if scene_id not in self._scene_cache:
            scene_dir = self.root / self.split / f"{scene_id:06d}"
            self._scene_cache[scene_id] = {
                "dir": scene_dir,
                "camera": _read_json(scene_dir / "scene_camera.json"),
                "gt": _read_json(scene_dir / "scene_gt.json"),
                "gt_info": _read_json(scene_dir / "scene_gt_info.json"),
            }
        return self._scene_cache[scene_id]

    def frame(self, scene_id: int, im_id: int) -> Frame:
        scene = self._scene(scene_id)
        scene_dir: Path = scene["dir"]
        cam = scene["camera"][str(im_id)]
        rgb_path = scene_dir / "rgb" / f"{im_id:06d}.png"
        if not rgb_path.exists():
            rgb_path = rgb_path.with_suffix(".jpg")
        depth_path = scene_dir / "depth" / f"{im_id:06d}.png"
        width, height = self.image_size()
        camera = Camera(
            K=np.asarray(cam["cam_K"], dtype=float).reshape(3, 3),
            depth_scale=float(cam["depth_scale"]),
            width=width,
            height=height,
        )
        gt = tuple(
            GtInstance(
                obj_id=int(g["obj_id"]),
                index=i,
                pose=Pose(
                    R=np.asarray(g["cam_R_m2c"], dtype=float).reshape(3, 3),
                    t=np.asarray(g["cam_t_m2c"], dtype=float),
                ),
                visib_fract=float(info["visib_fract"]),
                bbox_visib=tuple(int(v) for v in info["bbox_visib"]),  # type: ignore[arg-type]
            )
            for i, (g, info) in enumerate(
                zip(scene["gt"][str(im_id)], scene["gt_info"][str(im_id)], strict=True)
            )
        )
        return Frame(scene_id, im_id, rgb_path, depth_path, camera, gt, scene_dir)

    def frames(self, targets: Sequence[Target]) -> Iterator[Frame]:
        """Frames of the targeted images, in target order, each once."""
        seen: set[tuple[int, int]] = set()
        for t in targets:
            if (t.scene_id, t.im_id) not in seen:
                seen.add((t.scene_id, t.im_id))
                yield self.frame(t.scene_id, t.im_id)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
