"""2D detections in the BOP'23 default-detections format (COCO-style JSON list).

Each entry: scene_id, image_id, category_id (= BOP obj_id), score, bbox [x, y, w, h], time, and
optionally segmentation as uncompressed COCO RLE ({"counts": [...], "size": [h, w]}), which some
files store as a Python-literal string.
"""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from posegrasp.data.bop import Target


@dataclass(frozen=True, slots=True)
class Detection:
    scene_id: int
    im_id: int
    obj_id: int
    score: float
    bbox: tuple[float, float, float, float]  # x, y, w, h in pixels
    seconds: float  # detector time for the image, as reported
    rle: dict[str, Any] | None = None

    def mask(self, height: int, width: int) -> NDArray[np.bool_]:
        """Instance mask; the box itself when the detector gives no segmentation."""
        if self.rle is not None:
            return decode_rle(self.rle)
        mask = np.zeros((height, width), dtype=bool)
        x, y, w, h = self.bbox
        x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
        x1, y1 = min(width, int(np.ceil(x + w))), min(height, int(np.ceil(y + h)))
        mask[y0:y1, x0:x1] = True
        return mask

    @property
    def has_mask(self) -> bool:
        return self.rle is not None


def decode_rle(rle: dict[str, Any]) -> NDArray[np.bool_]:
    """Uncompressed COCO RLE: run lengths alternating 0/1 (starting with 0), column-major."""
    height, width = (int(v) for v in rle["size"])
    counts = np.asarray(rle["counts"], dtype=np.int64)
    if counts.sum() != height * width:
        raise ValueError(f"RLE covers {counts.sum()} pixels, image has {height * width}")
    values = np.zeros(len(counts), dtype=bool)
    values[1::2] = True
    flat = np.repeat(values, counts)
    return flat.reshape((width, height)).T


def encode_rle(mask: NDArray[np.bool_]) -> dict[str, Any]:
    """Inverse of decode_rle (used by tests and for writing masks)."""
    flat = np.asarray(mask, dtype=bool).T.reshape(-1)
    change = np.flatnonzero(np.diff(flat.astype(np.int8))) + 1
    bounds = np.concatenate([[0], change, [flat.size]])
    counts = np.diff(bounds).tolist()
    if flat[0]:
        counts = [0, *counts]
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def parse(entries: Iterable[dict[str, Any]]) -> list[Detection]:
    detections = []
    for e in entries:
        seg = e.get("segmentation")
        if isinstance(seg, str):
            seg = ast.literal_eval(seg)
        detections.append(
            Detection(
                scene_id=int(e["scene_id"]),
                im_id=int(e["image_id"]),
                obj_id=int(e["category_id"]),
                score=float(e["score"]),
                bbox=tuple(float(v) for v in e["bbox"]),  # type: ignore[arg-type]
                seconds=float(e.get("time", 0.0)),
                rle=seg,
            )
        )
    return detections


def load(path: Path) -> list[Detection]:
    return parse(json.loads(path.read_text(encoding="utf-8")))


def for_targets(
    detections: Sequence[Detection], targets: Sequence[Target], *, extra: int = 0
) -> dict[tuple[int, int, int], list[Detection]]:
    """Top-scoring detections of each target object in each targeted image.

    BOP's localisation task gives the number of instances per object and image; a method may use
    it. `extra` keeps that many more candidates for pose-level re-ranking.
    """
    wanted = {(t.scene_id, t.im_id, t.obj_id): t.inst_count for t in targets}
    grouped: dict[tuple[int, int, int], list[Detection]] = defaultdict(list)
    for d in detections:
        key = (d.scene_id, d.im_id, d.obj_id)
        if key in wanted:
            grouped[key].append(d)
    return {
        key: sorted(grouped.get(key, []), key=lambda d: -d.score)[: count + extra]
        for key, count in wanted.items()
    }
