"""Failure gallery: typical ways the pipeline fails, picked from the benchmark results.

Each panel is a crop of the RGB image with the object's outline at the true pose (blue) and at
the estimated pose (orange), with the case and its numbers in the caption.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from posegrasp.data.bop import Dataset, Pose
from posegrasp.render import render_depth
from posegrasp.report import CONDITIONS, LMO_OBJECTS, METHODS, merged_rows

TRUE_COLOUR = (42, 120, 214)  # RGB, the categorical blue of the charts
ESTIMATE_COLOUR = (235, 104, 52)  # orange
Row = dict[str, Any]


@dataclass(frozen=True, slots=True)
class Case:
    title: str
    method: str
    condition: str
    row: Row
    grasp_reason: str | None = None


def _error(row: Row) -> float:
    return np.inf if row.get("mssd") is None else row["mssd"] / row["diameter"]


def _key(row: Row) -> tuple[int, int, int]:
    return row["scene_id"], row["im_id"], row["obj_id"]


def _inside(dataset: Dataset, row: Row, margin: int = 80) -> bool:
    """Whether the object's true centre projects at least `margin` pixels inside the image."""
    frame = dataset.frame(row["scene_id"], row["im_id"])
    t = next(g for g in frame.gt if g.obj_id == row["obj_id"]).pose.t
    u, v = (frame.camera.K @ t)[:2] / t[2]
    w, h = frame.camera.width, frame.camera.height
    return bool(margin <= u <= w - margin and margin <= v <= h - margin)


def pick_cases(
    results: Path, dataset: Dataset, subset: str = "all", method: str = "ppf-score"
) -> list[Case]:
    """Up to four failures: heavy occlusion, a zero-shot detection miss, a box-only detection
    miss, and a good pose whose grasp still fails."""
    rows = {
        c: {_key(r): r for r in merged_rows(results / method / c, subset, "jsonl")}
        for c in CONDITIONS
    }
    gt = rows.get("gt", {})
    cases: list[Case] = []

    def typical(candidates: list[Row], title: str, condition: str) -> None:
        """The median failure: wrong (0.25 to 1.5 diameters off) but still in the picture."""
        shown = sorted(
            (
                r
                for r in candidates
                if 0.25 < _error(r) < 1.5 and r["visib_fract"] >= 0.1 and _inside(dataset, r)
            ),
            key=_error,
        )
        if shown:
            cases.append(Case(title, method, condition, shown[len(shown) // 2]))

    typical([r for r in gt.values() if r["visib_fract"] < 0.3], "Occlusion", "gt")
    for condition, title in (("cnos", "Zero-shot mask"), ("gdrnpp", "Box-only detection")):
        typical(
            [r for k, r in rows.get(condition, {}).items() if k in gt and _error(gt[k]) < 0.05],
            title,
            condition,
        )
    grasps = {_key(r): r for r in merged_rows(results / method / "gt", subset, "grasp.jsonl")}
    failed = [
        (gt[k], g["reason"])
        for k, g in grasps.items()
        if k in gt and not g["success"] and _error(gt[k]) < 0.1 and g["reason"] != "no_grasp"
    ]
    if failed:
        row, reason = min(failed, key=lambda rg: _error(rg[0]))
        cases.append(Case("Good pose, failed grasp", method, "gt", row, reason))
    return cases


def _outline(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    import cv2

    edges = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    outline: NDArray[np.bool_] = np.asarray(edges) > 0
    return outline


def render_case(dataset: Dataset, case: Case, *, size: int = 240) -> NDArray[np.uint8]:
    """The annotated crop of one case (size x size, RGB)."""
    import cv2

    row = case.row
    frame = dataset.frame(row["scene_id"], row["im_id"])
    model = dataset.models()[row["obj_id"]]
    K, w, h = frame.camera.K, frame.camera.width, frame.camera.height
    image = frame.rgb().copy()
    true_pose = next(g for g in frame.gt if g.obj_id == row["obj_id"]).pose
    true_mask = render_depth(model.mesh_path, true_pose, K, (0, 0, w, h)) > 0
    poses = [(true_mask, TRUE_COLOUR)]
    if row.get("R") is not None:
        est = Pose(np.asarray(row["R"]).reshape(3, 3), np.asarray(row["t"]))
        poses.append((render_depth(model.mesh_path, est, K, (0, 0, w, h)) > 0, ESTIMATE_COLOUR))
    for mask, colour in poses:
        edge = cv2.dilate(_outline(mask).astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
        image[edge] = colour
    union = np.logical_or.reduce([mask for mask, _ in poses])
    ys, xs = np.nonzero(union)
    cx, cy = (int(xs.min()) + int(xs.max())) // 2, (int(ys.min()) + int(ys.max())) // 2
    half = int(max(int(xs.ptp()), int(ys.ptp()), 80) * 0.6)
    padded = cv2.copyMakeBorder(image, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    crop = padded[cy : cy + 2 * half, cx : cx + 2 * half]  # centred on (cx, cy), square
    return np.asarray(cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), dtype=np.uint8)


def caption(case: Case) -> list[str]:
    row = case.row
    name = LMO_OBJECTS.get(row["obj_id"], str(row["obj_id"]))
    error = _error(row)
    lines = [
        case.title,
        f"{name}, {100 * row['visib_fract']:.0f} % visible",
        f"{METHODS[case.method]}, {CONDITIONS[case.condition]}",
        "no pose" if not np.isfinite(error) else f"pose error (MSSD) {100 * error:.0f} % of size",
    ]
    if case.grasp_reason:
        lines.append(f"grasp: {case.grasp_reason.replace('_', ' ')}")
    return lines


def render_gallery(dataset: Dataset, cases: list[Case], out: Path, *, size: int = 240) -> Path:
    """All cases side by side with captions, as one PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(cases), figsize=(2.6 * len(cases), 3.5), dpi=150, facecolor="#fcfcfb",
        squeeze=False,
    )  # fmt: skip
    for ax, case in zip(axes[0], cases, strict=True):
        ax.imshow(render_case(dataset, case, size=size))
        ax.set_axis_off()
        lines = caption(case)
        ax.set_title(lines[0], fontsize=9, color="#0b0b0b", loc="left", fontweight="bold")
        ax.text(0, -0.04, "\n".join(lines[1:]), transform=ax.transAxes, va="top", fontsize=7.5,
                color="#52514e")  # fmt: skip
    fig.text(0.01, 0.02, "outline: blue = true pose, orange = estimate", fontsize=7.5,
             color="#898781")  # fmt: skip
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.25, wspace=0.06)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="#fcfcfb")
    plt.close(fig)
    return out
