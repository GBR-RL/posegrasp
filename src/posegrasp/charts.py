"""Report charts (matplotlib, static PNG), drawn once per colour theme for the README's light and
dark modes.

Specs: bars at most 24 px thick with a 4 px rounded data end and a square baseline, 2 px surface
gaps between touching bars, 2 px lines, 8 px markers with a 2 px surface ring, hairline solid
gridlines, text in ink colours (never the series colour), a legend whenever there are two or more
series. Series colours are the first two slots of a categorical palette checked for colour-vision
deficiency in both themes; the method keeps its colour in every chart.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

THEMES: dict[str, dict[str, Any]] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": {"ppf-score": "#2a78d6", "fpfh-verify-r5": "#eb6834"},
        "single": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": {"ppf-score": "#3987e5", "fpfh-verify-r5": "#d95926"},
        "single": "#3987e5",
    },
}
WIDTH, DPI, SCALE = 760, 100, 2  # CSS pixels, matplotlib dpi, saved at 2x for sharp README images

Series = tuple[str, str, Sequence[float]]  # key (colour), label, values


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
        }
    )
    return plt


class _Chart:
    """A figure with `ncols` plot panels laid out in pixels, plus title, subtitle and legend."""

    def __init__(
        self,
        theme: dict[str, Any],
        *,
        title: str,
        subtitle: str,
        height: int = 380,
        ncols: int = 1,
        left: int = 52,
        right: int = 24,
        top: int = 92,
        bottom: int = 52,
        gap: int = 28,
    ) -> None:
        self.plt = _plt()
        self.t = theme
        self.height = height
        self.fig = self.plt.figure(
            figsize=(WIDTH / DPI, height / DPI), dpi=DPI, facecolor=theme["surface"]
        )
        panel = (WIDTH - left - right - gap * (ncols - 1)) / ncols
        self.axes = []
        for i in range(ncols):
            ax = self.fig.add_axes(
                (
                    (left + i * (panel + gap)) / WIDTH,
                    bottom / height,
                    panel / WIDTH,
                    (height - top - bottom) / height,
                )
            )
            self._style(ax)
            self.axes.append(ax)
        self.fig.text(
            16 / WIDTH,
            1 - 26 / height,
            title,
            color=theme["ink"],
            fontsize=13,
            fontweight="semibold",
            va="baseline",
        )
        self.fig.text(
            16 / WIDTH, 1 - 46 / height, subtitle, color=theme["ink2"], fontsize=9.5, va="baseline"
        )

    def _style(self, ax: Any) -> None:
        t = self.t
        ax.set_facecolor(t["surface"])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(t["axis"])
        ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(colors=t["muted"], length=0, labelsize=9, pad=6)
        ax.yaxis.grid(True, color=t["grid"], linewidth=1, linestyle="-")
        ax.set_axisbelow(True)

    def legend(self, items: Sequence[tuple[str, str]], *, line: bool = False) -> None:
        """Swatches (or line keys) with labels, in one row under the subtitle."""
        x = 16.0
        y = 1 - 72 / self.height
        for colour, label in items:
            if line:
                self.fig.add_artist(
                    self.plt.Line2D(
                        [x / WIDTH, (x + 16) / WIDTH],
                        [y + 4 / self.height] * 2,
                        color=colour,
                        linewidth=2,
                        solid_capstyle="round",
                    )
                )
                x += 22
            else:
                self.fig.add_artist(
                    self.plt.Rectangle(
                        (x / WIDTH, y),
                        10 / WIDTH,
                        10 / self.height,
                        color=colour,
                        transform=self.fig.transFigure,
                    )
                )
                x += 16
            text = self.fig.text(x / WIDTH, y, label, color=self.t["ink2"], fontsize=9.5)
            x += text.get_window_extent(self.fig.canvas.get_renderer()).width + 20

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=DPI * SCALE, facecolor=self.t["surface"])
        self.plt.close(self.fig)
        return path


def _per_px(ax: Any) -> tuple[float, float]:
    """Data units per pixel along x and y."""
    box = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    return (x1 - x0) / box.width, (y1 - y0) / box.height


def _bar(ax: Any, x: float, width: float, height: float, colour: str) -> None:
    """A column from the baseline with 4 px rounded top corners."""
    from matplotlib.patches import Polygon

    if not height > 0:
        return
    dx, dy = _per_px(ax)
    rx, ry = min(4 * dx, width / 2), min(4 * dy, height)
    left, right = x - width / 2, x + width / 2
    arc = np.linspace(0, math.pi / 2, 8)
    verts = [(left, 0.0), (left, height - ry)]
    verts += [(left + rx - rx * math.cos(a), height - ry + ry * math.sin(a)) for a in arc]
    verts += [(right - rx + rx * math.sin(a), height - ry + ry * math.cos(a)) for a in arc]
    verts += [(right, 0.0)]
    ax.add_patch(Polygon(verts, closed=True, facecolor=colour, edgecolor="none"))


def _grouped_bars(
    chart: _Chart,
    ax: Any,
    groups: Sequence[str],
    series: Sequence[Series],
    fmt: Callable[[float], str],
) -> None:
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0, 1)
    dx, dy = _per_px(ax)
    width, gap = 24 * dx, 2 * dx
    total = len(series) * width + (len(series) - 1) * gap
    for i, (key, _, values) in enumerate(series):
        offset = -total / 2 + width / 2 + i * (width + gap)
        for g, value in enumerate(values):
            if value is None or math.isnan(value):
                continue
            _bar(ax, g + offset, width, value, chart.t["series"].get(key, chart.t["single"]))
            ax.text(
                g + offset,
                value + 5 * dy,
                fmt(value),
                ha="center",
                va="bottom",
                color=chart.t["ink2"],
                fontsize=8.5,
            )
    ax.set_xticks(range(len(groups)), groups)


def _get(data: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return math.nan
        data = data[key]
    return float(data)  # type: ignore[arg-type]


def _percent(v: float) -> str:
    return f"{100 * v:.0f}%"


def _ar_ticks(ax: Any, percent: bool = False) -> None:
    ticks = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_yticks(ticks, [f"{100 * v:.0f}%" if percent else f"{v:.2f}" for v in ticks])


def pose_ar_chart(report: dict[str, Any], theme: str, path: Path) -> Path | None:
    from posegrasp.report import CONDITIONS, METHODS

    poses = report["poses"]
    if not poses:
        return None
    t = THEMES[theme]
    chart = _Chart(
        t,
        title="Pose accuracy on LM-O",
        subtitle="BOP average recall (AR, mean of VSD, MSSD and MSPD recall) by 2D "
        "detection input; higher is better",
    )
    series = [
        (
            m,
            label,
            [poses.get(f"{m}/{c}", {}).get("summary", {}).get("ar", math.nan) for c in CONDITIONS],
        )
        for m, label in METHODS.items()
    ]
    chart.legend([(t["series"][m], label) for m, label, _ in series])
    _grouped_bars(chart, chart.axes[0], list(CONDITIONS.values()), series, lambda v: f"{v:.2f}")
    _ar_ticks(chart.axes[0])
    return chart.save(path)


def grasp_chart(report: dict[str, Any], theme: str, path: Path) -> Path | None:
    from posegrasp.report import CONDITIONS, METHODS

    grasps = report["grasps"]
    if not any(k != "oracle" for k in grasps):
        return None
    t = THEMES[theme]
    chart = _Chart(
        t,
        title="Grasps that would work",
        subtitle="Grasp planned from the estimated pose, judged on the true pose; "
        "share of all target instances",
    )
    series = [
        (m, label, [_get(grasps, f"{m}/{c}", "success_rate") for c in CONDITIONS])
        for m, label in METHODS.items()
    ]
    chart.legend([(t["series"][m], label) for m, label, _ in series])
    ax = chart.axes[0]
    _grouped_bars(chart, ax, list(CONDITIONS.values()), series, _percent)
    if "oracle" in grasps:
        rate = grasps["oracle"]["success_rate"]
        ax.axhline(rate, color=t["ink2"], linewidth=1.5, zorder=3)
        _, dy = _per_px(ax)
        ax.text(
            ax.get_xlim()[1],
            rate + 4 * dy,
            f"planned from the true pose: {_percent(rate)}",
            ha="right",
            va="bottom",
            color=t["ink2"],
            fontsize=8.5,
        )
    _ar_ticks(ax, percent=True)
    return chart.save(path)


def success_by_error_chart(report: dict[str, Any], theme: str, path: Path) -> Path | None:
    bins = {k: v for k, v in report["grasp_success_by_mssd"].items() if v["n"]}
    if not bins:
        return None
    t = THEMES[theme]
    chart = _Chart(
        t,
        title="What a pose error costs in grasps",
        subtitle="Grasp success by the error of the pose it was planned from (MSSD / "
        "object diameter); all methods and detections pooled",
    )
    labels = [f"{k}\n(n={v['n']})" for k, v in bins.items()]
    values = [v["success_rate"] for v in bins.values()]
    _grouped_bars(chart, chart.axes[0], labels, [("single", "grasp success", values)], _percent)
    _ar_ticks(chart.axes[0], percent=True)
    return chart.save(path)


def visibility_chart(report: dict[str, Any], theme: str, path: Path) -> Path | None:
    from posegrasp.report import CONDITIONS, METHODS, VISIBILITY_LABELS

    poses = report["poses"]
    if not poses:
        return None
    t = THEMES[theme]
    chart = _Chart(
        t,
        title="Occlusion is what pose estimation struggles with",
        subtitle="AR by the visible fraction of the object, one panel per detection input",
        ncols=len(CONDITIONS),
        bottom=60,
        top=118,
        height=410,
    )
    chart.legend([(t["series"][m], label) for m, label in METHODS.items()], line=True)
    x = np.arange(len(VISIBILITY_LABELS))
    for ax, (condition, name) in zip(chart.axes, CONDITIONS.items(), strict=True):
        for m in METHODS:
            entry = poses.get(f"{m}/{condition}")
            if entry is None:
                continue
            ys = [entry["by_visibility"][b]["ar"] for b in VISIBILITY_LABELS]
            ax.plot(
                x,
                ys,
                color=t["series"][m],
                linewidth=2,
                solid_capstyle="round",
                solid_joinstyle="round",
                marker="o",
                markersize=8,
                markeredgecolor=t["surface"],
                markeredgewidth=2,
                zorder=3,
            )
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.4, len(x) - 0.6)
        ax.set_xticks(x, [b.replace("-", chr(0x2013)) for b in VISIBILITY_LABELS], fontsize=8)
        ax.set_title(name, color=t["ink2"], fontsize=9.5, loc="left", pad=8)
        _ar_ticks(ax)
        if ax is not chart.axes[0]:
            ax.tick_params(labelleft=False)
    chart.fig.text(
        0.5,
        14 / chart.height,
        "visible fraction of the object",
        ha="center",
        color=t["muted"],
        fontsize=9,
    )
    return chart.save(path)


CHARTS = {
    "pose_ar": pose_ar_chart,
    "grasp_success": grasp_chart,
    "success_by_error": success_by_error_chart,
    "ar_by_visibility": visibility_chart,
}


def render_report(report: dict[str, Any], out: Path) -> list[Path]:
    written = []
    for name, draw in CHARTS.items():
        for theme in THEMES:
            path = draw(report, theme, out / f"{name}-{theme}.png")
            if path is not None:
                written.append(path)
    return written
