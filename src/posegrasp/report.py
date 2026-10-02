"""Benchmark report: merges the result files of all runs (and shards) into tables, charts and BOP
submission files.

Layout read (written by `posegrasp eval` and `posegrasp pick`):
    <results>/<method>/<condition>/<subset>[-shard-i-of-n].jsonl        pose rows
    <results>/<method>/<condition>/<subset>[-shard-i-of-n].grasp.jsonl  grasp rows
    <results>/oracle/<subset>[-shard-i-of-n].grasp.jsonl                 grasp rows, true pose
    <latency>/<method>/gt/dev.jsonl                                      pose rows, one machine
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from posegrasp.picking import ERROR_BINS, error_bin, summarize_grasps
from posegrasp.pipeline import load_rows, summarize_rows, to_bop_csv

Rows = list[dict[str, Any]]

METHODS = {
    "ppf-score": "PPF + ICP",
    "fpfh-verify-r5": "FPFH + RANSAC + ICP",
}
CONDITIONS = {
    "gt": "GT masks",
    "gdrnpp": "GDRNPP boxes (trained)",
    "cnos": "CNOS masks (zero-shot)",
}
LMO_OBJECTS = {
    1: "ape",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
}
VISIBILITY_BINS = (0.3, 0.5, 0.7, 0.9)
DASH = chr(0x2013)  # en dash: an empty table cell


def _merged(folder: Path, subset: str, suffix: str) -> Rows:
    """Rows of `<subset>.<suffix>` or of all its shards, without duplicates."""
    pattern = re.compile(rf"^{re.escape(subset)}(-shard-\d+-of-\d+)?\.{re.escape(suffix)}$")
    rows: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in sorted(folder.glob(f"{subset}*.{suffix}")):
        if pattern.match(path.name):
            for r in load_rows(path):
                rows[(r["scene_id"], r["im_id"], r["obj_id"])] = r
    return sorted(rows.values(), key=lambda r: (r["scene_id"], r["im_id"], r["obj_id"]))


def _ar(rows: Rows) -> float:
    return summarize_rows(rows)["ar"] if rows else math.nan


def _binned(rows: Rows, key: Callable[[dict[str, Any]], str], labels: Sequence[str]) -> dict:  # type: ignore[type-arg]
    out = {}
    for label in labels:
        chosen = [r for r in rows if key(r) == label]
        out[label] = {"n": len(chosen), "ar": _ar(chosen)}
    return out


def visibility_bin(visib: float) -> str:
    lower = 0.0
    for upper in VISIBILITY_BINS:
        if visib < upper:
            return f"{lower:g}-{upper:g}"
        lower = upper
    return f"{lower:g}-1"


VISIBILITY_LABELS = [visibility_bin(v) for v in (0.0, *VISIBILITY_BINS)]
ERROR_LABELS = [error_bin({"mssd": m, "diameter": 1.0}) for m in (0.0, *ERROR_BINS)] + ["no pose"]


def _latency(rows: Rows) -> dict[str, float]:
    seconds = [r["seconds"] for r in rows if r.get("score") is not None]
    if not seconds:
        return {"p50": math.nan, "p90": math.nan, "n": 0}
    return {
        "p50": float(np.percentile(seconds, 50)),
        "p90": float(np.percentile(seconds, 90)),
        "n": len(seconds),
    }


def build(results: Path, *, subset: str = "all", latency: Path | None = None) -> dict[str, Any]:
    """All numbers of the report, as one JSON-ready dict."""
    poses: dict[str, Any] = {}
    grasps: dict[str, Any] = {}
    pooled_grasp_rows: Rows = []
    bop_files: dict[str, str] = {}
    for method in METHODS:
        for condition in CONDITIONS:
            folder = results / method / condition
            rows = _merged(folder, subset, "jsonl")
            if not rows:
                continue
            key = f"{method}/{condition}"
            summary = summarize_rows(rows)
            summary["detected"] = float(np.mean([bool(r["detected"]) for r in rows]))
            summary["seconds_p90"] = _latency(rows)["p90"]
            by_object = {
                LMO_OBJECTS.get(o, str(o)): _ar([r for r in rows if r["obj_id"] == o])
                for o in sorted({r["obj_id"] for r in rows})
            }
            by_visibility = _binned(
                rows, lambda r: visibility_bin(r["visib_fract"]), VISIBILITY_LABELS
            )
            poses[key] = {
                "summary": summary,
                "by_object": by_object,
                "by_visibility": by_visibility,
            }
            bop_files[f"{method}-{condition}_lmo-test.csv"] = to_bop_csv(rows)
            grasp_rows = _merged(folder, subset, "grasp.jsonl")
            if grasp_rows:
                grasps[key] = summarize_grasps(grasp_rows)
                pooled_grasp_rows += grasp_rows
    oracle_rows = _merged(results / "oracle", subset, "grasp.jsonl")
    if oracle_rows:
        grasps["oracle"] = summarize_grasps(oracle_rows)
    by_error = {}
    for label in ERROR_LABELS:
        chosen = [r for r in pooled_grasp_rows if error_bin(r) == label]
        rate = sum(r["success"] for r in chosen) / len(chosen) if chosen else math.nan
        by_error[label] = {"n": len(chosen), "success_rate": rate}
    timing = {}
    if latency is not None:
        for method in METHODS:
            rows = load_rows(latency / method / "gt" / "dev.jsonl")
            if rows:
                timing[method] = _latency(rows)
    return {
        "subset": subset,
        "poses": poses,
        "grasps": grasps,
        "grasp_success_by_mssd": by_error,
        "latency": timing,
        "bop_files": bop_files,
    }


def _f(value: float, digits: int = 3) -> str:
    return (
        DASH
        if value is None or (isinstance(value, float) and math.isnan(value))
        else f"{value:.{digits}f}"
    )


def _pct(value: float) -> str:
    return DASH if math.isnan(value) else f"{100 * value:.1f} %"


def markdown(report: dict[str, Any]) -> str:
    """The report tables as GitHub Markdown."""
    lines = [
        "# Benchmark results",
        "",
        f"LM-O (BOP'19 test subset), `{report['subset']}` images. Generated by `posegrasp report`; "
        "the protocol is in [docs/EVAL_PROTOCOL.md](../../docs/EVAL_PROTOCOL.md).",
        "",
        "## Pose estimation",
        "",
        "| Method | Detections | AR | AR_VSD | AR_MSSD | AR_MSPD | ADD(-S) < 0.1 d | Detected |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in report["poses"].items():
        method, condition = key.split("/")
        s = entry["summary"]
        lines.append(
            f"| {METHODS[method]} | {CONDITIONS[condition]} | **{_f(s['ar'])}** "
            f"| {_f(s['ar_vsd'])} | {_f(s['ar_mssd'])} | {_f(s['ar_mspd'])} "
            f"| {_pct(s['add_s_01d'])} | {_pct(s['detected'])} |"
        )
    if report["latency"]:
        lines += [
            "",
            "### Latency",
            "",
            "Per instance, pose estimation only (segment preparation, hypotheses, ICP, selection), "
            "on the development images with ground-truth masks, both methods in one job on one "
            "machine.",
            "",
            "| Method | median | 90th percentile |",
            "|---|---:|---:|",
        ]
        for method, t in report["latency"].items():
            lines.append(f"| {METHODS[method]} | {_f(t['p50'], 2)} s | {_f(t['p90'], 2)} s |")
    objects = sorted({o for e in report["poses"].values() for o in e["by_object"]})
    if objects:
        lines += [
            "",
            "### AR per object",
            "",
            "| Method | Detections | " + " | ".join(objects) + " |",
        ]
        lines.append("|---|---|" + "---:|" * len(objects))
        for key, entry in report["poses"].items():
            method, condition = key.split("/")
            cells = " | ".join(_f(entry["by_object"].get(o, math.nan), 2) for o in objects)
            lines.append(f"| {METHODS[method]} | {CONDITIONS[condition]} | {cells} |")
        lines += [
            "",
            "### AR by visible fraction of the object",
            "",
            "| Method | Detections | " + " | ".join(VISIBILITY_LABELS) + " |",
            "|---|---|" + "---:|" * len(VISIBILITY_LABELS),
        ]
        for key, entry in report["poses"].items():
            method, condition = key.split("/")
            cells = " | ".join(
                f"{_f(entry['by_visibility'][b]['ar'], 2)} ({entry['by_visibility'][b]['n']})"
                for b in VISIBILITY_LABELS
            )
            lines.append(f"| {METHODS[method]} | {CONDITIONS[condition]} | {cells} |")
    if report["grasps"]:
        lines += [
            "",
            "## Grasp success",
            "",
            "A grasp planned from the pose estimate, judged on the true pose (see the protocol). "
            "Failures: no pose, no feasible grasp in the scene, collision with the object or the "
            "scene, fingers closing on nothing, or contacts outside the friction cone.",
            "",
            "| Method | Detections | Pose AR | Grasp success | Feasible plan | Failures |",
            "|---|---|---:|---:|---:|---|",
        ]
        for key, g in report["grasps"].items():
            if key == "oracle":
                name, cond, ar = "Oracle (true pose)", DASH, DASH
            else:
                method, condition = key.split("/")
                name, cond = METHODS[method], CONDITIONS[condition]
                ar = _f(report["poses"][key]["summary"]["ar"])
            reasons = ", ".join(
                f"{k} {v}"
                for k, v in sorted(g["reasons"].items(), key=lambda kv: -kv[1])
                if k != "ok"
            )
            lines.append(
                f"| {name} | {cond} | {ar} | **{_pct(g['success_rate'])}** "
                f"| {_pct(g['planned_rate'])} | {reasons} |"
            )
        lines += [
            "",
            "### Grasp success by pose error (all methods and detections pooled)",
            "",
            "| MSSD / diameter | Instances | Grasp success |",
            "|---|---:|---:|",
        ]
        for label, b in report["grasp_success_by_mssd"].items():
            if b["n"]:
                lines.append(f"| {label} | {b['n']} | {_pct(b['success_rate'])} |")
    return "\n".join(lines) + "\n"


def write(report: dict[str, Any], out: Path, *, charts: bool = True) -> list[Path]:
    """summary.json, results.md, bop/*.csv and (optionally) charts/*.png under `out`."""
    out.mkdir(parents=True, exist_ok=True)
    written = []
    data = {k: v for k, v in report.items() if k != "bop_files"}
    (out / "summary.json").write_text(json.dumps(data, indent=2, allow_nan=True), encoding="utf-8")
    (out / "results.md").write_text(markdown(report), encoding="utf-8")
    written += [out / "summary.json", out / "results.md"]
    for name, text in report["bop_files"].items():
        path = out / "bop" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    if charts:
        from posegrasp.charts import render_report

        written += render_report(report, out / "charts")
    return written
