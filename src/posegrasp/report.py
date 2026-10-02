"""Benchmark report: merges the result files of all runs (and shards) into tables, charts and BOP
submission files.

Layout read (written by `posegrasp eval` and `posegrasp pick`):
    <results>/<method>/<condition>/<subset>[-shard-i-of-n].jsonl        pose rows
    <results>/<method>/<condition>/<subset>[-shard-i-of-n].grasp.jsonl  grasp rows
    <results>/oracle/<subset>[-shard-i-of-n].grasp.jsonl                 grasp rows, true pose
    ...grasp-nominal.jsonl                                               nominal ranking ablation
    ...sim.jsonl                                                         MuJoCo outcome per grasp
    <latency>/<method>/gt/dev.jsonl                                      pose rows, one machine
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
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


def merged_rows(folder: Path, subset: str, suffix: str) -> Rows:
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


def summarize_physics(rows: Rows) -> dict[str, Any]:
    """MuJoCo outcomes against the geometric judgement of the same grasps."""
    trials = [r for r in rows if r["physics_reason"] != "unstable_scene"]
    n = len(trials)
    both = sum(bool(r["success"] and r["physics_success"]) for r in trials)
    geometric_only = sum(bool(r["success"] and not r["physics_success"]) for r in trials)
    physics_only = sum(bool(r["physics_success"] and not r["success"]) for r in trials)
    neither = n - both - geometric_only - physics_only
    return {
        "trials": n,
        "unstable_scenes": len(rows) - n,
        "physics_success_rate": (both + physics_only) / n if n else math.nan,
        "geometric_success_rate": (both + geometric_only) / n if n else math.nan,
        "agreement": (both + neither) / n if n else math.nan,
        "confusion": {
            "both": both,
            "geometric_only": geometric_only,
            "physics_only": physics_only,
            "neither": neither,
        },
        "reasons": dict(Counter(r["physics_reason"] for r in rows)),
    }


def _by_error(rows: Rows) -> dict[str, dict[str, float]]:
    out = {}
    for label in ERROR_LABELS:
        chosen = [r for r in rows if error_bin(r) == label]
        rate = sum(r["success"] for r in chosen) / len(chosen) if chosen else math.nan
        out[label] = {"n": len(chosen), "success_rate": rate}
    return out


def build(results: Path, *, subset: str = "all", latency: Path | None = None) -> dict[str, Any]:
    """All numbers of the report, as one JSON-ready dict."""
    poses: dict[str, Any] = {}
    grasps: dict[str, Any] = {}
    nominal: dict[str, Any] = {}
    physics: dict[str, Any] = {}
    pooled_grasp_rows: Rows = []
    pooled_nominal_rows: Rows = []
    bop_files: dict[str, str] = {}
    for method in METHODS:
        for condition in CONDITIONS:
            folder = results / method / condition
            rows = merged_rows(folder, subset, "jsonl")
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
            grasp_rows = merged_rows(folder, subset, "grasp.jsonl")
            if grasp_rows:
                grasps[key] = summarize_grasps(grasp_rows)
                pooled_grasp_rows += grasp_rows
            nominal_rows = merged_rows(folder, subset, "grasp-nominal.jsonl")
            if nominal_rows:
                nominal[key] = summarize_grasps(nominal_rows)
                pooled_nominal_rows += nominal_rows
            sim_rows = merged_rows(folder, subset, "sim.jsonl")
            if sim_rows:
                physics[key] = summarize_physics(sim_rows)
    oracle = results / "oracle"
    for suffix, table in (("grasp.jsonl", grasps), ("grasp-nominal.jsonl", nominal)):
        oracle_rows = merged_rows(oracle, subset, suffix)
        if oracle_rows:
            table["oracle"] = summarize_grasps(oracle_rows)
    oracle_sim = merged_rows(oracle, subset, "sim.jsonl")
    if oracle_sim:
        physics["oracle"] = summarize_physics(oracle_sim)
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
        "grasps_nominal": nominal,
        "grasp_success_by_mssd": _by_error(pooled_grasp_rows),
        "grasp_success_by_mssd_nominal": _by_error(pooled_nominal_rows),
        "physics": physics,
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


def _labels(key: str) -> tuple[str, str]:
    """Method and detection names of a report key ("method/condition" or "oracle")."""
    if key == "oracle":
        return "Oracle (true pose)", DASH
    method, condition = key.split("/")
    return METHODS[method], CONDITIONS[condition]


def _pose_section(report: dict[str, Any]) -> list[str]:
    lines = [
        "## Pose estimation",
        "",
        "| Method | Detections | AR | AR_VSD | AR_MSSD | AR_MSPD | ADD(-S) < 0.1 d | Detected |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in report["poses"].items():
        s = entry["summary"]
        name, cond = _labels(key)
        lines.append(
            f"| {name} | {cond} | **{_f(s['ar'])}** | {_f(s['ar_vsd'])} | {_f(s['ar_mssd'])} "
            f"| {_f(s['ar_mspd'])} | {_pct(s['add_s_01d'])} | {_pct(s['detected'])} |"
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
    if not objects:
        return lines
    lines += [
        "",
        "### AR per object",
        "",
        "| Method | Detections | " + " | ".join(objects) + " |",
        "|---|---|" + "---:|" * len(objects),
    ]
    for key, entry in report["poses"].items():
        cells = " | ".join(_f(entry["by_object"].get(o, math.nan), 2) for o in objects)
        lines.append("| {} | {} | ".format(*_labels(key)) + cells + " |")
    lines += [
        "",
        "### AR by visible fraction of the object",
        "",
        "| Method | Detections | " + " | ".join(VISIBILITY_LABELS) + " |",
        "|---|---|" + "---:|" * len(VISIBILITY_LABELS),
    ]
    for key, entry in report["poses"].items():
        bins = entry["by_visibility"]
        cells = " | ".join(f"{_f(bins[b]['ar'], 2)} ({bins[b]['n']})" for b in VISIBILITY_LABELS)
        lines.append("| {} | {} | ".format(*_labels(key)) + cells + " |")
    return lines


def _grasp_section(report: dict[str, Any]) -> list[str]:
    lines = [
        "## Grasp success",
        "",
        "A grasp planned from the pose estimate, judged on the true pose (see the protocol). "
        "Failures: no pose, no feasible grasp in the scene, collision with the object or the "
        "scene, fingers closing on nothing, or contacts outside the friction cone.",
        "",
        "Grasps are ranked by their robustness to pose errors; the *nominal* column ranks them "
        "ignoring pose errors (the ablation).",
        "",
        "| Method | Detections | Pose AR | Grasp success | Nominal ranking | Feasible plan "
        "| Failures |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    nominal = report.get("grasps_nominal", {})
    for key, g in report["grasps"].items():
        name, cond = _labels(key)
        ar = DASH if key == "oracle" else _f(report["poses"][key]["summary"]["ar"])
        failures = sorted(g["reasons"].items(), key=lambda kv: -kv[1])
        reasons = ", ".join(f"{k} {v}" for k, v in failures if k != "ok")
        ablation = nominal.get(key, {}).get("success_rate", math.nan)
        lines.append(
            f"| {name} | {cond} | {ar} | **{_pct(g['success_rate'])}** | {_pct(ablation)} "
            f"| {_pct(g['planned_rate'])} | {reasons} |"
        )
    lines += [
        "",
        "### Grasp success by pose error (all methods and detections pooled)",
        "",
        "| MSSD / diameter | Instances | Grasp success | Nominal ranking |",
        "|---|---:|---:|---:|",
    ]
    nominal_bins = report.get("grasp_success_by_mssd_nominal", {})
    for label, b in report["grasp_success_by_mssd"].items():
        if b["n"]:
            other = nominal_bins.get(label, {}).get("success_rate", math.nan)
            lines.append(f"| {label} | {b['n']} | {_pct(b['success_rate'])} | {_pct(other)} |")
    return lines


def _physics_section(report: dict[str, Any]) -> list[str]:
    lines = [
        "## Physics check (MuJoCo)",
        "",
        "The planned grasps executed with the Franka Hand in MuJoCo, objects at their true poses "
        "(see the protocol). Agreement: the share of trials where the geometric judgement and the "
        "simulation give the same outcome.",
        "",
        "| Method | Detections | Trials | Physics success | Geometric success | Agreement "
        "| Geometric only | Physics only |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, ph in report["physics"].items():
        name, cond = _labels(key)
        c = ph["confusion"]
        lines.append(
            f"| {name} | {cond} | {ph['trials']} | **{_pct(ph['physics_success_rate'])}** "
            f"| {_pct(ph['geometric_success_rate'])} | {_pct(ph['agreement'])} "
            f"| {c['geometric_only']} | {c['physics_only']} |"
        )
    return lines


def markdown(report: dict[str, Any]) -> str:
    """The report tables as GitHub Markdown."""
    lines = [
        "# Benchmark results",
        "",
        f"LM-O (BOP'19 test subset), `{report['subset']}` images. Generated by `posegrasp report`; "
        "the protocol is in [docs/EVAL_PROTOCOL.md](../../docs/EVAL_PROTOCOL.md).",
        "",
        *_pose_section(report),
    ]
    if report["grasps"]:
        lines += ["", *_grasp_section(report)]
    if report.get("physics"):
        lines += ["", *_physics_section(report)]
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
