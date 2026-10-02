"""Command-line entry point: `posegrasp data | info | ...`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from posegrasp.config import get_settings

app = typer.Typer(add_completion=False, help="6-DoF pose estimation to grasping on BOP LM-O.")


@app.callback()
def _setup() -> None:
    """6-DoF pose estimation to grasping on BOP LM-O."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


DataDirOpt = Annotated[Path | None, typer.Option(help="Defaults to POSEGRASP_DATA_DIR")]


@app.command("data")
def data(data_dir: DataDirOpt = None) -> None:
    """Downloads LM-O (test BOP'19 subset, models) and the BOP'23 default detections (~140 MB)."""
    from posegrasp.data.download import prepare

    paths = prepare(data_dir or get_settings().data_dir)
    typer.echo(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


@app.command("info")
def info(data_dir: DataDirOpt = None) -> None:
    """Counts: objects, targets, instances, and detections available per detector."""
    from posegrasp.data import detections as det
    from posegrasp.data.bop import Dataset
    from posegrasp.data.download import DETECTION_ASSETS

    settings = get_settings()
    root = (data_dir or settings.data_dir) / settings.dataset
    ds = Dataset(root)
    targets = ds.targets()
    models = ds.models()
    summary: dict[str, object] = {
        "objects": {
            m.obj_id: {"diameter_mm": m.diameter, "symmetric": m.is_symmetric}
            for m in models.values()
        },
        "images": len({(t.scene_id, t.im_id) for t in targets}),
        "targets": len(targets),
        "instances": sum(t.inst_count for t in targets),
    }
    for name, asset in DETECTION_ASSETS.items():
        path = (data_dir or settings.data_dir) / "detections" / asset.filename
        if path.exists():
            chosen = det.for_targets(det.load(path), targets)
            summary[name] = {
                "instances_with_a_detection": sum(len(v) for v in chosen.values()),
                "masks": any(d.has_mask for v in chosen.values() for d in v),
            }
    typer.echo(json.dumps(summary, indent=2))


def _dev_split(targets: list, every: int = 10) -> tuple[list, list]:  # type: ignore[type-arg]
    """Development images (every `every`-th targeted image) and the rest."""
    images = sorted({(t.scene_id, t.im_id) for t in targets})
    dev = set(images[::every])
    return [t for t in targets if (t.scene_id, t.im_id) in dev], targets


def _select(targets: list, subset: str, shard: str | None) -> tuple[list, str]:  # type: ignore[type-arg]
    """Targets of the subset (dev | all), optionally one shard of its images, and a run name."""
    if subset not in {"dev", "all"}:
        raise typer.BadParameter("subset must be dev or all")
    dev, all_targets = _dev_split(targets)
    chosen = dev if subset == "dev" else all_targets
    if shard is None:
        return chosen, subset
    i, n = (int(v) for v in shard.split("/"))
    keep = set(sorted({(t.scene_id, t.im_id) for t in chosen})[i::n])
    return [t for t in chosen if (t.scene_id, t.im_id) in keep], f"{subset}-shard-{i}-of-{n}"


@app.command("eval")
def evaluate(
    estimator: Annotated[str, typer.Option(help="fpfh | ppf")] = "fpfh",
    condition: Annotated[str, typer.Option(help="gt | gdrnpp | cnos")] = "gt",
    subset: Annotated[str, typer.Option(help="dev (20 images) | all (200 images)")] = "dev",
    selection: Annotated[str, typer.Option(help="score | verify (depth verification)")] = "score",
    restarts: Annotated[int, typer.Option(help="FPFH: RANSAC restarts (hypotheses)")] = 1,
    shard: Annotated[str | None, typer.Option(help="INDEX/COUNT of the targeted images")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N targets")] = None,
    data_dir: DataDirOpt = None,
    results_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Estimates poses for the targets and scores them; resumes an interrupted run."""
    from posegrasp.data.bop import Dataset
    from posegrasp.estimators import create_estimator
    from posegrasp.pipeline import CONDITIONS, evaluate_targets, load_rows, summarize_rows

    if condition not in CONDITIONS:
        raise typer.BadParameter(f"condition must be one of {sorted(CONDITIONS)}")
    settings = get_settings()
    base = data_dir or settings.data_dir
    ds = Dataset(base / settings.dataset)
    targets, name = _select(ds.targets(), subset, shard)
    if limit is not None:
        targets = targets[:limit]
    method = f"{estimator}-{selection}" + (f"-r{restarts}" if estimator == "fpfh" else "")
    out = (results_dir or settings.results_dir) / method / condition / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_rows(out)
    done = {(r["scene_id"], r["im_id"], r["obj_id"]) for r in rows}
    todo = [t for t in targets if (t.scene_id, t.im_id, t.obj_id) not in done]
    typer.echo(f"{estimator}/{condition}: {len(targets)} targets, {len(todo)} to go -> {out}")
    with out.open("a", encoding="utf-8") as f:
        kwargs = {"restarts": restarts} if estimator == "fpfh" else {}
        rows_iter = evaluate_targets(
            ds,
            todo,
            estimator=create_estimator(estimator, **kwargs),
            condition=condition,
            detections_dir=base / "detections",
            selection=selection,
        )
        for row in rows_iter:
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
    typer.echo(json.dumps(summarize_rows(rows), indent=2))


@app.command("grasps")
def grasps(data_dir: DataDirOpt = None) -> None:
    """Synthesises antipodal grasps for every object model once (cached in DATA_DIR/grasps)."""
    from posegrasp.data.bop import Dataset
    from posegrasp.picking import grasp_sets

    settings = get_settings()
    base = data_dir or settings.data_dir
    sets = grasp_sets(Dataset(base / settings.dataset).models(), base / "grasps", log=True)
    typer.echo(json.dumps({obj_id: len(s) for obj_id, s in sets.items()}))


@app.command("pick")
def pick(
    method: Annotated[
        str, typer.Option(help="Results folder of a pose method, or 'oracle' (the true pose)")
    ] = "ppf-verify",
    condition: Annotated[str, typer.Option(help="gt | gdrnpp | cnos")] = "gt",
    subset: Annotated[str, typer.Option(help="dev | all")] = "dev",
    shard: Annotated[str | None, typer.Option(help="INDEX/COUNT of the targeted images")] = None,
    data_dir: DataDirOpt = None,
    results_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Plans a grasp from each pose estimate and judges it on the true pose; resumable."""
    from posegrasp.data.bop import Dataset
    from posegrasp.picking import evaluate_grasps, grasp_sets, summarize_grasps
    from posegrasp.pipeline import load_rows

    settings = get_settings()
    base = data_dir or settings.data_dir
    root = results_dir or settings.results_dir
    ds = Dataset(base / settings.dataset)
    targets, name = _select(ds.targets(), subset, shard)
    oracle = method == "oracle"
    if oracle:
        rows = [
            {"scene_id": t.scene_id, "im_id": t.im_id, "obj_id": t.obj_id}
            for t in targets
            for _ in range(t.inst_count)
        ]
        out = root / "oracle" / f"{name}.grasp.jsonl"
    else:
        rows = load_rows(root / method / condition / f"{name}.jsonl")
        if not rows:
            raise typer.BadParameter(f"no pose results in {root / method / condition}/{name}.jsonl")
        out = root / method / condition / f"{name}.grasp.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done_rows = load_rows(out)
    done = {(r["scene_id"], r["im_id"], r["obj_id"]) for r in done_rows}
    todo = [r for r in rows if (r["scene_id"], r["im_id"], r["obj_id"]) not in done]
    typer.echo(f"pick {method}/{condition}: {len(rows)} poses, {len(todo)} to go -> {out}")
    sets = grasp_sets(ds.models(), base / "grasps", log=True)
    with out.open("a", encoding="utf-8") as f:
        for row in evaluate_grasps(ds, todo, grasps=sets, oracle=oracle):
            f.write(json.dumps(row) + "\n")
            f.flush()
            done_rows.append(row)
    typer.echo(json.dumps(summarize_grasps(done_rows), indent=2))


if __name__ == "__main__":
    app()
