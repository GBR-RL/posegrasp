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


if __name__ == "__main__":
    app()
