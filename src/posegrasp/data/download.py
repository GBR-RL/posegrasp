"""Downloads LM-O and the BOP'23 default detections from the BOP Hugging Face mirror."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

HF = "https://huggingface.co/datasets/bop-benchmark"
DETECTIONS = f"{HF}/bop_extra/resolve/main/default_detections/classic_bop23_model_based"


@dataclass(frozen=True, slots=True)
class Asset:
    url: str
    filename: str
    size: int  # bytes, checked after download


LMO_ASSETS = (
    Asset(f"{HF}/lmo/resolve/main/lmo_base.zip", "lmo_base.zip", 5_684),
    Asset(f"{HF}/lmo/resolve/main/lmo_models.zip", "lmo_models.zip", 5_367_178),
    Asset(f"{HF}/lmo/resolve/main/lmo_test_bop19.zip", "lmo_test_bop19.zip", 117_550_985),
)

# Default 2D detections of the BOP Challenge 2023, used as given (not re-trained here).
DETECTION_ASSETS = {
    # GDRNPP detector trained on the synthetic (PBR) training images of the LM-O objects: boxes
    "gdrnppdet-pbr": Asset(
        f"{DETECTIONS}_seen/gdrnppdet-pbr/"
        "gdrnppdet-pbr_lmo-test_468338cf-4416-4142-9bba-34e5fba4eb86.json",
        "det_gdrnppdet-pbr_lmo.json",
        5_881_760,
    ),
    # CNOS: zero-shot segmentation of objects never trained on (FastSAM + DINOv2): masks
    "cnos-fastsam": Asset(
        f"{DETECTIONS}_unseen/cnos-fastsam/"
        "cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json",
        "det_cnos-fastsam_lmo.json",
        12_183_654,
    ),
}


def download(asset: Asset, dest_dir: Path) -> Path:
    """Downloads once; a file of the expected size is kept."""
    dest = dest_dir / asset.filename
    if dest.exists() and dest.stat().st_size == asset.size:
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(asset.url, timeout=120) as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if partial.stat().st_size != asset.size:
        raise OSError(
            f"{asset.filename}: expected {asset.size} bytes, got {partial.stat().st_size}"
        )
    partial.replace(dest)
    return dest


def extract(archive: Path, dataset_dir: Path) -> None:
    """Extracts into the dataset folder. The base archive carries a top-level `lmo/`, the others
    do not; members are placed relative to the dataset folder either way."""
    prefix = dataset_dir.name + "/"
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = member.filename.removeprefix(prefix)
            if not name or member.is_dir():
                continue
            target = dataset_dir / name
            if target.exists() and target.stat().st_size == member.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)


def prepare(data_dir: Path, dataset: str = "lmo") -> dict[str, Path]:
    """Downloads and extracts the dataset and the default detections. Returns their paths."""
    raw = data_dir / "raw"
    dataset_dir = data_dir / dataset
    for asset in LMO_ASSETS:
        extract(download(asset, raw), dataset_dir)
    paths = {"dataset": dataset_dir}
    detections_dir = data_dir / "detections"
    detections_dir.mkdir(parents=True, exist_ok=True)
    for name, asset in DETECTION_ASSETS.items():
        path = download(asset, raw)
        shutil.copyfile(path, detections_dir / asset.filename)
        paths[name] = detections_dir / asset.filename
    return paths
