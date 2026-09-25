"""Runtime settings, overridable with POSEGRASP_* environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSEGRASP_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    dataset: str = "lmo"

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / self.dataset

    @property
    def detections_dir(self) -> Path:
        return self.data_dir / "detections"


def get_settings() -> Settings:
    return Settings()
