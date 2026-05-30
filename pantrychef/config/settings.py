"""Runtime settings, sourced from environment / .env.

Why pydantic-settings: typed, validated, env-overridable config without
scattering os.getenv calls. Path defaults assume the repo layout but can be
overridden (e.g. PANTRYCHEF_DATA_DIR on Colab/Kaggle).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application-wide runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="PANTRYCHEF_",
        env_file=".env",
        extra="ignore",
    )

    # Paths
    repo_root: Path = _REPO_ROOT
    data_dir: Path = Field(default=_REPO_ROOT / "data")
    models_dir: Path = Field(default=_REPO_ROOT / "models")
    experiments_dir: Path = Field(default=_REPO_ROOT / "experiments")

    # Tracking
    mlflow_tracking_uri: str = Field(
        default=f"file:{(_REPO_ROOT / 'experiments' / 'mlruns').as_posix()}"
    )

    # Logging
    log_level: str = "INFO"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton accessor."""
    return Settings()
