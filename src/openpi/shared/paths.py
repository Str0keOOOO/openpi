"""Locations shared by RoboCasa training, data preparation and evaluation."""

import os
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def workspace_root() -> Path:
    return Path(os.environ.get("ROBOCASA365_ROOT", repository_root().parent)).expanduser().resolve()


def workspace_path(*parts: str) -> Path:
    return workspace_root().joinpath(*parts)
