from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


@dataclass(frozen=True)
class City:
    name: str
    country: str
    latitude: float
    longitude: float
    timezone: str = "auto"

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "_")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_cities(config: dict[str, Any]) -> list[City]:
    return [City(**entry) for entry in config["cities"]]


def last_complete_year(config: dict[str, Any], today: dt.date | None = None) -> int:
    configured = config["data"].get("end_year")
    if configured:
        return int(configured)
    today = today or dt.date.today()
    return today.year - 1


def resolve_path(config: dict[str, Any], key: str) -> Path:
    path = ROOT / config["paths"][key]
    path.mkdir(parents=True, exist_ok=True)
    return path
