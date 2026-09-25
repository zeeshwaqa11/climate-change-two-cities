from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.analysis import CityAnalysis, analyze_city
from src.clean import clean_file
from src.config import ROOT, City, last_complete_year
from src.fetch import download_city
from src.geocode import search_city

CACHE_DIR = ROOT / "app" / "cache"


def cache_stem(city: City) -> str:
    coordinates = f"{city.latitude:.2f}_{city.longitude:.2f}".replace("-", "m").replace(".", "p")
    return re.sub(r"[^a-z0-9_]+", "", f"{city.slug}_{coordinates}")


def cache_paths(city: City, directory: Path | None = None) -> tuple[Path, Path]:
    directory = directory or CACHE_DIR
    stem = cache_stem(city)
    return directory / f"{stem}.csv", directory / f"{stem}_meta.json"


def is_stale(meta_path: Path, config: dict[str, Any]) -> bool:
    if not meta_path.exists():
        return True
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("end_year") != last_complete_year(config) or meta.get("model") != config["data"]["model"]


def load_history(city: City, config: dict[str, Any], directory: Path | None = None) -> pd.DataFrame:
    csv_path, meta_path = cache_paths(city, directory)
    download_city(city, config, csv_path, meta_path, force=csv_path.exists() and is_stale(meta_path, config))
    frame, _ = clean_file(csv_path, config)
    return frame


def analyse_place(city: City, config: dict[str, Any], directory: Path | None = None) -> CityAnalysis:
    return analyze_city(load_history(city, config, directory), city, config)


def grid_cell(city: City, directory: Path | None = None) -> dict[str, Any]:
    _, meta_path = cache_paths(city, directory)
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def place_label(city: City) -> str:
    parts = [city.name, city.admin1, city.country]
    return ", ".join(part for part in parts if part)


def search_places(query: str, config: dict[str, Any], count: int = 8) -> list[City]:
    query = query.strip()
    if len(query) < 2:
        return []
    return search_city(query, config, count)
