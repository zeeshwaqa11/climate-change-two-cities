from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import City, get_cities, last_complete_year, load_config, resolve_path

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def build_params(city: City, config: dict[str, Any], end_year: int) -> dict[str, str | float]:
    return {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": config["data"]["start_date"],
        "end_date": f"{end_year}-12-31",
        "daily": ",".join(config["data"]["variables"]),
        "timezone": city.timezone,
        "models": config["data"]["model"],
    }


def request_with_retries(url: str, params: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    retries = config["fetch"]["retries"]
    backoff = config["fetch"]["backoff_seconds"]
    timeout = config["fetch"]["timeout_seconds"]
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code in RETRYABLE_STATUS:
                raise requests.HTTPError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            if attempt == retries:
                break
            wait = backoff * 2 ** (attempt - 1)
            print(f"  attempt {attempt}/{retries} failed ({error}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Request failed after {retries} attempts: {last_error}")


def payload_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def payload_metadata(payload: dict[str, Any], city: City, end_year: int, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "city": city.name,
        "requested_latitude": city.latitude,
        "requested_longitude": city.longitude,
        "grid_latitude": payload.get("latitude"),
        "grid_longitude": payload.get("longitude"),
        "elevation_m": payload.get("elevation"),
        "timezone": payload.get("timezone"),
        "units": payload.get("daily_units"),
        "end_year": end_year,
        "model": config["data"]["model"],
        "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
    }


def download_city(city: City, config: dict[str, Any], csv_path: Path, meta_path: Path, force: bool = False) -> Path:
    if csv_path.exists() and not force:
        print(f"{city.name}: {csv_path.name} exists, skipping (use --force to re-download)")
        return csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    end_year = last_complete_year(config)
    print(f"{city.name}: downloading {config['data']['start_date']} to {end_year}-12-31")
    payload = request_with_retries(config["data"]["archive_url"], build_params(city, config, end_year), config)
    frame = payload_to_frame(payload)
    frame.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    meta_path.write_text(json.dumps(payload_metadata(payload, city, end_year, config), indent=2), encoding="utf-8")
    print(f"{city.name}: saved {len(frame):,} rows to {csv_path.name}")
    return csv_path


def fetch_city(city: City, config: dict[str, Any], force: bool = False) -> Path:
    raw_dir = resolve_path(config, "raw")
    return download_city(city, config, raw_dir / f"{city.slug}.csv", raw_dir / f"{city.slug}_meta.json", force)


def fetch_all(config: dict[str, Any] | None = None, force: bool = False, only: str | None = None) -> list[Path]:
    config = config or load_config()
    cities = get_cities(config)
    if only:
        cities = [city for city in cities if city.name.lower() == only.lower()]
        if not cities:
            raise SystemExit(f"City '{only}' is not defined in config.yaml")
    return [fetch_city(city, config, force) for city in cities]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily ERA5 weather history from Open-Meteo")
    parser.add_argument("--force", action="store_true", help="re-download even if the raw file exists")
    parser.add_argument("--city", help="fetch only this city from config.yaml")
    args = parser.parse_args()
    fetch_all(force=args.force, only=args.city)


if __name__ == "__main__":
    main()
