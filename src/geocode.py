from __future__ import annotations

import argparse
from typing import Any

import requests

from src.config import City, load_config


def search_city(name: str, config: dict[str, Any], count: int = 5) -> list[City]:
    response = requests.get(
        config["data"]["geocoding_url"],
        params={"name": name, "count": count, "language": "en", "format": "json"},
        timeout=config["fetch"]["timeout_seconds"],
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    return [
        City(
            name=item["name"],
            country=item.get("country", ""),
            latitude=round(item["latitude"], 5),
            longitude=round(item["longitude"], 5),
            timezone=item.get("timezone", "auto"),
        )
        for item in results
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Look up city coordinates with the Open-Meteo Geocoding API")
    parser.add_argument("name")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    for city in search_city(args.name, load_config(), args.count):
        print(f"- name: {city.name}")
        print(f"  country: {city.country}")
        print(f"  latitude: {city.latitude}")
        print(f"  longitude: {city.longitude}")
        print(f"  timezone: {city.timezone}")


if __name__ == "__main__":
    main()
