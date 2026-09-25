from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import City, get_cities, last_complete_year, load_config, resolve_path

COLUMN_MAP = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "temperature_2m_mean": "tmean",
    "precipitation_sum": "precip",
    "wind_speed_10m_max": "wind",
}

DAILY_COLUMNS = ["date", "tmax", "tmin", "tmean", "precip", "wind", "year", "month", "day_of_year", "dtr", "is_wet_day"]


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).rename(columns=COLUMN_MAP)


def reindex_daily(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.drop_duplicates(subset="date").sort_values("date").set_index("date")
    full_range = pd.date_range(frame.index.min(), frame.index.max(), freq="D", name="date")
    return frame.reindex(full_range).reset_index()


def trim_to_complete_years(frame: pd.DataFrame, end_year: int) -> pd.DataFrame:
    return frame[frame["date"].dt.year <= end_year].reset_index(drop=True)


def mask_impossible_values(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = frame.copy()
    swapped = frame["tmin"] > frame["tmax"]
    negative_precip = frame["precip"] < 0
    negative_wind = frame["wind"] < 0
    frame.loc[swapped, ["tmax", "tmin", "tmean"]] = float("nan")
    frame.loc[negative_precip, "precip"] = float("nan")
    frame.loc[negative_wind, "wind"] = float("nan")
    counts = {
        "tmin_above_tmax": int(swapped.sum()),
        "negative_precip": int(negative_precip.sum()),
        "negative_wind": int(negative_wind.sum()),
    }
    return frame, counts


def add_features(frame: pd.DataFrame, wet_day_mm: float) -> pd.DataFrame:
    frame = frame.copy()
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["day_of_year"] = frame["date"].dt.dayofyear
    frame["dtr"] = frame["tmax"] - frame["tmin"]
    frame["is_wet_day"] = frame["precip"] >= wet_day_mm
    return frame


def quality_report(raw: pd.DataFrame, cleaned: pd.DataFrame, issues: dict[str, int]) -> dict[str, Any]:
    return {
        "rows_raw": len(raw),
        "rows_clean": len(cleaned),
        "first_date": cleaned["date"].min().date().isoformat(),
        "last_date": cleaned["date"].max().date().isoformat(),
        "duplicate_dates": int(raw["date"].duplicated().sum()),
        "missing_values": {col: int(cleaned[col].isna().sum()) for col in COLUMN_MAP.values()},
        **issues,
    }


def clean_city(city: City, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_path = resolve_path(config, "raw") / f"{city.slug}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run `python -m src.fetch` first.")
    raw = load_raw(raw_path)
    frame = reindex_daily(raw)
    frame = trim_to_complete_years(frame, last_complete_year(config))
    frame, issues = mask_impossible_values(frame)
    frame = add_features(frame, config["thresholds"]["wet_day_mm"])
    return frame[DAILY_COLUMNS], quality_report(raw, frame, issues)


def processed_path(city: City, config: dict[str, Any]) -> Path:
    return resolve_path(config, "processed") / f"{city.slug}_daily.csv"


def clean_all(config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    config = config or load_config()
    reports = {}
    for city in get_cities(config):
        frame, report = clean_city(city, config)
        frame.to_csv(processed_path(city, config), index=False, date_format="%Y-%m-%d", float_format="%.2f")
        reports[city.name] = report
        print(f"{city.name}: {report['rows_clean']:,} rows, missing {report['missing_values']}")
    return reports


def load_daily(city: City, config: dict[str, Any]) -> pd.DataFrame:
    return pd.read_csv(processed_path(city, config), parse_dates=["date"])


def main() -> None:
    argparse.ArgumentParser(description="Clean raw daily data and add feature columns").parse_args()
    clean_all()


if __name__ == "__main__":
    main()
