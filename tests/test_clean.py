from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src import clean
from src.config import City, last_complete_year


def raw_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def test_reindex_daily_fills_gaps_and_removes_duplicates():
    frame = raw_frame(
        [
            {"date": "2000-01-01", "tmax": 1.0},
            {"date": "2000-01-01", "tmax": 9.0},
            {"date": "2000-01-04", "tmax": 4.0},
        ]
    )
    result = clean.reindex_daily(frame)
    assert len(result) == 4
    assert result["date"].is_unique
    assert result["tmax"].isna().sum() == 2


def test_trim_to_complete_years_drops_partial_year():
    frame = pd.DataFrame({"date": pd.date_range("2024-12-30", "2026-01-02")})
    trimmed = clean.trim_to_complete_years(frame, 2025)
    assert trimmed["date"].max() == pd.Timestamp("2025-12-31")


def test_mask_impossible_values_flags_swapped_temperatures_and_negatives():
    frame = pd.DataFrame(
        {
            "tmax": [20.0, 10.0, 25.0],
            "tmin": [10.0, 15.0, 15.0],
            "tmean": [15.0, 12.0, 20.0],
            "precip": [0.0, 1.0, -2.0],
            "wind": [5.0, 5.0, -1.0],
        }
    )
    cleaned, counts = clean.mask_impossible_values(frame)
    assert counts == {"tmin_above_tmax": 1, "negative_precip": 1, "negative_wind": 1}
    assert cleaned.loc[1, ["tmax", "tmin", "tmean"]].isna().all()
    assert np.isnan(cleaned.loc[2, "precip"])
    assert cleaned.loc[0, "tmax"] == 20.0


def test_add_features_creates_expected_columns():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2000-02-29", "2001-12-31"]),
            "tmax": [20.0, 10.0],
            "tmin": [12.0, 4.0],
            "tmean": [16.0, 7.0],
            "precip": [1.0, 0.5],
            "wind": [10.0, 12.0],
        }
    )
    result = clean.add_features(frame, wet_day_mm=1.0)
    assert result["year"].tolist() == [2000, 2001]
    assert result["month"].tolist() == [2, 12]
    assert result["day_of_year"].tolist() == [60, 365]
    assert result["dtr"].tolist() == [8.0, 6.0]
    assert result["is_wet_day"].tolist() == [True, False]


def test_last_complete_year_uses_previous_calendar_year():
    config = {"data": {"end_year": None}}
    assert last_complete_year(config, today=dt.date(2026, 9, 26)) == 2025
    assert last_complete_year(config, today=dt.date(2027, 1, 1)) == 2026


def test_last_complete_year_respects_configured_override():
    assert last_complete_year({"data": {"end_year": 2020}}, today=dt.date(2026, 9, 26)) == 2020


def test_city_heat_threshold_falls_back_to_global_default():
    config = {"thresholds": {"extreme_heat_fixed_c": 40.0}}
    assert City("A", "X", 0, 0).heat_threshold(config) == 40.0
    assert City("B", "X", 0, 0, extreme_heat_fixed_c=30.0).heat_threshold(config) == 30.0


def test_city_slug():
    assert City("New York", "US", 0, 0).slug == "new_york"
