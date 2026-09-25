from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import interactive, live
from src.analysis import analyze_city
from src.config import City, load_config
from tests.helpers import make_daily

CITY = City("New York", "United States", 40.71427, -74.00597, timezone="America/New_York", admin1="New York")


def raw_payload(config: dict) -> dict:
    rng = np.random.default_rng(3)
    dates = pd.date_range("1940-01-01", "2025-12-31", freq="D")
    tmean = 12 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + 0.02 * (dates.year - 1940) + rng.normal(0, 2, len(dates))
    return {
        "latitude": 40.7,
        "longitude": -74.0,
        "elevation": 10.0,
        "timezone": "America/New_York",
        "daily_units": {},
        "daily": {
            "time": [date.strftime("%Y-%m-%d") for date in dates],
            "temperature_2m_max": (tmean + 5).round(1).tolist(),
            "temperature_2m_min": (tmean - 5).round(1).tolist(),
            "temperature_2m_mean": tmean.round(1).tolist(),
            "precipitation_sum": (rng.exponential(2, len(dates)) * (rng.random(len(dates)) < 0.4)).round(1).tolist(),
            "wind_speed_10m_max": rng.uniform(5, 30, len(dates)).round(1).tolist(),
        },
    }


@pytest.fixture()
def config():
    return load_config()


def test_cache_stem_is_filesystem_safe_and_coordinate_specific():
    stem = live.cache_stem(CITY)
    assert stem == "new_york_40p71_m74p01"
    other = City("New York", "United States", 43.0, -75.5)
    assert live.cache_stem(other) != stem


def test_place_label_skips_empty_parts():
    assert live.place_label(CITY) == "New York, New York, United States"
    assert live.place_label(City("Atlantis", "", 0.0, 0.0)) == "Atlantis"


def test_search_places_ignores_short_queries(config, monkeypatch):
    monkeypatch.setattr(live, "search_city", lambda *args, **kwargs: pytest.fail("should not call the API"))
    assert live.search_places(" a ", config) == []


def test_load_history_downloads_once_then_uses_cache(config, tmp_path, monkeypatch):
    calls = []

    def fake_request(url, params, cfg):
        calls.append(params)
        return raw_payload(cfg)

    monkeypatch.setattr("src.fetch.request_with_retries", fake_request)
    first = live.load_history(CITY, config, tmp_path)
    second = live.load_history(CITY, config, tmp_path)
    assert len(calls) == 1
    assert calls[0]["models"] == "era5"
    assert len(first) == len(second) == 31412
    assert first["date"].max() == pd.Timestamp("2025-12-31")
    assert live.grid_cell(CITY, tmp_path)["model"] == "era5"


def test_stale_cache_is_refreshed(config, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("src.fetch.request_with_retries", lambda url, params, cfg: calls.append(1) or raw_payload(cfg))
    live.load_history(CITY, config, tmp_path)
    _, meta_path = live.cache_paths(CITY, tmp_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["end_year"] = 2019
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert live.is_stale(meta_path, config)
    live.load_history(CITY, config, tmp_path)
    assert len(calls) == 2


def test_is_stale_when_meta_missing(config, tmp_path):
    assert live.is_stale(tmp_path / "nothing.json", config)


@pytest.fixture(scope="module")
def analysis():
    config = load_config()
    rng = np.random.default_rng(5)
    daily = make_daily(1940, 2025)
    seasonal = 8 * np.sin(2 * np.pi * daily["day_of_year"] / 365.25)
    daily["tmean"] = 15 + seasonal + 0.03 * (daily["year"] - 1940) + rng.normal(0, 2, len(daily))
    daily["tmax"] = daily["tmean"] + 5
    daily["tmin"] = daily["tmean"] - 5
    daily["dtr"] = 10.0
    daily["precip"] = rng.exponential(1.5, len(daily))
    return analyze_city(daily, City("Testville", "Nowhere", 0.0, 0.0, extreme_heat_fixed_c=25.0), config)


def test_stripes_figure_has_one_bar_per_year(analysis):
    fig = interactive.stripes_figure(analysis)
    heatmap = fig.data[0]
    assert len(heatmap.x) == 86
    assert heatmap.zmid == 0
    assert heatmap.zmin == -heatmap.zmax


def test_stripes_colorscale_spans_blue_to_red():
    scale = interactive.stripes_colorscale()
    assert scale[0][0] == 0 and scale[-1][0] == 1
    assert scale[0][1] != scale[-1][1]


def test_anomaly_figure_has_bars_and_trend_line(analysis):
    fig = interactive.anomaly_figure(analysis, 1960)
    assert [trace.type for trace in fig.data] == ["bar", "scatter"]
    assert "/decade" in fig.data[1].name


@pytest.mark.parametrize("metric", ["hot_days_pct", "hot_days_fixed"])
def test_heat_figure_has_bars_and_rolling_line(analysis, metric):
    fig = interactive.heat_figure(analysis, metric, 10)
    assert [trace.type for trace in fig.data] == ["bar", "scatter"]
    assert fig.layout.title.text.startswith("Days above")


def test_suggest_threshold_is_rounded_baseline_percentile(config):
    daily = make_daily(1961, 1990)
    rng = np.random.default_rng(9)
    daily["tmax"] = rng.normal(30, 4, len(daily))
    expected = round(np.percentile(daily["tmax"], 99))
    assert live.suggest_threshold(daily, config) == expected
    assert isinstance(live.suggest_threshold(daily, config), float)
