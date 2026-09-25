from __future__ import annotations

import numpy as np
import pytest

from src import analysis, report
from src.config import City, load_config
from tests.helpers import make_daily


@pytest.fixture(scope="module")
def results():
    config = load_config()
    rng = np.random.default_rng(21)
    output = {}
    for index, name in enumerate(["Alpha", "Beta"]):
        daily = make_daily(1940, 2025)
        seasonal = 8 * np.sin(2 * np.pi * daily["day_of_year"] / 365.25)
        daily["tmean"] = 15 + seasonal + 0.03 * (daily["year"] - 1940) * (index + 1) + rng.normal(0, 2, len(daily))
        daily["tmax"] = daily["tmean"] + 5
        daily["tmin"] = daily["tmean"] - 5
        daily["dtr"] = daily["tmax"] - daily["tmin"]
        daily["precip"] = rng.exponential(1.5, len(daily)) * (rng.random(len(daily)) < 0.4)
        output[name] = analysis.analyze_city(daily, City(name, "Nowhere", 0.0, 0.0, extreme_heat_fixed_c=25.0), config)
    return output


def test_markdown_table_shape():
    table = report.markdown_table(["a", "b"], [["1", "2"], ["3", "4"]])
    lines = table.splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "|---|---|"
    assert len(lines) == 4


@pytest.mark.parametrize("value, expected", [(0.0004, "< 0.001"), (0.0312, "0.031"), (float("nan"), "n/a")])
def test_format_p(value, expected):
    assert report.format_p(value) == expected


def test_signed_uses_unicode_minus():
    assert report.signed(0.214) == "+0.21"
    assert report.signed(-0.05) == "−0.05"


def test_trend_table_has_row_per_city_and_metric(results):
    table = report.trend_table_markdown(results, ["tmean", "tmax"], 1960)
    assert len(table.splitlines()) == 2 + 4
    assert "Alpha" in table and "Beta" in table
    assert "increasing" in table


def test_period_means_table_contains_all_periods(results):
    table = report.period_means_markdown(results, ["tmean"], [(1961, 1990), (1991, 2020)])
    assert "1961–1990" in table and "1991–2020" in table
    assert len(table.splitlines()) == 2 + 2


def test_comparison_table_reports_difference(results):
    table = report.comparison_markdown(results, ["tmean"], 1960)
    assert "Difference (Alpha minus Beta)" in table
    assert len(table.splitlines()) == 3


def test_sensitivity_table_has_column_per_start_year(results):
    table = report.sensitivity_markdown(results, ["tmean"], [1940, 1960, 1979])
    header = table.splitlines()[0]
    assert "from 1940" in header and "from 1979" in header


def test_normals_table_has_twelve_months_and_total(results):
    table = report.normals_markdown(results)
    assert len(table.splitlines()) == 2 + 12 + 1
    assert "Jan" in table and "Dec" in table


def test_thresholds_table_lists_each_city(results):
    table = report.thresholds_markdown(results)
    assert "Alpha" in table and "25" in table
