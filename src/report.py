from __future__ import annotations

import json
from typing import Any

import pandas as pd

from src.analysis import TREND_METRICS, CityAnalysis, compare_cities, trend_summary
from src.clean import load_daily
from src.config import get_cities, resolve_path

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEGREE = "°C"

Results = dict[str, CityAnalysis]


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def format_p(p_value: float) -> str:
    if pd.isna(p_value):
        return "n/a"
    return "< 0.001" if p_value < 0.001 else f"{p_value:.3f}"


def signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def digits_for(unit: str) -> int:
    return 2 if unit == DEGREE else 1


def trend_rows(results: Results, metric: str, start_year: int) -> list[dict[str, Any]]:
    rows = []
    for name, analysis in results.items():
        annual = analysis.annual
        summary = trend_summary(annual[metric], start_year, int(annual.index.max()))
        rows.append({"city": name, **summary})
    return rows


def trend_table_markdown(results: Results, metrics: list[str], start_year: int) -> str:
    header = ["City", "Metric", "Trend per decade", "95% CI", "p (regression)", "Mann-Kendall", "Sen's slope per decade", "p (autocorr.-adjusted)"]
    rows = []
    for metric in metrics:
        label, unit = TREND_METRICS[metric]
        digits = digits_for(unit)
        for item in trend_rows(results, metric, start_year):
            rows.append(
                [
                    item["city"],
                    label,
                    f"{signed(item['slope_per_decade'], digits)} {unit}",
                    f"[{signed(item['ci_low_per_decade'], digits)}, {signed(item['ci_high_per_decade'], digits)}]",
                    format_p(item["p_value"]),
                    item["mk_trend"],
                    f"{signed(item['sen_slope_per_decade'], digits)} {unit}",
                    format_p(item["mk_p_value_autocorr_adjusted"]),
                ]
            )
    return markdown_table(header, rows)


def period_means_markdown(results: Results, metrics: list[str], periods: list[tuple[int, int]]) -> str:
    header = ["City", "Metric"] + [f"{start}–{end}" for start, end in periods]
    rows = []
    for metric in metrics:
        label, unit = TREND_METRICS[metric]
        for name, analysis in results.items():
            values = [f"{analysis.annual.loc[start:end, metric].mean():.{digits_for(unit) if unit == DEGREE else 1}f}" for start, end in periods]
            rows.append([name, f"{label} ({unit}{'/year' if unit != DEGREE else ''})"] + values)
    return markdown_table(header, rows)


def comparison_markdown(results: Results, metrics: list[str], start_year: int) -> str:
    names = list(results)
    header = ["Metric", f"{names[0]} trend", f"{names[1]} trend", f"Difference ({names[0]} minus {names[1]})", "p (difference)"]
    rows = []
    for metric in metrics:
        label, unit = TREND_METRICS[metric]
        digits = digits_for(unit)
        first, second = [trend_summary(results[name].annual[metric], start_year, int(results[name].annual.index.max())) for name in names]
        difference = compare_cities(results, metric, start_year)
        rows.append(
            [
                f"{label} ({unit}/decade)",
                signed(first["slope_per_decade"], digits),
                signed(second["slope_per_decade"], digits),
                signed(difference["difference_per_decade"], digits),
                format_p(difference["p_value"]),
            ]
        )
    return markdown_table(header, rows)


def sensitivity_markdown(results: Results, metrics: list[str], start_years: list[int]) -> str:
    header = ["City", "Metric"] + [f"from {year}" for year in start_years]
    rows = []
    for metric in metrics:
        label, unit = TREND_METRICS[metric]
        digits = digits_for(unit)
        for name, analysis in results.items():
            cells = []
            for year in start_years:
                summary = trend_summary(analysis.annual[metric], year, int(analysis.annual.index.max()))
                cells.append(f"{signed(summary['slope_per_decade'], digits)} (p {format_p(summary['p_value']).replace('< ', '<')})")
            rows.append([name, f"{label} ({unit}/decade)"] + cells)
    return markdown_table(header, rows)


def normals_markdown(results: Results) -> str:
    names = list(results)
    header = ["Month"]
    for name in names:
        header += [f"{name} temp. change ({DEGREE})", f"{name} rain change (mm)"]
    rows = []
    for month in range(1, 13):
        row = [MONTH_NAMES[month - 1]]
        for name in names:
            normals = results[name].normals
            row += [signed(normals.loc[month, "tmean_change"]), signed(normals.loc[month, "precip_change"], 1)]
        rows.append(row)
    totals = ["**Year**"]
    for name in names:
        normals = results[name].normals
        totals += [f"**{signed(normals['tmean_change'].mean())}** (mean)", f"**{signed(normals['precip_change'].sum(), 1)}** (total)"]
    rows.append(totals)
    return markdown_table(header, rows)


def thresholds_markdown(results: Results) -> str:
    header = ["City", "Fixed threshold", "Baseline 95th percentile of daily max", "Wet day", "Heavy-rain day"]
    rows = []
    for name, analysis in results.items():
        limits = analysis.thresholds
        rows.append(
            [
                name,
                f"{limits['fixed_heat_c']:.0f} {DEGREE}",
                f"{limits['percentile_heat_c']:.1f} {DEGREE}",
                f"≥ {limits['wet_day_mm']:g} mm",
                f"≥ {limits['heavy_rain_mm']:g} mm",
            ]
        )
    return markdown_table(header, rows)


def grid_cell_markdown(config: dict[str, Any]) -> str:
    raw = resolve_path(config, "raw")
    header = ["City", "Requested (lat, lon)", "Reanalysis grid cell (lat, lon)", "Cell elevation", "Model"]
    rows = []
    for city in get_cities(config):
        meta = json.loads((raw / f"{city.slug}_meta.json").read_text(encoding="utf-8"))
        rows.append(
            [
                city.name,
                f"{meta['requested_latitude']:.3f}, {meta['requested_longitude']:.3f}",
                f"{meta['grid_latitude']:.3f}, {meta['grid_longitude']:.3f}",
                f"{meta['elevation_m']:.0f} m",
                meta.get("model", "n/a"),
            ]
        )
    return markdown_table(header, rows)


def data_quality_markdown(config: dict[str, Any]) -> str:
    header = ["City", "Daily records", "First day", "Last day", "Complete years", "Missing daily values"]
    rows = []
    for city in get_cities(config):
        daily = load_daily(city, config)
        missing = int(daily[["tmax", "tmin", "tmean", "precip", "wind"]].isna().sum().sum())
        rows.append(
            [
                city.name,
                f"{len(daily):,}",
                daily["date"].min().date().isoformat(),
                daily["date"].max().date().isoformat(),
                str(daily["year"].nunique()),
                f"{missing} of {len(daily) * 5:,}",
            ]
        )
    return markdown_table(header, rows)
