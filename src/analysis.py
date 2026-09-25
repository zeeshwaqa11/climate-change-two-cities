from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pymannkendall as mk
from scipy import stats

from src.clean import load_daily
from src.config import City, get_cities, load_config, resolve_path

MIN_COVERAGE = 0.95

TREND_METRICS: dict[str, tuple[str, str]] = {
    "tmean": ("Annual mean temperature", "°C"),
    "tmax": ("Annual mean of daily maximum", "°C"),
    "tmin": ("Annual mean of daily minimum", "°C"),
    "dtr": ("Diurnal range (Tmax minus Tmin)", "°C"),
    "hot_days_fixed": ("Days above fixed heat threshold", "days"),
    "hot_days_pct": ("Days above baseline 95th percentile", "days"),
    "precip_total": ("Annual rainfall", "mm"),
    "rx1day": ("Heaviest rain day of the year", "mm"),
    "heavy_rain_days": ("Days with heavy rain", "days"),
    "longest_dry_spell": ("Longest dry spell", "days"),
}


def annual_mean(daily: pd.DataFrame, column: str, min_coverage: float = MIN_COVERAGE) -> pd.Series:
    grouped = daily.groupby("year")[column]
    coverage = grouped.count() / grouped.size()
    return grouped.mean().where(coverage >= min_coverage).rename(column)


def annual_sum(daily: pd.DataFrame, column: str, min_coverage: float = MIN_COVERAGE) -> pd.Series:
    grouped = daily.groupby("year")[column]
    coverage = grouped.count() / grouped.size()
    return grouped.sum(min_count=1).where(coverage >= min_coverage).rename(column)


def annual_max(daily: pd.DataFrame, column: str, min_coverage: float = MIN_COVERAGE) -> pd.Series:
    grouped = daily.groupby("year")[column]
    coverage = grouped.count() / grouped.size()
    return grouped.max().where(coverage >= min_coverage).rename(column)


def baseline_mean(series: pd.Series, baseline: tuple[int, int]) -> float:
    return float(series.loc[baseline[0] : baseline[1]].mean())


def anomalies(series: pd.Series, baseline: tuple[int, int]) -> pd.Series:
    return series - baseline_mean(series, baseline)


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=window // 2).mean()


def lag1_autocorrelation(values: np.ndarray) -> float:
    if len(values) < 3:
        return float("nan")
    return float(np.corrcoef(values[1:], values[:-1])[0, 1])


def linear_trend(series: pd.Series, start_year: int | None = None, end_year: int | None = None) -> dict[str, float]:
    window = series.loc[start_year:end_year].dropna()
    years = window.index.to_numpy(dtype=float)
    result = stats.linregress(years, window.to_numpy())
    margin = stats.t.ppf(0.975, len(window) - 2) * result.stderr
    residuals = window.to_numpy() - (result.intercept + result.slope * years)
    return {
        "n_years": int(len(window)),
        "slope_per_decade": float(result.slope * 10),
        "ci_low_per_decade": float((result.slope - margin) * 10),
        "ci_high_per_decade": float((result.slope + margin) * 10),
        "stderr_per_decade": float(result.stderr * 10),
        "r_squared": float(result.rvalue**2),
        "p_value": float(result.pvalue),
        "residual_lag1_autocorr": lag1_autocorrelation(residuals),
    }


def mann_kendall_trend(series: pd.Series, start_year: int | None = None, end_year: int | None = None) -> dict[str, float | str]:
    values = series.loc[start_year:end_year].dropna().to_numpy()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        standard = mk.original_test(values)
        adjusted = mk.hamed_rao_modification_test(values)
    return {
        "mk_trend": str(standard.trend),
        "mk_tau": float(standard.Tau),
        "mk_p_value": float(standard.p),
        "mk_p_value_autocorr_adjusted": float(adjusted.p),
        "sen_slope_per_decade": float(standard.slope * 10),
    }


def trend_summary(series: pd.Series, start_year: int | None = None, end_year: int | None = None) -> dict[str, float | str]:
    return {**linear_trend(series, start_year, end_year), **mann_kendall_trend(series, start_year, end_year)}


def slope_difference(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    difference = first["slope_per_decade"] - second["slope_per_decade"]
    stderr = float(np.hypot(first["stderr_per_decade"], second["stderr_per_decade"]))
    z_score = difference / stderr
    return {
        "difference_per_decade": float(difference),
        "stderr_per_decade": stderr,
        "z_score": float(z_score),
        "p_value": float(2 * stats.norm.sf(abs(z_score))),
    }


def count_days(daily: pd.DataFrame, column: str, threshold: float, inclusive: bool = False) -> pd.Series:
    exceeds = daily[column] >= threshold if inclusive else daily[column] > threshold
    return exceeds.groupby(daily["year"]).sum().astype(int)


def baseline_percentile(daily: pd.DataFrame, column: str, percentile: float, baseline: tuple[int, int]) -> float:
    in_baseline = daily["year"].between(baseline[0], baseline[1])
    return float(daily.loc[in_baseline, column].quantile(percentile / 100))


def longest_run(flags: np.ndarray) -> int:
    padded = np.concatenate(([0], np.asarray(flags, dtype=int), [0]))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return int((ends - starts).max()) if len(starts) else 0


def longest_dry_spell(daily: pd.DataFrame, wet_day_mm: float) -> pd.Series:
    is_dry = (daily["precip"] < wet_day_mm).to_numpy()
    return pd.Series(
        {year: longest_run(is_dry[positions]) for year, positions in daily.groupby("year").indices.items()},
        name="longest_dry_spell",
    ).astype(int)


def monthly_climatology(daily: pd.DataFrame, period: tuple[int, int]) -> pd.DataFrame:
    subset = daily[daily["year"].between(period[0], period[1])]
    temperatures = subset.groupby("month")[["tmax", "tmin", "tmean"]].mean()
    monthly_totals = subset.groupby(["year", "month"])["precip"].sum(min_count=1)
    rainfall = monthly_totals.groupby("month").mean().rename("precip")
    return temperatures.join(rainfall)


def normals_comparison(daily: pd.DataFrame, baseline: tuple[int, int], recent: tuple[int, int]) -> pd.DataFrame:
    first = monthly_climatology(daily, baseline).add_suffix("_base")
    second = monthly_climatology(daily, recent).add_suffix("_recent")
    table = first.join(second)
    for column in ["tmax", "tmin", "tmean", "precip"]:
        table[f"{column}_change"] = table[f"{column}_recent"] - table[f"{column}_base"]
    return table


def monthly_anomaly_matrix(daily: pd.DataFrame, column: str, baseline: tuple[int, int]) -> pd.DataFrame:
    monthly = daily.groupby(["year", "month"])[column].mean().unstack("month")
    climatology = monthly.loc[baseline[0] : baseline[1]].mean()
    return monthly - climatology


@dataclass
class CityAnalysis:
    city: City
    annual: pd.DataFrame
    monthly_anomalies: pd.DataFrame
    normals: pd.DataFrame
    thresholds: dict[str, float]
    trends: pd.DataFrame


def build_annual_table(daily: pd.DataFrame, city: City, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    baseline = tuple(config["periods"]["baseline"])
    limits = config["thresholds"]
    fixed = city.heat_threshold(config)
    percentile_value = baseline_percentile(daily, "tmax", limits["extreme_heat_percentile"], baseline)
    annual = pd.DataFrame(
        {
            "tmean": annual_mean(daily, "tmean"),
            "tmax": annual_mean(daily, "tmax"),
            "tmin": annual_mean(daily, "tmin"),
            "dtr": annual_mean(daily, "dtr"),
            "hot_days_fixed": count_days(daily, "tmax", fixed),
            "hot_days_pct": count_days(daily, "tmax", percentile_value),
            "precip_total": annual_sum(daily, "precip"),
            "rx1day": annual_max(daily, "precip"),
            "heavy_rain_days": count_days(daily, "precip", limits["heavy_rain_mm"], inclusive=True),
            "wet_days": count_days(daily, "precip", limits["wet_day_mm"], inclusive=True),
            "longest_dry_spell": longest_dry_spell(daily, limits["wet_day_mm"]),
        }
    )
    for column in ["tmean", "tmax", "tmin"]:
        annual[f"{column}_anomaly"] = anomalies(annual[column], baseline)
    annual.index.name = "year"
    thresholds = {
        "fixed_heat_c": fixed,
        "percentile_heat_c": percentile_value,
        "percentile": float(limits["extreme_heat_percentile"]),
        "wet_day_mm": float(limits["wet_day_mm"]),
        "heavy_rain_mm": float(limits["heavy_rain_mm"]),
    }
    return annual, thresholds


def trend_table(annual: pd.DataFrame, city_name: str, start_years: list[int]) -> pd.DataFrame:
    rows = []
    end_year = int(annual.index.max())
    for start_year in start_years:
        for metric, (label, unit) in TREND_METRICS.items():
            summary = trend_summary(annual[metric], start_year, end_year)
            rows.append(
                {"city": city_name, "metric": metric, "label": label, "unit": unit, "start_year": start_year, "end_year": end_year, **summary}
            )
    return pd.DataFrame(rows)


def analyze_city(daily: pd.DataFrame, city: City, config: dict[str, Any]) -> CityAnalysis:
    periods = config["periods"]
    baseline = tuple(periods["baseline"])
    recent = tuple(periods["recent_normal"])
    annual, thresholds = build_annual_table(daily, city, config)
    start_years = list(dict.fromkeys([periods["trend_start"], *periods["sensitivity_starts"]]))
    trends = trend_table(annual, city.name, start_years)
    return CityAnalysis(
        city=city,
        annual=annual,
        monthly_anomalies=monthly_anomaly_matrix(daily, "tmean", baseline),
        normals=normals_comparison(daily, baseline, recent),
        thresholds=thresholds,
        trends=trends,
    )


def analyze_all(config: dict[str, Any] | None = None) -> dict[str, CityAnalysis]:
    config = config or load_config()
    return {city.name: analyze_city(load_daily(city, config), city, config) for city in get_cities(config)}


def compare_cities(results: dict[str, CityAnalysis], metric: str, start_year: int) -> dict[str, float]:
    first, second = [
        trend_summary(analysis.annual[metric], start_year, int(analysis.annual.index.max())) for analysis in results.values()
    ]
    return slope_difference(first, second)


def export_all(results: dict[str, CityAnalysis], config: dict[str, Any] | None = None) -> None:
    config = config or load_config()
    processed = resolve_path(config, "processed")
    for name, analysis in results.items():
        slug = analysis.city.slug
        analysis.annual.round(3).to_csv(processed / f"{slug}_annual.csv")
        analysis.normals.round(3).to_csv(processed / f"{slug}_normals.csv")
        analysis.monthly_anomalies.round(3).to_csv(processed / f"{slug}_monthly_anomalies.csv")
    all_trends = pd.concat([analysis.trends for analysis in results.values()], ignore_index=True)
    all_trends.to_csv(processed / "trend_summary.csv", index=False, float_format="%.5g")
    thresholds = pd.DataFrame({name: analysis.thresholds for name, analysis in results.items()}).T
    thresholds.to_csv(processed / "thresholds.csv", float_format="%.3f")


def main() -> None:
    config = load_config()
    results = analyze_all(config)
    export_all(results, config)
    print(f"Exported analysis tables for {', '.join(results)} to {resolve_path(config, 'processed')}")


if __name__ == "__main__":
    main()
