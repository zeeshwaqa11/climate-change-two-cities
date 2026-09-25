from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import analysis
from tests.helpers import linear_series, make_daily


def test_linear_trend_recovers_known_slope():
    series = linear_series(1960, 2020, slope_per_year=0.03)
    result = analysis.linear_trend(series)
    assert result["slope_per_decade"] == pytest.approx(0.3)
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["p_value"] < 1e-10
    assert result["n_years"] == 61


def test_linear_trend_respects_start_year():
    values = np.concatenate([np.full(20, 5.0), np.arange(41) * 0.05 + 5.0])
    series = pd.Series(values, index=np.arange(1940, 2001))
    assert analysis.linear_trend(series, start_year=1960)["slope_per_decade"] == pytest.approx(0.5)
    assert analysis.linear_trend(series)["slope_per_decade"] != pytest.approx(0.5)


def test_linear_trend_flat_series_has_zero_slope_and_high_p():
    rng = np.random.default_rng(0)
    series = pd.Series(10 + rng.normal(0, 1, 60), index=np.arange(1960, 2020))
    result = analysis.linear_trend(series)
    assert abs(result["slope_per_decade"]) < 0.5
    assert result["p_value"] > 0.05


def test_linear_trend_confidence_interval_contains_slope():
    rng = np.random.default_rng(1)
    years = np.arange(1960, 2020)
    series = pd.Series(0.02 * (years - 1960) + rng.normal(0, 0.3, len(years)), index=years)
    result = analysis.linear_trend(series)
    assert result["ci_low_per_decade"] < result["slope_per_decade"] < result["ci_high_per_decade"]


def test_mann_kendall_detects_increase_and_sen_slope():
    series = linear_series(1960, 2020, slope_per_year=0.03)
    result = analysis.mann_kendall_trend(series)
    assert result["mk_trend"] == "increasing"
    assert result["mk_p_value"] < 0.001
    assert result["sen_slope_per_decade"] == pytest.approx(0.3)


def test_mann_kendall_detects_decrease():
    series = linear_series(1960, 2020, slope_per_year=-0.02)
    result = analysis.mann_kendall_trend(series)
    assert result["mk_trend"] == "decreasing"
    assert result["sen_slope_per_decade"] == pytest.approx(-0.2)


def test_mann_kendall_reports_no_trend_for_alternating_series():
    series = pd.Series(np.tile([1.0, 2.0], 30), index=np.arange(1960, 2020))
    result = analysis.mann_kendall_trend(series)
    assert result["mk_trend"] == "no trend"
    assert result["mk_p_value"] > 0.05


def test_mann_kendall_ignores_outlier_that_skews_regression():
    values = np.arange(60) * 0.02
    values[-1] = 50.0
    series = pd.Series(values, index=np.arange(1960, 2020))
    assert analysis.mann_kendall_trend(series)["sen_slope_per_decade"] == pytest.approx(0.2, abs=0.02)
    assert analysis.linear_trend(series)["slope_per_decade"] > 1.0


def test_trend_summary_merges_both_methods():
    summary = analysis.trend_summary(linear_series(1960, 2020, 0.03))
    assert {"slope_per_decade", "p_value", "mk_trend", "sen_slope_per_decade"} <= summary.keys()


def test_slope_difference_of_identical_trends_is_zero():
    trend = {"slope_per_decade": 0.3, "stderr_per_decade": 0.05}
    result = analysis.slope_difference(trend, dict(trend))
    assert result["difference_per_decade"] == 0
    assert result["z_score"] == 0
    assert result["p_value"] == pytest.approx(1.0)


def test_slope_difference_detects_large_gap():
    result = analysis.slope_difference(
        {"slope_per_decade": 0.5, "stderr_per_decade": 0.02},
        {"slope_per_decade": 0.1, "stderr_per_decade": 0.02},
    )
    assert result["difference_per_decade"] == pytest.approx(0.4)
    assert result["p_value"] < 0.001


def test_anomalies_are_relative_to_baseline_mean():
    series = linear_series(1950, 2000, slope_per_year=0.1, intercept=10.0)
    baseline = (1961, 1990)
    anomaly = analysis.anomalies(series, baseline)
    assert anomaly.loc[1961:1990].mean() == pytest.approx(0.0)
    assert anomaly.loc[2000] - anomaly.loc[1990] == pytest.approx(1.0)


def test_baseline_mean_uses_only_baseline_years():
    series = pd.Series([1.0, 2.0, 3.0, 100.0], index=[1960, 1961, 1962, 1963])
    assert analysis.baseline_mean(series, (1961, 1962)) == pytest.approx(2.5)


def test_rolling_mean_of_constant_is_constant():
    series = pd.Series(4.0, index=np.arange(1960, 1990))
    assert analysis.rolling_mean(series, 10).dropna().eq(4.0).all()


def test_rolling_mean_smooths_a_ramp():
    series = pd.Series(np.arange(30, dtype=float), index=np.arange(1960, 1990))
    smoothed = analysis.rolling_mean(series, 5)
    assert smoothed.loc[1975] == pytest.approx(15.0)


def test_annual_mean_matches_known_value():
    daily = make_daily(2000, 2001, tmean=20.0)
    daily.loc[daily["year"] == 2001, "tmean"] = 22.0
    result = analysis.annual_mean(daily, "tmean")
    assert result.loc[2000] == pytest.approx(20.0)
    assert result.loc[2001] == pytest.approx(22.0)


def test_annual_mean_drops_years_with_too_little_data():
    daily = make_daily(2000, 2001)
    year_2001 = daily.index[daily["year"] == 2001]
    daily.loc[year_2001[:60], "tmean"] = np.nan
    result = analysis.annual_mean(daily, "tmean")
    assert not np.isnan(result.loc[2000])
    assert np.isnan(result.loc[2001])


def test_annual_sum_and_max():
    daily = make_daily(2000, 2000, precip=0.0)
    daily.loc[[10, 20, 30], "precip"] = [5.0, 12.5, 2.5]
    assert analysis.annual_sum(daily, "precip").loc[2000] == pytest.approx(20.0)
    assert analysis.annual_max(daily, "precip").loc[2000] == pytest.approx(12.5)


def test_count_days_strictly_above_threshold():
    daily = make_daily(2000, 2001, tmax=30.0)
    daily.loc[[0, 1, 2], "tmax"] = [40.0, 40.1, 45.0]
    daily.loc[daily.index[daily["year"] == 2001][0], "tmax"] = 41.0
    counts = analysis.count_days(daily, "tmax", 40.0)
    assert counts.loc[2000] == 2
    assert counts.loc[2001] == 1


def test_count_days_inclusive_counts_threshold_value():
    daily = make_daily(2000, 2000, precip=0.0)
    daily.loc[[0, 1, 2], "precip"] = [20.0, 19.9, 35.0]
    assert analysis.count_days(daily, "precip", 20.0, inclusive=True).loc[2000] == 2
    assert analysis.count_days(daily, "precip", 20.0, inclusive=False).loc[2000] == 1


def test_count_days_reports_zero_for_years_without_exceedances():
    daily = make_daily(2000, 2002, tmax=25.0)
    counts = analysis.count_days(daily, "tmax", 40.0)
    assert counts.to_dict() == {2000: 0, 2001: 0, 2002: 0}


def test_count_days_ignores_missing_values():
    daily = make_daily(2000, 2000, tmax=25.0)
    daily.loc[0:9, "tmax"] = np.nan
    daily.loc[10, "tmax"] = 41.0
    assert analysis.count_days(daily, "tmax", 40.0).loc[2000] == 1


def test_baseline_percentile_matches_numpy():
    daily = make_daily(1961, 1990)
    rng = np.random.default_rng(3)
    daily["tmax"] = rng.normal(25, 5, len(daily))
    expected = np.percentile(daily["tmax"], 95)
    assert analysis.baseline_percentile(daily, "tmax", 95, (1961, 1990)) == pytest.approx(expected)


def test_baseline_percentile_ignores_years_outside_baseline():
    daily = make_daily(1950, 2000, tmax=20.0)
    daily.loc[daily["year"] > 1990, "tmax"] = 60.0
    assert analysis.baseline_percentile(daily, "tmax", 95, (1961, 1990)) == pytest.approx(20.0)


def test_percentile_method_gives_about_five_percent_of_baseline_days():
    daily = make_daily(1961, 1990)
    rng = np.random.default_rng(4)
    daily["tmax"] = rng.normal(35, 4, len(daily))
    threshold = analysis.baseline_percentile(daily, "tmax", 95, (1961, 1990))
    counts = analysis.count_days(daily, "tmax", threshold)
    assert counts.mean() == pytest.approx(365.25 * 0.05, rel=0.05)


def test_percentile_method_is_comparable_across_climates():
    hot = make_daily(1961, 1990)
    cool = make_daily(1961, 1990)
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 3, len(hot))
    hot["tmax"] = 38.0 + noise
    cool["tmax"] = 14.0 + noise
    hot_days = analysis.count_days(hot, "tmax", analysis.baseline_percentile(hot, "tmax", 95, (1961, 1990)))
    cool_days = analysis.count_days(cool, "tmax", analysis.baseline_percentile(cool, "tmax", 95, (1961, 1990)))
    assert hot_days.equals(cool_days)


@pytest.mark.parametrize(
    "flags, expected",
    [
        ([0, 1, 1, 0, 1, 1, 1, 0], 3),
        ([1, 1, 1, 1], 4),
        ([0, 0, 0], 0),
        ([], 0),
        ([1, 0, 1, 0, 1], 1),
        ([1, 1, 0, 0, 0, 1, 1], 2),
    ],
)
def test_longest_run(flags, expected):
    assert analysis.longest_run(np.array(flags)) == expected


def test_longest_dry_spell_finds_correct_run():
    daily = make_daily(2000, 2000, precip=5.0)
    daily.loc[10:19, "precip"] = 0.0
    daily.loc[100:132, "precip"] = 0.0
    daily.loc[200:204, "precip"] = 0.0
    assert analysis.longest_dry_spell(daily, 1.0).loc[2000] == 33


def test_dry_spell_threshold_is_wet_day_cutoff():
    daily = make_daily(2000, 2000, precip=5.0)
    daily.loc[0:9, "precip"] = 0.9
    daily.loc[20:24, "precip"] = 1.0
    assert analysis.longest_dry_spell(daily, 1.0).loc[2000] == 10


def test_dry_spell_does_not_cross_year_boundaries():
    daily = make_daily(2000, 2001, precip=5.0)
    december = daily.index[(daily["year"] == 2000) & (daily["month"] == 12)][-5:]
    january = daily.index[(daily["year"] == 2001) & (daily["month"] == 1)][:5]
    daily.loc[december, "precip"] = 0.0
    daily.loc[january, "precip"] = 0.0
    spells = analysis.longest_dry_spell(daily, 1.0)
    assert spells.loc[2000] == 5
    assert spells.loc[2001] == 5


def test_dry_spell_of_a_completely_dry_year():
    daily = make_daily(2001, 2001, precip=0.0)
    assert analysis.longest_dry_spell(daily, 1.0).loc[2001] == 365


def test_dry_spell_is_broken_by_missing_data():
    daily = make_daily(2000, 2000, precip=5.0)
    daily.loc[0:9, "precip"] = 0.0
    daily.loc[5, "precip"] = np.nan
    assert analysis.longest_dry_spell(daily, 1.0).loc[2000] == 5


def test_monthly_climatology_temperature_and_rainfall():
    daily = make_daily(1961, 1962, tmax=25.0, tmin=15.0, tmean=20.0, precip=0.0)
    january = daily["month"] == 1
    daily.loc[january, "tmean"] = 5.0
    daily.loc[january & (daily["date"].dt.day == 1), "precip"] = 30.0
    climatology = analysis.monthly_climatology(daily, (1961, 1962))
    assert climatology.loc[1, "tmean"] == pytest.approx(5.0)
    assert climatology.loc[7, "tmean"] == pytest.approx(20.0)
    assert climatology.loc[1, "precip"] == pytest.approx(30.0)
    assert climatology.loc[7, "precip"] == pytest.approx(0.0)


def test_monthly_rainfall_is_average_of_yearly_monthly_totals():
    daily = make_daily(1961, 1962, precip=0.0)
    daily.loc[(daily["year"] == 1961) & (daily["month"] == 3) & (daily["date"].dt.day == 1), "precip"] = 10.0
    daily.loc[(daily["year"] == 1962) & (daily["month"] == 3) & (daily["date"].dt.day == 1), "precip"] = 30.0
    assert analysis.monthly_climatology(daily, (1961, 1962)).loc[3, "precip"] == pytest.approx(20.0)


def test_normals_comparison_reports_change():
    base = make_daily(1961, 1990, tmean=10.0, tmax=15.0, tmin=5.0, precip=1.0)
    recent = make_daily(1991, 2020, tmean=11.5, tmax=16.0, tmin=7.0, precip=2.0)
    table = analysis.normals_comparison(pd.concat([base, recent], ignore_index=True), (1961, 1990), (1991, 2020))
    assert table["tmean_change"].eq(1.5).all()
    assert table["tmax_change"].eq(1.0).all()
    assert table["tmin_change"].eq(2.0).all()
    assert (table["precip_change"] > 0).all()
    assert list(table.index) == list(range(1, 13))


def test_monthly_anomaly_matrix_is_zero_on_baseline_average():
    daily = make_daily(1961, 2000, tmean=10.0)
    daily.loc[daily["year"] >= 1991, "tmean"] = 12.0
    matrix = analysis.monthly_anomaly_matrix(daily, "tmean", (1961, 1990))
    assert matrix.shape == (40, 12)
    assert matrix.loc[1961:1990].abs().to_numpy().max() == pytest.approx(0.0)
    assert matrix.loc[1995].eq(2.0).all()


def test_build_annual_table_columns_and_counts():
    config = {
        "periods": {"baseline": [1961, 1990]},
        "thresholds": {
            "extreme_heat_fixed_c": 40.0,
            "extreme_heat_percentile": 95,
            "wet_day_mm": 1.0,
            "heavy_rain_mm": 20.0,
        },
    }
    from src.config import City

    daily = make_daily(1961, 1995, tmax=25.0, tmin=15.0, tmean=20.0, precip=0.0)
    daily.loc[daily["year"] == 1995, "tmax"] = 35.0
    daily.loc[[0, 1, 2], "precip"] = [25.0, 3.0, 0.5]
    city = City("Testville", "Nowhere", 0.0, 0.0, extreme_heat_fixed_c=30.0)
    annual, thresholds = analysis.build_annual_table(daily, city, config)
    assert thresholds["fixed_heat_c"] == 30.0
    assert thresholds["percentile_heat_c"] == pytest.approx(25.0)
    assert annual.loc[1995, "hot_days_fixed"] == 365
    assert annual.loc[1994, "hot_days_fixed"] == 0
    assert annual.loc[1995, "hot_days_pct"] == 365
    assert annual.loc[1961, "heavy_rain_days"] == 1
    assert annual.loc[1961, "wet_days"] == 2
    assert annual.loc[1961, "rx1day"] == pytest.approx(25.0)
    assert annual.loc[1961, "precip_total"] == pytest.approx(28.5)
    assert annual.loc[1995, "tmax_anomaly"] == pytest.approx(10.0)


def test_trend_table_has_row_per_metric_and_period():
    config = {
        "periods": {"baseline": [1961, 1990]},
        "thresholds": {
            "extreme_heat_fixed_c": 40.0,
            "extreme_heat_percentile": 95,
            "wet_day_mm": 1.0,
            "heavy_rain_mm": 20.0,
        },
    }
    from src.config import City

    rng = np.random.default_rng(6)
    daily = make_daily(1950, 2000)
    daily["tmax"] = 25 + rng.normal(0, 3, len(daily)) + 0.02 * (daily["year"] - 1950)
    daily["precip"] = rng.exponential(1.0, len(daily))
    annual, _ = analysis.build_annual_table(daily, City("T", "X", 0.0, 0.0), config)
    table = analysis.trend_table(annual, "T", [1960, 1950])
    assert len(table) == 2 * len(analysis.TREND_METRICS)
    tmax_row = table[(table["metric"] == "tmax") & (table["start_year"] == 1950)].iloc[0]
    assert tmax_row["slope_per_decade"] == pytest.approx(0.2, abs=0.05)
    assert tmax_row["mk_trend"] == "increasing"
