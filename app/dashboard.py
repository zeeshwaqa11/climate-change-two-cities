from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src import interactive, live
from src.analysis import CityAnalysis
from src.config import City, load_config
from src.plots import trend_row

st.set_page_config(page_title="Climate history for any city", layout="wide")

CONFIG = load_config()
TREND_START = CONFIG["periods"]["trend_start"]
WINDOW = CONFIG["thresholds"]["rolling_window_years"]
BASELINE = tuple(CONFIG["periods"]["baseline"])


@st.cache_data(ttl=3600, show_spinner=False)
def cached_search(query: str) -> list[City]:
    return live.search_places(query, CONFIG)


@st.cache_data(show_spinner=False)
def cached_analysis(city: City) -> CityAnalysis:
    return live.analyse_place(city, CONFIG)


def signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def format_p(p_value: float) -> str:
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def sidebar_selection() -> City | None:
    st.sidebar.header("Choose a place")
    query = st.sidebar.text_input("City name", value="Karachi", help="Type any city and press Enter.")
    try:
        matches = cached_search(query)
    except Exception as error:
        st.sidebar.error(f"Geocoding failed: {error}")
        return None
    if not matches:
        st.sidebar.warning("No matching place found.")
        return None
    labels = [live.place_label(match) for match in matches]
    chosen = st.sidebar.selectbox("Matches", range(len(matches)), format_func=lambda index: labels[index])
    threshold = st.sidebar.number_input(
        "Fixed heat threshold (°C)",
        min_value=0.0,
        max_value=60.0,
        value=30.0,
        step=1.0,
        help="Used for the fixed-threshold chart. The percentile chart adapts to each city automatically.",
    )
    st.sidebar.caption(
        "Data: Open-Meteo.com (CC BY 4.0), ERA5 reanalysis from the Copernicus Climate Change Service. "
        "First download of a new city takes about 10 to 30 seconds and is then cached."
    )
    return replace(matches[chosen], extreme_heat_fixed_c=float(threshold))


def headline_metrics(analysis: CityAnalysis) -> None:
    annual = analysis.annual
    row = trend_row(analysis, "tmean", TREND_START)
    recent = annual.loc[2016:2025]
    baseline_days = annual.loc[BASELINE[0] : BASELINE[1], "hot_days_pct"].mean()
    warmest = annual["tmean"].idxmax()
    columns = st.columns(4)
    columns[0].metric(
        f"Warming trend since {TREND_START}",
        f"{signed(row['slope_per_decade'])} °C / decade",
        f"95% CI {signed(row['ci_low_per_decade'])} to {signed(row['ci_high_per_decade'])}, {format_p(row['p_value'])}",
        delta_color="off",
    )
    columns[1].metric(
        "2016–2025 vs 1961–1990",
        f"{signed(recent['tmean'].mean() - annual.loc[BASELINE[0] : BASELINE[1], 'tmean'].mean())} °C",
        "mean temperature",
        delta_color="off",
    )
    columns[2].metric(
        "Extreme hot days per year",
        f"{recent['hot_days_pct'].mean():.0f}",
        f"{baseline_days:.0f} in 1961–1990 (2016–2025 average shown)",
        delta_color="off",
    )
    columns[3].metric("Warmest year", str(warmest), f"{annual.loc[warmest, 'tmean']:.1f} °C mean", delta_color="off")


def interpretation(analysis: CityAnalysis) -> str:
    row = trend_row(analysis, "tmean", TREND_START)
    significance = "statistically significant" if row["p_value"] < 0.05 else "not statistically significant"
    agreement = "agrees" if (row["mk_trend"] != "no trend") == (row["p_value"] < 0.05) else "does not agree"
    return (
        f"Annual mean temperature in {analysis.city.name} is changing by {signed(row['slope_per_decade'])} °C per decade since "
        f"{TREND_START}, which is {significance} ({format_p(row['p_value'])}). The non-parametric Mann-Kendall test "
        f"{agreement} with the regression (Sen's slope {signed(row['sen_slope_per_decade'])} °C per decade). "
        "This is a reanalysis grid-cell average, not a station record, and it describes change rather than its cause."
    )


def render(city: City) -> None:
    with st.spinner(f"Loading 86 years of daily data for {live.place_label(city)}..."):
        try:
            analysis = cached_analysis(city)
        except Exception as error:
            st.error(f"Could not load data for {city.name}: {error}")
            return
    st.title(f"Climate change in {live.place_label(city)}")
    headline_metrics(analysis)
    st.caption(interpretation(analysis))
    stripes, anomaly, heat, data = st.tabs(["Warming stripes", "Annual anomaly", "Extreme heat", "Data"])
    with stripes:
        st.plotly_chart(interactive.stripes_figure(analysis), use_container_width=True)
        st.caption("One bar per year, coloured by its difference from the 1961–1990 average (blue cooler, red warmer).")
    with anomaly:
        st.plotly_chart(interactive.anomaly_figure(analysis, TREND_START), use_container_width=True)
        st.caption("Solid trend line means statistically significant (p < 0.05); dashed means not significant.")
    with heat:
        left, right = st.columns(2)
        left.plotly_chart(interactive.heat_figure(analysis, "hot_days_pct", WINDOW), use_container_width=True)
        right.plotly_chart(interactive.heat_figure(analysis, "hot_days_fixed", WINDOW), use_container_width=True)
        st.caption(
            "Left: days hotter than the city's own 1961–1990 95th percentile of daily maximum, about 18 a year by "
            "definition in the baseline, so cities are comparable. Right: days above the fixed threshold set in the sidebar."
        )
    with data:
        table = analysis.annual.round(2).reset_index()
        st.dataframe(table, use_container_width=True, height=420)
        st.download_button("Download annual table (CSV)", table.to_csv(index=False), file_name=f"{city.slug}_annual.csv", mime="text/csv")
        cell = live.grid_cell(city)
        if cell:
            st.caption(
                f"Reanalysis grid cell used: {cell['grid_latitude']:.2f}, {cell['grid_longitude']:.2f} "
                f"(you asked for {cell['requested_latitude']:.2f}, {cell['requested_longitude']:.2f}); model {cell['model']}."
            )


def main() -> None:
    city = sidebar_selection()
    if city is None:
        st.title("Climate change in any city")
        st.info("Type a city name in the sidebar to begin.")
        return
    render(city)


main()
