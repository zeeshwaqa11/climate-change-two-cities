from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator

from src.analysis import CityAnalysis, analyze_all, rolling_mean
from src.config import load_config, resolve_path

INK = "#1F2933"
MUTED = "#616E7C"
GRID = "#E4E7EB"
WARM = "#B2182B"
COOL = "#2166AC"
DAY = "#E69F00"
NIGHT = "#3B3B98"
PERIOD_BASE = "#8C96A3"
PERIOD_RECENT = "#C1272D"
CITY_PALETTE = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DPI = 200

STRIPE_BLUES = ["#08306b", "#08519c", "#2171b5", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef", "#deebf7"]
STRIPE_REDS = ["#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d"]
STRIPES_CMAP = LinearSegmentedColormap.from_list("stripes", STRIPE_BLUES + STRIPE_REDS, N=256)

Results = dict[str, CityAnalysis]


def apply_style() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.dpi": 110,
        },
    )


def city_colors(results: Results) -> dict[str, str]:
    return {name: CITY_PALETTE[index % len(CITY_PALETTE)] for index, name in enumerate(results)}


def credit_text(config: dict[str, Any]) -> str:
    start, end = config["periods"]["baseline"]
    return (
        "Data: Open-Meteo.com (CC BY 4.0), ERA5 reanalysis from the Copernicus Climate Change Service. "
        f"Anomalies relative to {start}–{end}."
    )


def finalize(
    fig: Figure,
    config: dict[str, Any],
    title: str,
    subtitle: str | None = None,
    note: str | None = None,
    rect: tuple[float, float, float, float] | None = None,
    legend: list[Any] | None = None,
    legend_columns: int = 3,
) -> Figure:
    height = fig.get_size_inches()[1]
    top = 1 - (0.95 if subtitle else 0.6) / height
    left, bottom, right, _ = rect or (0.0, 0.04, 1.0, top)
    fig.get_layout_engine().set(rect=(left, bottom, right, rect[3] if rect else top))
    fig.text(0.012, 1 - 0.1 / height, title, ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 1 - 0.47 / height, subtitle, ha="left", va="top", fontsize=10, color=MUTED)
    if legend:
        fig.legend(handles=legend, loc="upper right", ncol=legend_columns, bbox_to_anchor=(0.99, 1 - 0.42 / height), borderaxespad=0)
    fig.text(0.988, 0.012, credit_text(config), ha="right", va="bottom", fontsize=7, color=MUTED)
    if note:
        fig.text(0.012, 0.012, note, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    return fig


def new_figure(width: float, height: float) -> Figure:
    apply_style()
    return plt.figure(figsize=(width, height), layout="constrained")


def trend_row(analysis: CityAnalysis, metric: str, start_year: int) -> pd.Series:
    trends = analysis.trends
    return trends[(trends["metric"] == metric) & (trends["start_year"] == start_year)].iloc[0]


def trend_text(row: pd.Series, unit: str, digits: int = 2) -> str:
    p_value = row["p_value"]
    p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
    return minus(f"{row['slope_per_decade']:+.{digits}f} {unit}/decade ({p_text})")


def minus(text: str) -> str:
    return text.replace("-", "\u2212")


def draw_trend(ax: Any, series: pd.Series, row: pd.Series, start_year: int, color: str = INK) -> None:
    window = series.loc[start_year:].dropna()
    slope, intercept = np.polyfit(window.index.to_numpy(dtype=float), window.to_numpy(), 1)
    x = np.array([window.index.min(), window.index.max()], dtype=float)
    style = "-" if row["p_value"] < 0.05 else (0, (4, 3))
    ax.plot(x, intercept + slope * x, color=color, lw=2.2, ls=style, zorder=4, solid_capstyle="round")


def format_year_axis(ax: Any, series: pd.Series, label: bool = True) -> None:
    ax.set_xlim(series.index.min() - 1, series.index.max() + 1)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    if label:
        ax.set_xlabel("Year")


def draw_bars(ax: Any, series: pd.Series, color: str | list[str], alpha: float = 0.9) -> None:
    ax.bar(series.index, series.to_numpy(), width=0.82, color=color, alpha=alpha, linewidth=0, zorder=2)


def round_up(value: float, step: float = 0.25) -> float:
    return float(np.ceil(value / step) * step)


def signed_colors(values: pd.Series) -> list[str]:
    return [WARM if value >= 0 else COOL for value in values]


SIGNIFICANCE_NOTE = "Trend lines fitted from {start}: solid = statistically significant (p < 0.05), dashed = not significant."


def plot_warming_stripes(results: Results, config: dict[str, Any]) -> Figure:
    count = len(results)
    fig = new_figure(12, 2.3 * count + 1.5)
    axes = fig.subplots(count, 1, squeeze=False)[:, 0]
    for ax, (name, analysis) in zip(axes, results.items()):
        anomaly = analysis.annual["tmean_anomaly"]
        limit = round_up(anomaly.abs().max())
        image = ax.imshow(
            anomaly.to_numpy()[np.newaxis, :],
            aspect="auto",
            cmap=STRIPES_CMAP,
            vmin=-limit,
            vmax=limit,
            extent=(anomaly.index.min() - 0.5, anomaly.index.max() + 0.5, 0, 1),
            interpolation="nearest",
        )
        ax.grid(False)
        ax.set_yticks([])
        for side in ["left", "bottom"]:
            ax.spines[side].set_visible(False)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.tick_params(axis="x", length=3)
        ax.set_title(f"{name}, {analysis.city.country}", fontsize=12)
        ax.set_title(f"{anomaly.index.min()}–{anomaly.index.max()}", loc="right", fontsize=9.5, color=MUTED, fontweight="normal")
        cax = ax.inset_axes((1.012, 0.0, 0.014, 1.0))
        colorbar = fig.colorbar(image, cax=cax, ticks=[-limit, 0, limit])
        colorbar.ax.set_yticklabels([f"−{limit:.2g}", "0", f"+{limit:.2g}"], fontsize=8)
        colorbar.ax.set_title("°C", fontsize=8, color=MUTED, fontweight="normal", loc="left")
        colorbar.outline.set_visible(False)
    return finalize(
        fig,
        config,
        "Warming stripes: annual mean temperature",
        "Each bar is one year, coloured by its difference from the 1961–1990 average. Colour range is set per city.",
        rect=(0.0, 0.04, 0.955, 1 - 0.95 / (2.3 * count + 1.5)),
    )


def plot_stripes_clean(analysis: CityAnalysis, config: dict[str, Any]) -> Figure:
    apply_style()
    fig = plt.figure(figsize=(10, 4.2))
    ax = fig.add_axes((0, 0.09, 1, 0.91))
    anomaly = analysis.annual["tmean_anomaly"]
    limit = round_up(anomaly.abs().max())
    ax.imshow(anomaly.to_numpy()[np.newaxis, :], aspect="auto", cmap=STRIPES_CMAP, vmin=-limit, vmax=limit, interpolation="nearest")
    ax.axis("off")
    fig.text(0.012, 0.03, f"{analysis.city.name}  {anomaly.index.min()}–{anomaly.index.max()}", fontsize=11, fontweight="bold", color=INK, va="center")
    fig.text(0.988, 0.03, "Open-Meteo.com (CC BY 4.0) · ERA5 / Copernicus C3S", fontsize=7, color=MUTED, ha="right", va="center")
    return fig


def plot_anomaly_trends(results: Results, config: dict[str, Any]) -> Figure:
    start = config["periods"]["trend_start"]
    count = len(results)
    fig = new_figure(12, 3.6 * count + 1.2)
    axes = fig.subplots(count, 1, sharex=True, squeeze=False)[:, 0]
    for ax, (name, analysis) in zip(axes, results.items()):
        series = analysis.annual["tmean_anomaly"]
        draw_bars(ax, series, signed_colors(series))
        row = trend_row(analysis, "tmean", start)
        draw_trend(ax, series, row, start)
        ax.axhline(0, color=MUTED, lw=0.9, zorder=3)
        ax.set_ylabel("Anomaly vs 1961–1990 (°C)")
        ax.set_title(f"{name}: annual mean temperature anomaly")
        ax.set_title(f"{start}–{series.index.max()}: {trend_text(row, chr(176) + 'C')}", loc="right", fontsize=9.5, color=INK, fontweight="normal")
        format_year_axis(ax, series, label=ax is axes[-1])
        ax.grid(axis="x", visible=False)
    handles = [
        Patch(color=WARM, label="Warmer than baseline"),
        Patch(color=COOL, label="Cooler than baseline"),
        Line2D([], [], color=INK, lw=2.2, label="Linear trend"),
    ]
    return finalize(
        fig,
        config,
        "How much warmer is each year than the 1961–1990 average?",
        "Annual mean of daily mean temperature, reanalysis grid cell nearest the city centre.",
        SIGNIFICANCE_NOTE.format(start=start),
        legend=handles,
    )


def plot_tmax_vs_tmin(results: Results, config: dict[str, Any]) -> Figure:
    start = config["periods"]["trend_start"]
    count = len(results)
    fig = new_figure(12, 8.6)
    grid = GridSpec(2, count, figure=fig, height_ratios=[1.15, 0.95])
    top_axes = []
    for index, (name, analysis) in enumerate(results.items()):
        ax = fig.add_subplot(grid[0, index])
        top_axes.append(ax)
        for metric, color in [("tmax", DAY), ("tmin", NIGHT)]:
            series = analysis.annual[f"{metric}_anomaly"]
            ax.plot(series.index, series.to_numpy(), color=color, lw=1, alpha=0.45, zorder=2)
            draw_trend(ax, series, trend_row(analysis, metric, start), start, color)
        ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
        ax.set_title(f"{name}: days vs nights")
        ax.set_ylabel("Anomaly vs 1961–1990 (°C)")
        format_year_axis(ax, analysis.annual["tmax"])
    legend_handles = [
        Line2D([], [], color=DAY, lw=2.2, label="Daily maximum (days)"),
        Line2D([], [], color=NIGHT, lw=2.2, label="Daily minimum (nights)"),
    ]
    ax = fig.add_subplot(grid[1, :])
    names = list(results)
    positions = np.arange(count)
    width = 0.32
    for offset, metric, color, label in [(-width / 2 - 0.02, "tmax", DAY, "Daily maximum"), (width / 2 + 0.02, "tmin", NIGHT, "Daily minimum")]:
        rows = [trend_row(results[name], metric, start) for name in names]
        slopes = np.array([row["slope_per_decade"] for row in rows])
        errors = np.array([[row["slope_per_decade"] - row["ci_low_per_decade"] for row in rows], [row["ci_high_per_decade"] - row["slope_per_decade"] for row in rows]])
        ax.bar(positions + offset, slopes, width=width, color=color, label=label, zorder=2)
        ax.errorbar(positions + offset, slopes, yerr=errors, fmt="none", ecolor=INK, elinewidth=1.4, capsize=4, zorder=3)
        for x_position, slope, row in zip(positions + offset, slopes, rows):
            ax.text(x_position, row["ci_high_per_decade"] + 0.015, f"{slope:+.2f}", ha="center", va="bottom", fontsize=9, color=INK, fontweight="bold")
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xticks(positions, names)
    ax.tick_params(axis="x", labelsize=10.5, labelcolor=INK)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Trend (°C per decade)")
    ax.set_title(f"Warming rate {start}–{results[names[0]].annual.index.max()} with 95% confidence intervals")
    ax.margins(y=0.15)
    return finalize(
        fig,
        config,
        "Are nights warming faster than days?",
        "Linear trends in annual mean daily maximum and daily minimum temperature.",
        SIGNIFICANCE_NOTE.format(start=start),
        legend=legend_handles,
        legend_columns=2,
    )


def plot_extreme_heat(results: Results, config: dict[str, Any]) -> Figure:
    start = config["periods"]["trend_start"]
    window = config["thresholds"]["rolling_window_years"]
    percentile = int(config["thresholds"]["extreme_heat_percentile"])
    colors = city_colors(results)
    count = len(results)
    expected = 365.25 * (100 - percentile) / 100
    fig = new_figure(12, 3.5 * count + 1.2)
    axes = fig.subplots(count, 2, sharex=True, squeeze=False)
    for row_axes, (name, analysis) in zip(axes, results.items()):
        thresholds = analysis.thresholds
        specs = [
            ("hot_days_fixed", f"{name}: days above {thresholds['fixed_heat_c']:.0f} °C"),
            ("hot_days_pct", f"{name}: days above {thresholds['percentile_heat_c']:.1f} °C ({percentile}th pct.)"),
        ]
        for ax, (metric, title) in zip(row_axes, specs):
            series = analysis.annual[metric].astype(float)
            draw_bars(ax, series, colors[name], alpha=0.55)
            ax.plot(series.index, rolling_mean(series, window), color=INK, lw=2.2, zorder=4, solid_capstyle="round")
            row = trend_row(analysis, metric, start)
            ax.set_title(title, fontsize=10)
            ax.set_title(trend_text(row, "days", 2 if metric == "hot_days_fixed" else 1), loc="right", fontsize=8.5, color=INK, fontweight="normal")
            ax.set_ylabel("Days per year")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            format_year_axis(ax, series, label=row_axes is axes[-1])
            ax.grid(axis="x", visible=False)
            if metric == "hot_days_pct":
                expected = 365.25 * (100 - percentile) / 100
                ax.axhline(expected, color=MUTED, lw=1.1, ls=(0, (2, 3)), zorder=3)
    return finalize(
        fig,
        config,
        "How many extreme hot days does each year bring?",
        f"Left: fixed threshold. Right: above the {percentile}th percentile of 1961\u20131990 daily maximum.",
        "Panel-title trend is the linear trend since " + str(start) + " (days per decade).",
        legend=[
            Patch(color=colors[next(iter(results))], alpha=0.55, label="Annual count"),
            Line2D([], [], color=INK, lw=2.2, label=f"{window}-year rolling mean"),
            Line2D([], [], color=MUTED, lw=1.1, ls=(0, (2, 3)), label=f"Baseline average ({expected:.0f} days)"),
        ],
        legend_columns=3,
    )


def plot_monthly_heatmaps(results: Results, config: dict[str, Any]) -> Figure:
    count = len(results)
    fig = new_figure(12, 2.9 * count + 1.5)
    axes = fig.subplots(count, 1, sharex=True, squeeze=False)[:, 0]
    matrices = [analysis.monthly_anomalies for analysis in results.values()]
    limit = round_up(float(np.nanpercentile(np.abs(np.concatenate([m.to_numpy().ravel() for m in matrices])), 99.5)), 0.5)
    mesh = None
    for ax, (name, analysis), matrix in zip(axes, results.items(), matrices):
        years = matrix.index.to_numpy()
        mesh = ax.pcolormesh(np.append(years, years[-1] + 1) - 0.5, np.arange(13), matrix.to_numpy().T, cmap=STRIPES_CMAP, vmin=-limit, vmax=limit, shading="flat")
        ax.set_ylim(12, 0)
        ax.set_yticks(np.arange(12) + 0.5, MONTHS, fontsize=8)
        ax.tick_params(axis="y", length=0)
        ax.grid(False)
        ax.set_title(f"{name}: monthly mean temperature anomaly")
        ax.set_ylabel("Month")
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
        for side in ["left", "bottom"]:
            ax.spines[side].set_visible(False)
        if ax is axes[-1]:
            ax.set_xlabel("Year")
    colorbar = fig.colorbar(mesh, ax=list(axes), shrink=0.9, pad=0.015, aspect=30, extend="both")
    colorbar.set_label("Anomaly vs same month in 1961–1990 (°C)")
    colorbar.outline.set_visible(False)
    return finalize(
        fig,
        config,
        "When in the year is the warming happening?",
        "Each cell is one calendar month of one year. Both cities share one colour scale.",
    )


def plot_monthly_normals(results: Results, config: dict[str, Any]) -> Figure:
    base = tuple(config["periods"]["baseline"])
    recent = tuple(config["periods"]["recent_normal"])
    base_label = f"{base[0]}–{base[1]}"
    recent_label = f"{recent[0]}–{recent[1]}"
    count = len(results)
    fig = new_figure(12.5, 8.4)
    axes = fig.subplots(2, count, squeeze=False)
    months = np.arange(1, 13)
    for column, (name, analysis) in enumerate(results.items()):
        normals = analysis.normals
        ax = axes[0, column]
        ax.plot(months, normals["tmean_base"], color=PERIOD_BASE, lw=2.2, marker="o", ms=5, label=base_label, zorder=3)
        ax.plot(months, normals["tmean_recent"], color=PERIOD_RECENT, lw=2.2, marker="o", ms=5, label=recent_label, zorder=4)
        ax.fill_between(months, normals["tmean_base"], normals["tmean_recent"], color=PERIOD_RECENT, alpha=0.12, linewidth=0)
        ax.set_title(f"{name}: mean temperature")
        ax.set_title(f"{normals['tmean_change'].mean():+.2f} °C on average", loc="right", fontsize=9, color=INK, fontweight="normal")
        ax.set_ylabel("Monthly mean temperature (°C)")
        ax.set_xticks(months, MONTHS, fontsize=8.5)
        ax.grid(axis="x", visible=False)
        ax = axes[1, column]
        ax.bar(months - 0.2, normals["precip_base"], width=0.38, color=PERIOD_BASE, label=base_label, zorder=2)
        ax.bar(months + 0.2, normals["precip_recent"], width=0.38, color=PERIOD_RECENT, label=recent_label, zorder=2)
        ax.set_title(f"{name}: rainfall")
        ax.set_title(minus(f"{normals['precip_recent'].sum() - normals['precip_base'].sum():+.0f} mm per year"), loc="right", fontsize=9, color=INK, fontweight="normal")
        ax.set_ylabel("Monthly rainfall (mm)")
        ax.set_xticks(months, MONTHS, fontsize=8.5)
        ax.grid(axis="x", visible=False)
    return finalize(
        fig,
        config,
        "How do the seasons differ between two climate normals?",
        "Monthly averages for two 30-year periods.",
        legend=[Patch(color=PERIOD_BASE, label=base_label), Patch(color=PERIOD_RECENT, label=recent_label)],
        legend_columns=2,
    )


def plot_rainfall_trends(results: Results, config: dict[str, Any]) -> Figure:
    start = config["periods"]["trend_start"]
    colors = city_colors(results)
    count = len(results)
    fig = new_figure(16, 3.4 * count + 1.2)
    axes = fig.subplots(count, 3, sharex=True, squeeze=False)
    specs = [
        ("precip_total", "annual total", "Rainfall (mm per year)", "mm", 0),
        ("rx1day", "wettest day", "Heaviest day (mm)", "mm", 1),
        ("longest_dry_spell", "longest dry spell", "Consecutive dry days", "days", 1),
    ]
    for row_axes, (name, analysis) in zip(axes, results.items()):
        for ax, (metric, label, ylabel, unit, digits) in zip(row_axes, specs):
            series = analysis.annual[metric].astype(float)
            draw_bars(ax, series, colors[name], alpha=0.6)
            row = trend_row(analysis, metric, start)
            draw_trend(ax, series, row, start)
            ax.set_title(f"{name}: {label}", fontsize=10)
            ax.set_title(trend_text(row, unit, digits), loc="right", fontsize=8.5, color=INK, fontweight="normal")
            ax.set_ylabel(ylabel)
            format_year_axis(ax, series, label=row_axes is axes[-1])
            ax.grid(axis="x", visible=False)
    wet = config["thresholds"]["wet_day_mm"]
    return finalize(
        fig,
        config,
        "Is the rain changing?",
        f"Annual totals, the single wettest day, and the longest run of consecutive days with less than {wet:g} mm of rain.",
        SIGNIFICANCE_NOTE.format(start=start),
    )


def plot_city_comparison(results: Results, config: dict[str, Any]) -> Figure:
    window = config["thresholds"]["rolling_window_years"]
    colors = city_colors(results)
    fig = new_figure(12.5, 5.6)
    axes = fig.subplots(1, 2)
    specs = [
        (axes[0], "tmean_anomaly", "Annual mean temperature anomaly", "Anomaly vs 1961–1990 (°C)"),
        (axes[1], "hot_days_pct", "Days above baseline 95th percentile", "Days per year"),
    ]
    for ax, metric, title, ylabel in specs:
        for name, analysis in results.items():
            series = analysis.annual[metric].astype(float)
            ax.plot(series.index, series.to_numpy(), color=colors[name], lw=1, alpha=0.3, zorder=2)
            smoothed = rolling_mean(series, window)
            ax.plot(smoothed.index, smoothed.to_numpy(), color=colors[name], lw=2.8, zorder=4, solid_capstyle="round", label=name)
            last = smoothed.dropna()
            ax.text(last.index[-1] + 0.8, last.iloc[-1], name, color=colors[name], fontsize=10, fontweight="bold", va="center")
        if metric == "tmean_anomaly":
            ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        format_year_axis(ax, next(iter(results.values())).annual["tmean"])
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] + 9)
        ax.set_xticks(range(1940, 2026, 10))
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left")
    return finalize(
        fig,
        config,
        "Two cities, two very different warming stories",
        f"Thick lines: {window}-year rolling mean. Thin lines: individual years.",
    )


FIGURES = {
    "warming_stripes": plot_warming_stripes,
    "anomaly_trend": plot_anomaly_trends,
    "tmax_vs_tmin": plot_tmax_vs_tmin,
    "extreme_heat_days": plot_extreme_heat,
    "monthly_anomaly_heatmap": plot_monthly_heatmaps,
    "monthly_normals": plot_monthly_normals,
    "rainfall_trends": plot_rainfall_trends,
    "city_comparison": plot_city_comparison,
}


def save_figure(fig: Figure, name: str, config: dict[str, Any]) -> Path:
    path = resolve_path(config, "figures") / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def export_figures(results: Results, config: dict[str, Any] | None = None) -> list[Path]:
    config = config or load_config()
    paths = []
    for name, builder in FIGURES.items():
        paths.append(save_figure(builder(results, config), name, config))
    for analysis in results.values():
        paths.append(save_figure(plot_stripes_clean(analysis, config), f"stripes_{analysis.city.slug}", config))
    return paths


def main() -> None:
    config = load_config()
    for path in export_figures(analyze_all(config), config):
        print(f"saved {path.name}")


if __name__ == "__main__":
    main()
