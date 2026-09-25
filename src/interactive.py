from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import to_hex

from src.analysis import CityAnalysis, rolling_mean
from src.plots import COOL, INK, MUTED, STRIPES_CMAP, WARM, round_up, trend_row, trend_text

FONT = {"family": "Inter, Segoe UI, Helvetica, Arial, sans-serif", "color": INK, "size": 13}
GRID = "#E4E7EB"
HEAT_COLOR = "#D55E00"


def stripes_colorscale(steps: int = 17) -> list[list[Any]]:
    return [[index / (steps - 1), to_hex(STRIPES_CMAP(index / (steps - 1)))] for index in range(steps)]


def base_layout(fig: go.Figure, title: str, height: int, ytitle: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 16}},
        font=FONT,
        height=height,
        margin={"l": 60, "r": 20, "t": 70, "b": 50},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text="Year", showgrid=False, dtick=10, linecolor=GRID)
    fig.update_yaxes(title_text=ytitle, gridcolor=GRID, zeroline=False)
    return fig


def stripes_figure(analysis: CityAnalysis) -> go.Figure:
    anomaly = analysis.annual["tmean_anomaly"].dropna()
    limit = round_up(float(anomaly.abs().max()))
    fig = go.Figure(
        go.Heatmap(
            z=[anomaly.to_numpy()],
            x=anomaly.index.to_numpy(),
            y=[""],
            colorscale=stripes_colorscale(),
            zmin=-limit,
            zmax=limit,
            zmid=0,
            xgap=0,
            colorbar={"title": {"text": "°C vs baseline"}, "thickness": 12, "len": 0.9},
            hovertemplate="%{x}: %{z:+.2f} °C<extra></extra>",
        )
    )
    base_layout(fig, f"{analysis.city.name}: warming stripes", 330)
    fig.update_yaxes(visible=False)
    fig.update_xaxes(showgrid=False, dtick=10)
    fig.update_layout(hovermode="closest")
    return fig


def anomaly_figure(analysis: CityAnalysis, start_year: int) -> go.Figure:
    anomaly = analysis.annual["tmean_anomaly"].dropna()
    colors = [WARM if value >= 0 else COOL for value in anomaly]
    fig = go.Figure(
        go.Bar(
            x=anomaly.index.to_numpy(),
            y=anomaly.to_numpy(),
            marker={"color": colors},
            name="Annual anomaly",
            hovertemplate="%{x}: %{y:+.2f} °C<extra></extra>",
        )
    )
    window = anomaly.loc[start_year:]
    slope, intercept = np.polyfit(window.index.to_numpy(dtype=float), window.to_numpy(), 1)
    x = np.array([window.index.min(), window.index.max()])
    row = trend_row(analysis, "tmean", start_year)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=intercept + slope * x,
            mode="lines",
            line={"color": INK, "width": 3, "dash": "solid" if row["p_value"] < 0.05 else "dash"},
            name=f"Trend since {start_year}: {trend_text(row, chr(176) + 'C')}",
            hoverinfo="skip",
        )
    )
    base_layout(fig, f"{analysis.city.name}: annual mean temperature anomaly", 420, "Anomaly vs 1961–1990 (°C)")
    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    return fig


def heat_figure(analysis: CityAnalysis, metric: str, window: int) -> go.Figure:
    series = analysis.annual[metric].astype(float)
    limits = analysis.thresholds
    if metric == "hot_days_pct":
        title = f"Days above {limits['percentile_heat_c']:.1f} °C ({limits['percentile']:.0f}th percentile of 1961–1990 daily maximum)"
    else:
        title = f"Days above {limits['fixed_heat_c']:.0f} °C"
    fig = go.Figure(
        go.Bar(
            x=series.index.to_numpy(),
            y=series.to_numpy(),
            marker={"color": HEAT_COLOR, "opacity": 0.55},
            name="Days per year",
            hovertemplate="%{x}: %{y:.0f} days<extra></extra>",
        )
    )
    smoothed = rolling_mean(series, window)
    fig.add_trace(
        go.Scatter(
            x=smoothed.index.to_numpy(),
            y=smoothed.to_numpy(),
            mode="lines",
            line={"color": INK, "width": 3},
            name=f"{window}-year rolling mean",
            hovertemplate="%{y:.1f} days<extra></extra>",
        )
    )
    if metric == "hot_days_pct":
        expected = 365.25 * (100 - limits["percentile"]) / 100
        fig.add_hline(
            y=expected,
            line_dash="dot",
            line_color=MUTED,
            annotation_text=f"baseline average: {expected:.0f} days",
            annotation_position="top left",
            annotation_font={"size": 11, "color": MUTED},
        )
    base_layout(fig, f"{analysis.city.name}: {title}", 420, "Days per year")
    return fig
