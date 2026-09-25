from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import analysis, plots
from src.config import City, load_config
from tests.helpers import make_daily


@pytest.fixture(scope="module")
def results():
    config = load_config()
    rng = np.random.default_rng(11)
    output = {}
    for index, name in enumerate(["Alpha", "Beta"]):
        daily = make_daily(1940, 2025)
        seasonal = 8 * np.sin(2 * np.pi * daily["day_of_year"] / 365.25)
        daily["tmean"] = 15 + seasonal + 0.02 * (daily["year"] - 1940) * (index + 1) + rng.normal(0, 2, len(daily))
        daily["tmax"] = daily["tmean"] + 5
        daily["tmin"] = daily["tmean"] - 5
        daily["dtr"] = daily["tmax"] - daily["tmin"]
        daily["precip"] = rng.exponential(1.5, len(daily)) * (rng.random(len(daily)) < 0.4)
        output[name] = analysis.analyze_city(daily, City(name, "Nowhere", 0.0, 0.0, extreme_heat_fixed_c=25.0), config)
    return output, config


@pytest.mark.parametrize("name", list(plots.FIGURES))
def test_every_figure_builds(results, name):
    data, config = results
    fig = plots.FIGURES[name](data, config)
    assert len(fig.axes) >= 2
    assert any("Open-Meteo" in text.get_text() for text in fig.texts)
    plt.close(fig)


def test_clean_stripes_figure_builds(results):
    data, config = results
    fig = plots.plot_stripes_clean(next(iter(data.values())), config)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_save_figure_writes_png_at_configured_resolution(results, tmp_path):
    data, config = results
    config = {**config, "paths": {**config["paths"], "figures": str(tmp_path)}}
    fig = plots.plot_city_comparison(data, config)
    path = plots.save_figure(fig, "smoke", {**config, "paths": {**config["paths"], "figures": str(tmp_path)}})
    assert path.exists()
    assert path.suffix == ".png"


def test_city_colors_are_distinct_and_stable(results):
    data, _ = results
    colors = plots.city_colors(data)
    assert len(set(colors.values())) == len(colors)
    assert colors == plots.city_colors(data)


def test_trend_text_formats_sign_and_significance():
    import pandas as pd

    significant = pd.Series({"slope_per_decade": 0.3123, "p_value": 0.0001})
    weak = pd.Series({"slope_per_decade": -0.05, "p_value": 0.2})
    assert plots.trend_text(significant, "°C") == "+0.31 °C/decade (p < 0.001)"
    assert plots.trend_text(weak, "days", 1) == "−0.1 days/decade (p = 0.200)"
