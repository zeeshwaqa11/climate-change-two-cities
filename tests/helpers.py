from __future__ import annotations

import numpy as np
import pandas as pd

from src.clean import add_features


def make_daily(first_year: int, last_year: int, **overrides: float) -> pd.DataFrame:
    dates = pd.date_range(f"{first_year}-01-01", f"{last_year}-12-31", freq="D")
    values = {"tmax": 25.0, "tmin": 15.0, "tmean": 20.0, "precip": 0.0, "wind": 10.0}
    values.update(overrides)
    frame = pd.DataFrame({"date": dates, **{name: np.full(len(dates), value, dtype=float) for name, value in values.items()}})
    return add_features(frame, wet_day_mm=1.0)


def linear_series(first_year: int, last_year: int, slope_per_year: float, intercept: float = 10.0) -> pd.Series:
    years = np.arange(first_year, last_year + 1)
    return pd.Series(intercept + slope_per_year * (years - first_year), index=pd.Index(years, name="year"))
