from __future__ import annotations

import pandas as pd
import pytest
import requests

from src import fetch
from src.config import City

CONFIG = {
    "data": {
        "start_date": "1940-01-01",
        "model": "era5",
        "variables": ["temperature_2m_max", "precipitation_sum"],
        "archive_url": "https://example.test/archive",
    },
    "fetch": {"retries": 3, "backoff_seconds": 0, "timeout_seconds": 5},
    "paths": {"raw": "data/raw"},
}

CITY = City("Testville", "Nowhere", 10.5, 20.25, timezone="Europe/London")


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = "fake"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


def test_build_params_pins_model_and_end_date():
    params = fetch.build_params(CITY, CONFIG, 2025)
    assert params["models"] == "era5"
    assert params["start_date"] == "1940-01-01"
    assert params["end_date"] == "2025-12-31"
    assert params["daily"] == "temperature_2m_max,precipitation_sum"
    assert params["latitude"] == 10.5
    assert params["timezone"] == "Europe/London"


def test_request_retries_after_rate_limit(monkeypatch):
    responses = iter([FakeResponse(429), FakeResponse(503), FakeResponse(200, {"ok": True})])
    calls = []

    def fake_get(url, params, timeout):
        calls.append(url)
        return next(responses)

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)
    assert fetch.request_with_retries("https://example.test", {}, CONFIG) == {"ok": True}
    assert len(calls) == 3


def test_request_gives_up_after_configured_retries(monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda url, params, timeout: FakeResponse(429))
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="3 attempts"):
        fetch.request_with_retries("https://example.test", {}, CONFIG)


def test_request_does_not_retry_client_errors(monkeypatch):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(1)
        return FakeResponse(400)

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError):
        fetch.request_with_retries("https://example.test", {}, CONFIG)
    assert len(calls) == 3


def test_payload_to_frame_parses_dates():
    payload = {"daily": {"time": ["2000-01-01", "2000-01-02"], "temperature_2m_max": [1.0, 2.0]}}
    frame = fetch.payload_to_frame(payload)
    assert list(frame.columns) == ["date", "temperature_2m_max"]
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_fetch_city_skips_existing_file_and_force_redownloads(tmp_path, monkeypatch):
    config = {**CONFIG, "paths": {"raw": str(tmp_path)}, "data": {**CONFIG["data"], "end_year": 2001}}
    payload = {
        "latitude": 10.5,
        "longitude": 20.25,
        "elevation": 5.0,
        "timezone": "Europe/London",
        "daily_units": {},
        "daily": {"time": ["2000-01-01"], "temperature_2m_max": [1.0], "precipitation_sum": [0.0]},
    }
    downloads = []

    def fake_request(url, params, config):
        downloads.append(params)
        return payload

    monkeypatch.setattr(fetch, "request_with_retries", fake_request)
    path = fetch.fetch_city(CITY, config)
    assert path.exists()
    assert (tmp_path / "testville_meta.json").exists()
    fetch.fetch_city(CITY, config)
    assert len(downloads) == 1
    fetch.fetch_city(CITY, config, force=True)
    assert len(downloads) == 2
