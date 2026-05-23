"""Data acquisition for daily temperatures in Dublin.

Uses the Open-Meteo archive API (no API key required).
Docs: https://open-meteo.com/en/docs/historical-weather-api
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DUBLIN_LATITUDE = 53.3498
DUBLIN_LONGITUDE = -6.2603
DUBLIN_TIMEZONE = "Europe/Dublin"
LOCATION_NAME = "Dublin"


def fetch_daily_temperatures(days: int = 120, end: date | None = None) -> pd.DataFrame:
    """Fetch the last ``days`` daily mean temperatures for Dublin.

    The archive API typically lags real-time by a few days, so we shift the
    requested window back by a small buffer to avoid empty responses.
    """
    if days < 14:
        raise ValueError("Need at least 14 days of history for lag features.")

    end_date = end or date.today() - timedelta(days=5)
    start_date = end_date - timedelta(days=days - 1)

    params = {
        "latitude": DUBLIN_LATITUDE,
        "longitude": DUBLIN_LONGITUDE,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_mean",
        "timezone": DUBLIN_TIMEZONE,
    }

    response = requests.get(ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily", {})
    times = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    if not times or not temps:
        raise RuntimeError(f"No data returned for {LOCATION_NAME}: {payload}")

    df = pd.DataFrame({"date": pd.to_datetime(times), "temperature_c": temps})
    df = df.dropna().reset_index(drop=True)
    return df


def save_dataset(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
