"""Data acquisition for daily temperatures in Dublin.

Uses the Open-Meteo forecast API's ``past_days`` window (no API key required).
The archive endpoint (``archive-api.open-meteo.com``) is occasionally
unreachable; the forecast endpoint covers the last ~90 days, which is enough
for our lag/rolling features.
Docs: https://open-meteo.com/en/docs
"""

from __future__ import annotations

import socket
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import urllib3.util.connection as urllib3_cn

# Containers (Docker bridge networking, Docker Desktop's Linux VM, and
# GitHub-hosted runners) get AAAA records back from DNS but typically have
# no IPv6 route, so requests fail with ENETUNREACH. Force IPv4 resolution.
urllib3_cn.allowed_gai_family = lambda: socket.AF_INET

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_PAST_DAYS = 92  # upper bound the forecast API accepts

DUBLIN_LATITUDE = 53.3498
DUBLIN_LONGITUDE = -6.2603
DUBLIN_TIMEZONE = "Europe/Dublin"
LOCATION_NAME = "Dublin"


def fetch_daily_temperatures(days: int = 90, end: date | None = None) -> pd.DataFrame:
    """Fetch the last ``days`` daily mean temperatures for Dublin.

    Uses the forecast API's ``past_days`` window (≤ 92 days). The ``end``
    argument is accepted for backwards compatibility but the forecast API
    always anchors at "now"; we drop any rows past ``end`` if it's provided.
    """
    if days < 14:
        raise ValueError("Need at least 14 days of history for lag features.")
    if days > MAX_PAST_DAYS:
        raise ValueError(
            f"Forecast endpoint supports at most {MAX_PAST_DAYS} past days "
            f"(requested {days})."
        )

    params = {
        "latitude": DUBLIN_LATITUDE,
        "longitude": DUBLIN_LONGITUDE,
        "daily": "temperature_2m_mean",
        "timezone": DUBLIN_TIMEZONE,
        "past_days": days,
        "forecast_days": 1,
    }

    response = requests.get(FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily", {})
    times = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    if not times or not temps:
        raise RuntimeError(f"No data returned for {LOCATION_NAME}: {payload}")

    df = pd.DataFrame({"date": pd.to_datetime(times), "temperature_c": temps})
    df = df.dropna().reset_index(drop=True)

    # Keep only rows on/before `end` (the forecast row carries a NaN we already
    # dropped above; this guard is for callers that pass a historical anchor).
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)].reset_index(drop=True)

    return df


def save_dataset(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
