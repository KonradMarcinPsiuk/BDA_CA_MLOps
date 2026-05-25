from __future__ import annotations


import numpy as np
import pandas as pd


LAGS: tuple[int, ...] = (1, 2, 3, 7)
ROLLING_WINDOW = 7

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling-mean features.

    Expects a DataFrame with columns ``date`` and ``temperature_c`` sorted
    chronologically. Returns a copy with NaNs from lagging dropped.
    """
    if not {"date", "temperature_c"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'date' and 'temperature_c'.")

    out = df.sort_values("date").reset_index(drop=True).copy()
    for lag in LAGS:
        out[f"lag_{lag}"] = out["temperature_c"].shift(lag)
    out[f"roll_mean_{ROLLING_WINDOW}"] = (
        out["temperature_c"].shift(1).rolling(ROLLING_WINDOW).mean()
    )
    out["day_of_year"] = out["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
    return out.dropna().reset_index(drop=True)


def feature_columns() -> list[str]:
    cols = [f"lag_{lag}" for lag in LAGS]
    cols.append(f"roll_mean_{ROLLING_WINDOW}")
    cols.extend(["doy_sin", "doy_cos"])
    return cols


def latest_feature_row(df: pd.DataFrame) -> pd.DataFrame:
    """Build a single-row feature frame to predict the day after the last entry."""
    if len(df) < max(LAGS) + ROLLING_WINDOW:
        raise ValueError("Not enough history to build features for prediction.")

    history = df.sort_values("date").reset_index(drop=True)
    next_date = history["date"].iloc[-1] + pd.Timedelta(days=1)
    doy = next_date.dayofyear

    row = {f"lag_{lag}": history["temperature_c"].iloc[-lag] for lag in LAGS}
    row[f"roll_mean_{ROLLING_WINDOW}"] = (
        history["temperature_c"].iloc[-ROLLING_WINDOW:].mean()
    )
    row["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    row["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return pd.DataFrame([row], columns=feature_columns())
