from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.temp_predictor.features import (
    LAGS,
    ROLLING_WINDOW,
    build_features,
    feature_columns,
    latest_feature_row,
)


def _synthetic_history(n: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    temps = 10 + 5 * np.sin(np.arange(n) / 10) + np.random.default_rng(0).normal(0, 0.5, n)
    return pd.DataFrame({"date": dates, "temperature_c": temps})


def test_build_features_produces_expected_columns() -> None:
    df = _synthetic_history()
    feats = build_features(df)
    for col in feature_columns():
        assert col in feats.columns
    # No NaNs should remain after dropna.
    assert not feats[feature_columns()].isna().any().any()


def test_latest_feature_row_returns_single_row() -> None:
    df = _synthetic_history()
    row = latest_feature_row(df)
    assert len(row) == 1
    assert list(row.columns) == feature_columns()


def test_latest_feature_row_requires_enough_history() -> None:
    short = _synthetic_history(n=max(LAGS) + ROLLING_WINDOW - 1)
    with pytest.raises(ValueError):
        latest_feature_row(short)
