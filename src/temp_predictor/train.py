"""Train and persist the temperature regression model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_features, feature_columns

TARGET_COLUMN = "temperature_c"


@dataclass
class TrainResult:
    model_path: Path
    mae: float
    r2: float
    n_train: int
    n_test: int


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )


def train_model(
    df: pd.DataFrame,
    model_path: Path,
    test_fraction: float = 0.2,
) -> TrainResult:
    """Fit a Ridge regression on lag features and persist it to ``model_path``."""
    features = build_features(df)
    if len(features) < 20:
        raise ValueError("Need at least 20 rows after feature engineering.")

    columns = feature_columns()
    x = features[columns].to_numpy()
    y = features[TARGET_COLUMN].to_numpy()

    # Time-aware split: hold out the most recent slice for evaluation.
    split_idx = max(int(len(features) * (1 - test_fraction)), 1)
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    pipeline = _build_pipeline()
    pipeline.fit(x_train, y_train)

    if len(y_test) > 0:
        preds = pipeline.predict(x_test)
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds)) if len(y_test) > 1 else float("nan")
    else:
        mae = float("nan")
        r2 = float("nan")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": columns}, model_path)

    return TrainResult(
        model_path=model_path,
        mae=mae,
        r2=r2,
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
    )
