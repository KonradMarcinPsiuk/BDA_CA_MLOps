"""Load a trained model and predict the next day's temperature."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .features import latest_feature_row


@dataclass
class Prediction:
    target_date: pd.Timestamp
    predicted_temperature_c: float


def predict_next_day(model_path: Path, history: pd.DataFrame) -> Prediction:
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    columns = bundle["feature_columns"]

    feature_row = latest_feature_row(history)[columns]
    yhat = float(pipeline.predict(feature_row.to_numpy())[0])

    target_date = history.sort_values("date")["date"].iloc[-1] + pd.Timedelta(days=1)
    return Prediction(target_date=target_date, predicted_temperature_c=yhat)
