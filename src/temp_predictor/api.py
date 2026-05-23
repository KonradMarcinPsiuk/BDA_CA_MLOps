"""Flask API serving next-day temperature forecasts for Dublin."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Lock

import pandas as pd
from flask import Flask, jsonify, request

from . import __version__
from .data import LOCATION_NAME, fetch_daily_temperatures
from .predict import predict_next_day

logger = logging.getLogger(__name__)

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
MODEL_FILENAME = "ridge.joblib"


def _model_path() -> Path:
    return MODEL_DIR / MODEL_FILENAME


def create_app() -> Flask:
    app = Flask(__name__)
    cache_lock = Lock()
    history_cache: dict[int, pd.DataFrame] = {}

    def _get_history(days: int) -> pd.DataFrame:
        with cache_lock:
            cached = history_cache.get(days)
        if cached is not None:
            return cached.copy()

        df = fetch_daily_temperatures(days=days)
        with cache_lock:
            history_cache[days] = df
        return df.copy()

    @app.get("/health")
    def health() -> tuple:
        return jsonify({"status": "ok", "version": __version__}), 200

    @app.get("/predict")
    def predict_get() -> tuple:
        try:
            days = int(request.args.get("days", "120"))
        except ValueError:
            return jsonify({"error": "bad_request", "message": "'days' must be int."}), 400

        model_path = _model_path()
        if not model_path.exists():
            return (
                jsonify(
                    {
                        "error": "model_not_found",
                        "message": "No trained model present. Run training first.",
                        "expected_path": str(model_path),
                    }
                ),
                503,
            )

        try:
            history = _get_history(days)
            prediction = predict_next_day(model_path, history)
        except Exception as exc:  # noqa: BLE001 - surface a clean message
            logger.exception("Prediction failed")
            return jsonify({"error": "prediction_failed", "message": str(exc)}), 500

        return (
            jsonify(
                {
                    "location": LOCATION_NAME,
                    "target_date": prediction.target_date.date().isoformat(),
                    "predicted_temperature_c": round(
                        prediction.predicted_temperature_c, 2
                    ),
                    "history_days": int(len(history)),
                }
            ),
            200,
        )

    @app.post("/predict")
    def predict_post() -> tuple:
        payload = request.get_json(silent=True) or {}
        history_payload = payload.get("history")
        if not isinstance(history_payload, list) or not history_payload:
            return (
                jsonify(
                    {
                        "error": "bad_request",
                        "message": (
                            "'history' must be a non-empty list of "
                            "{date, temperature_c} objects."
                        ),
                    }
                ),
                400,
            )

        try:
            history = pd.DataFrame(history_payload)
            history["date"] = pd.to_datetime(history["date"])
            history["temperature_c"] = history["temperature_c"].astype(float)
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": "bad_request", "message": str(exc)}), 400

        model_path = _model_path()
        if not model_path.exists():
            return (
                jsonify(
                    {
                        "error": "model_not_found",
                        "expected_path": str(model_path),
                    }
                ),
                503,
            )

        try:
            prediction = predict_next_day(model_path, history)
        except Exception as exc:  # noqa: BLE001
            logger.exception("POST /predict failed")
            return jsonify({"error": "prediction_failed", "message": str(exc)}), 500

        return (
            jsonify(
                {
                    "location": LOCATION_NAME,
                    "target_date": prediction.target_date.date().isoformat(),
                    "predicted_temperature_c": round(
                        prediction.predicted_temperature_c, 2
                    ),
                    "history_days": int(len(history)),
                }
            ),
            200,
        )

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
