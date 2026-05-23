"""CLI entry point: fetch Dublin temperatures, train a model, predict next day."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.temp_predictor.data import (
    LOCATION_NAME,
    fetch_daily_temperatures,
    save_dataset,
)
from src.temp_predictor.predict import predict_next_day
from src.temp_predictor.train import train_model

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_FILENAME = "ridge.joblib"
DATASET_FILENAME = "dublin_temperatures.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily Dublin temperature forecaster.")
    parser.add_argument(
        "--days",
        type=int,
        default=120,
        help="Number of historical days to fetch (default: 120).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip retraining if a model already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Fetching {args.days} days of temperatures for {LOCATION_NAME}...")
    df = fetch_daily_temperatures(days=args.days)
    dataset_path = DATA_DIR / DATASET_FILENAME
    save_dataset(df, dataset_path)
    print(f"  saved {len(df)} rows to {dataset_path}")

    model_path = MODEL_DIR / MODEL_FILENAME
    if args.skip_train and model_path.exists():
        print(f"Skipping training, using existing model at {model_path}.")
    else:
        print("Training model...")
        result = train_model(df, model_path)
        print(
            f"  trained on {result.n_train} rows, evaluated on {result.n_test}: "
            f"MAE={result.mae:.2f} C, R2={result.r2:.3f}"
        )
        print(f"  model saved to {result.model_path}")

    print("Predicting next day's temperature...")
    prediction = predict_next_day(model_path, df)
    print(
        f"  {prediction.target_date.date()} forecast for {LOCATION_NAME}: "
        f"{prediction.predicted_temperature_c:.2f} C"
    )


if __name__ == "__main__":
    main()
