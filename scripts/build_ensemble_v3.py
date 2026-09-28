from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import tensorflow as tf

from ml_pipeline.src.ensemble import (
    combine_probabilities,
    search_three_model_weights,
)
from ml_pipeline.src.evaluation import (
    CLASS_NAMES,
    calculate_classification_metrics,
)


SEQUENCE_DIRECTORY = Path(
    "ml_pipeline/data/processed/sequences_v3"
)

MODEL_DIRECTORY = Path(
    "ml_pipeline/artifacts/models/v3"
)

METRICS_DIRECTORY = Path(
    "ml_pipeline/reports/metrics/v3"
)

LOGISTIC_MODEL_PATH = (
    MODEL_DIRECTORY / "baseline_logistic_regression_v3.joblib"
)
CONV1D_MODEL_PATH = MODEL_DIRECTORY / "conv1d_model_v3.keras"
LSTM_MODEL_PATH = MODEL_DIRECTORY / "lstm_model_v3.keras"

CONFIG_PATH = MODEL_DIRECTORY / "ensemble_config_v3.json"
METRICS_PATH = METRICS_DIRECTORY / "ensemble_metrics_v3.json"
CONFUSION_PATH = (
    METRICS_DIRECTORY / "ensemble_test_confusion_matrix_v3.csv"
)

BATCH_SIZE = 256


def load_datasets() -> dict[str, np.ndarray]:
    files = sorted(
        SEQUENCE_DIRECTORY.glob("*_sequences_v3.npz")
    )

    if not files:
        raise FileNotFoundError("No sequence datasets were found.")

    datasets: dict[str, list[np.ndarray]] = {
        "x_validation": [],
        "y_validation": [],
        "x_test": [],
        "y_test": [],
    }

    for path in files:
        with np.load(path) as data:
            datasets["x_validation"].append(
                data["x_validation"].astype(np.float32)
            )
            datasets["y_validation"].append(
                data["y_validation"].astype(np.int64)
            )
            datasets["x_test"].append(
                data["x_test"].astype(np.float32)
            )
            datasets["y_test"].append(
                data["y_test"].astype(np.int64)
            )

    return {
        name: np.concatenate(values, axis=0)
        for name, values in datasets.items()
    }


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def main() -> None:
    required_files = [
        LOGISTIC_MODEL_PATH,
        CONV1D_MODEL_PATH,
        LSTM_MODEL_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required model does not exist: {path}"
            )

    METRICS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Loading validation and test datasets...")
    data = load_datasets()

    x_validation = data["x_validation"]
    y_validation = data["y_validation"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    print("Loading trained models...")

    logistic_model = joblib.load(LOGISTIC_MODEL_PATH)

    conv1d_model = tf.keras.models.load_model(
        CONV1D_MODEL_PATH,
        compile=False,
    )

    lstm_model = tf.keras.models.load_model(
        LSTM_MODEL_PATH,
        compile=False,
    )

    print("Generating validation probabilities...")

    logistic_validation = logistic_model.predict_proba(
        x_validation[:, -1, :]
    )

    conv1d_validation = conv1d_model.predict(
        x_validation,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    lstm_validation = lstm_model.predict(
        x_validation,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    print("Searching ensemble weights using validation data...")

    search_result = search_three_model_weights(
        logistic_probabilities=logistic_validation,
        conv1d_probabilities=conv1d_validation,
        lstm_probabilities=lstm_validation,
        y_true=y_validation,
        step=0.05,
    )

    weights = search_result.weights

    print("\nOPTIMAL VALIDATION WEIGHTS")
    print("=" * 60)
    print(f"Logistic Regression: {weights[0]:.2f}")
    print(f"Conv1D             : {weights[1]:.2f}")
    print(f"LSTM               : {weights[2]:.2f}")
    print(f"Validation macro F1: {search_result.macro_f1:.4f}")

    validation_probabilities = combine_probabilities(
        [
            logistic_validation,
            conv1d_validation,
            lstm_validation,
        ],
        weights,
    )

    validation_predictions = np.argmax(
        validation_probabilities,
        axis=1,
    )

    validation_metrics = calculate_classification_metrics(
        y_validation,
        validation_predictions,
    )

    print("\nGenerating test probabilities...")

    logistic_test = logistic_model.predict_proba(
        x_test[:, -1, :]
    )

    conv1d_test = conv1d_model.predict(
        x_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    lstm_test = lstm_model.predict(
        x_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    test_probabilities = combine_probabilities(
        [
            logistic_test,
            conv1d_test,
            lstm_test,
        ],
        weights,
    )

    test_predictions = np.argmax(
        test_probabilities,
        axis=1,
    )

    test_metrics = calculate_classification_metrics(
        y_test,
        test_predictions,
    )

    configuration = {
    "ensemble_version": "v3_cost_aware",
    "target_configuration": {
        "prediction_horizon_candles": 4,
        "prediction_horizon_minutes": 60,
        "classification_threshold": 0.004,
        "classification_threshold_percent": 0.40,
        "same_session_only": True,
        "estimated_round_trip_cost_percent": 0.20,
    },
    "model_order": [
            "logistic_regression_v3",
            "conv1d_v3",
            "lstm_v3",
        ],
        "weights": {
            "logistic_regression_v3": weights[0],
            "conv1d_v3": weights[1],
            "lstm_v3": weights[2],
        },
        "class_mapping": {
            index: class_name
            for index, class_name in enumerate(CLASS_NAMES)
        },
        "optimization_metric": "validation_macro_f1",
        "weight_search_step": 0.05,
    }

    results = {
        **configuration,
        "sample_counts": {
            "validation": len(y_validation),
            "test": len(y_test),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            json_compatible(configuration),
            file,
            indent=4,
        )

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            json_compatible(results),
            file,
            indent=4,
        )

    np.savetxt(
        CONFUSION_PATH,
        np.asarray(
            test_metrics["confusion_matrix"],
            dtype=np.int64,
        ),
        delimiter=",",
        fmt="%d",
        header="predicted_SELL,predicted_HOLD,predicted_BUY",
        comments="",
    )

    print("\nENSEMBLE RESULTS")
    print("=" * 60)

    print(
        "Validation: "
        f"accuracy={validation_metrics['accuracy']:.4f}, "
        f"balanced_accuracy="
        f"{validation_metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={validation_metrics['macro_f1']:.4f}"
    )

    print(
        "Test      : "
        f"accuracy={test_metrics['accuracy']:.4f}, "
        f"balanced_accuracy="
        f"{test_metrics['balanced_accuracy']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )

    print(f"\nConfiguration saved: {CONFIG_PATH}")
    print(f"Metrics saved      : {METRICS_PATH}")


if __name__ == "__main__":
    main()