from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from ml_pipeline.src.lstm_model import (
    build_lstm_model,
    calculate_class_weights,
    set_random_seeds,
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

MODEL_PATH = MODEL_DIRECTORY / "lstm_model_v3.keras"
METRICS_PATH = METRICS_DIRECTORY / "lstm_metrics_v3.json"
HISTORY_PATH = METRICS_DIRECTORY / "lstm_training_history_v3.csv"

VALIDATION_CONFUSION_PATH = (
    METRICS_DIRECTORY / "lstm_validation_confusion_matrix_v3.csv"
)

TEST_CONFUSION_PATH = (
    METRICS_DIRECTORY / "lstm_test_confusion_matrix_v3.csv"
)

BATCH_SIZE = 256
MAXIMUM_EPOCHS = 30
RANDOM_SEED = 42


def load_sequence_datasets() -> dict[str, np.ndarray]:
    sequence_files = sorted(
        SEQUENCE_DIRECTORY.glob("*_sequences_v3.npz")
    )

    if not sequence_files:
        raise FileNotFoundError(
            f"No sequence files found in {SEQUENCE_DIRECTORY}."
        )

    combined: dict[str, list[np.ndarray]] = {
        "x_train": [],
        "y_train": [],
        "x_validation": [],
        "y_validation": [],
        "x_test": [],
        "y_test": [],
    }

    print("Loading sequence datasets...")

    for sequence_path in sequence_files:
        with np.load(sequence_path) as data:
            stock_name = sequence_path.stem.replace(
                "_sequences_v3",
                "",
            ).upper()

            x_train = data["x_train"].astype(np.float32)
            y_train = data["y_train"].astype(np.int64)

            x_validation = data["x_validation"].astype(
                np.float32
            )
            y_validation = data["y_validation"].astype(
                np.int64
            )

            x_test = data["x_test"].astype(np.float32)
            y_test = data["y_test"].astype(np.int64)

            combined["x_train"].append(x_train)
            combined["y_train"].append(y_train)
            combined["x_validation"].append(x_validation)
            combined["y_validation"].append(y_validation)
            combined["x_test"].append(x_test)
            combined["y_test"].append(y_test)

            print(
                f"  {stock_name:<12} "
                f"train={len(y_train):,} | "
                f"validation={len(y_validation):,} | "
                f"test={len(y_test):,}"
            )

    return {
        key: np.concatenate(arrays, axis=0)
        for key, arrays in combined.items()
    }


def convert_to_json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): convert_to_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            convert_to_json_compatible(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            convert_to_json_compatible(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


def save_training_history(
    history: tf.keras.callbacks.History,
) -> None:
    history_rows = history.history
    column_names = ["epoch", *history_rows.keys()]
    epoch_count = len(history_rows["loss"])

    with HISTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=column_names,
        )
        writer.writeheader()

        for epoch_index in range(epoch_count):
            row: dict[str, float | int] = {
                "epoch": epoch_index + 1
            }

            for metric_name, metric_values in history_rows.items():
                row[metric_name] = float(
                    metric_values[epoch_index]
                )

            writer.writerow(row)


def save_confusion_matrix(
    matrix: list[list[int]],
    output_path: Path,
) -> None:
    matrix_array = np.asarray(matrix, dtype=np.int64)

    np.savetxt(
        output_path,
        matrix_array,
        delimiter=",",
        fmt="%d",
        header="predicted_SELL,predicted_HOLD,predicted_BUY",
        comments="",
    )


def evaluate_model(
    model: tf.keras.Model,
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    probability_predictions = model.predict(
        x_data,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    class_predictions = np.argmax(
        probability_predictions,
        axis=1,
    )

    metrics = calculate_classification_metrics(
        y_data,
        class_predictions,
    )

    return metrics, class_predictions


def main() -> None:
    set_random_seeds(RANDOM_SEED)

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    METRICS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    datasets = load_sequence_datasets()

    x_train = datasets["x_train"]
    y_train = datasets["y_train"]

    x_validation = datasets["x_validation"]
    y_validation = datasets["y_validation"]

    x_test = datasets["x_test"]
    y_test = datasets["y_test"]

    print("\nCOMBINED DATASET")
    print("=" * 60)
    print(f"Training samples   : {len(y_train):,}")
    print(f"Validation samples : {len(y_validation):,}")
    print(f"Testing samples    : {len(y_test):,}")
    print(f"Input shape        : {x_train.shape[1:]}")

    class_weights = calculate_class_weights(y_train)

    print("\nCLASS WEIGHTS")
    print("=" * 60)

    for class_number, class_name in enumerate(CLASS_NAMES):
        print(
            f"{class_name:<5}: "
            f"{class_weights[class_number]:.4f}"
        )

    model = build_lstm_model(
        input_shape=(
            x_train.shape[1],
            x_train.shape[2],
        ),
        number_of_classes=len(CLASS_NAMES),
    )

    print("\nMODEL ARCHITECTURE")
    print("=" * 60)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=0.0,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=0.00001,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    print("\nTRAINING LSTM MODEL")
    print("=" * 60)

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(
            x_validation,
            y_validation,
        ),
        epochs=MAXIMUM_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    model.save(MODEL_PATH)
    save_training_history(history)

    validation_metrics, _ = evaluate_model(
        model=model,
        x_data=x_validation,
        y_data=y_validation,
    )

    test_metrics, _ = evaluate_model(
        model=model,
        x_data=x_test,
        y_data=y_test,
    )

    save_confusion_matrix(
        validation_metrics["confusion_matrix"],
        VALIDATION_CONFUSION_PATH,
    )

    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        TEST_CONFUSION_PATH,
    )

    best_epoch = int(
        np.argmin(history.history["val_loss"]) + 1
    )

    results = {
    "model": "lstm_v3",
    "target_configuration": {
        "prediction_horizon_candles": 4,
        "prediction_horizon_minutes": 60,
        "classification_threshold": 0.004,
        "classification_threshold_percent": 0.40,
        "same_session_only": True,
        "estimated_round_trip_cost_percent": 0.20,
    },
        "input_shape": list(x_train.shape[1:]),
        "sample_counts": {
            "train": len(y_train),
            "validation": len(y_validation),
            "test": len(y_test),
        },
        "class_weights": class_weights,
        "training": {
            "maximum_epochs": MAXIMUM_EPOCHS,
            "completed_epochs": len(history.history["loss"]),
            "best_epoch": best_epoch,
            "batch_size": BATCH_SIZE,
            "best_validation_loss": float(
                min(history.history["val_loss"])
            ),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            convert_to_json_compatible(results),
            metrics_file,
            indent=4,
        )

    print("\nLSTM RESULTS")
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

    print(f"\nBest epoch   : {best_epoch}")
    print(f"Model saved  : {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print(f"History saved: {HISTORY_PATH}")


if __name__ == "__main__":
    main()