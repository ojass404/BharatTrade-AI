import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml_pipeline.src.evaluation import (
    CLASS_NAMES,
    calculate_classification_metrics,
    create_majority_predictions,
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


def load_combined_data() -> dict[str, np.ndarray]:
    sequence_files = sorted(
        SEQUENCE_DIRECTORY.glob("*_sequences_v3.npz")
    )

    if not sequence_files:
        raise SystemExit(
            "No sequence files found. "
            "Run scripts.prepare_ml_sequences_v3 first."
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

    for file_path in sequence_files:
        symbol = file_path.stem.removesuffix(
            "_sequences"
        ).upper()

        with np.load(
            file_path,
            allow_pickle=False,
        ) as dataset:
            # Logistic Regression uses only the latest
            # feature row from each 60-candle sequence.
            combined["x_train"].append(
                dataset["x_train"][:, -1, :]
            )

            combined["y_train"].append(
                dataset["y_train"]
            )

            combined["x_validation"].append(
                dataset["x_validation"][:, -1, :]
            )

            combined["y_validation"].append(
                dataset["y_validation"]
            )

            combined["x_test"].append(
                dataset["x_test"][:, -1, :]
            )

            combined["y_test"].append(
                dataset["y_test"]
            )

            print(
                f"  {symbol:<12} "
                f"train={len(dataset['y_train']):>6,} | "
                f"validation="
                f"{len(dataset['y_validation']):>5,} | "
                f"test={len(dataset['y_test']):>5,}"
            )

    return {
        key: np.concatenate(parts, axis=0)
        for key, parts in combined.items()
    }


def label_distribution(
    labels: np.ndarray,
) -> dict[str, int]:
    counts = np.bincount(
        labels,
        minlength=3,
    )

    return {
        CLASS_NAMES[index]: int(counts[index])
        for index in range(3)
    }


def save_confusion_matrix(
    matrix: list[list[int]],
    output_path: Path,
) -> None:
    frame = pd.DataFrame(
        matrix,
        index=[
            f"actual_{name.lower()}"
            for name in CLASS_NAMES
        ],
        columns=[
            f"predicted_{name.lower()}"
            for name in CLASS_NAMES
        ],
    )

    frame.to_csv(output_path)


def main() -> None:
    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_combined_data()

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_validation = data["x_validation"]
    y_validation = data["y_validation"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    print("\nCOMBINED DATASET")
    print("=" * 60)
    print(f"Training samples   : {len(y_train):,}")
    print(f"Validation samples : {len(y_validation):,}")
    print(f"Testing samples    : {len(y_test):,}")
    print(
        f"Training distribution: "
        f"{label_distribution(y_train)}"
    )

    majority_class, majority_validation_predictions = (
        create_majority_predictions(
            training_labels=y_train,
            prediction_count=len(y_validation),
        )
    )

    _, majority_test_predictions = (
        create_majority_predictions(
            training_labels=y_train,
            prediction_count=len(y_test),
        )
    )

    majority_validation_metrics = (
        calculate_classification_metrics(
            y_validation,
            majority_validation_predictions,
        )
    )

    majority_test_metrics = (
        calculate_classification_metrics(
            y_test,
            majority_test_predictions,
        )
    )

    print("\nTraining Logistic Regression baseline...")

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )

    model.fit(
        x_train,
        y_train,
    )

    validation_predictions = model.predict(
        x_validation
    )

    test_predictions = model.predict(
        x_test
    )

    validation_metrics = calculate_classification_metrics(
        y_validation,
        validation_predictions,
    )

    test_metrics = calculate_classification_metrics(
        y_test,
        test_predictions,
    )

    metrics: dict[str, Any] = {
        "class_mapping": {
            "0": "SELL",
            "1": "HOLD",
            "2": "BUY",
        },
        "majority_class": CLASS_NAMES[majority_class],
        "sample_counts": {
            "train": len(y_train),
            "validation": len(y_validation),
            "test": len(y_test),
        },
        "class_distributions": {
            "train": label_distribution(y_train),
            "validation": label_distribution(y_validation),
            "test": label_distribution(y_test),
        },
        "majority_baseline": {
            "validation": majority_validation_metrics,
            "test": majority_test_metrics,
        },
        "logistic_regression": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
    }

    model_path = (
        MODEL_DIRECTORY
        / "baseline_logistic_regression_v3.joblib"
    )

    metrics_path = (
        METRICS_DIRECTORY
        / "baseline_metrics_v3.json"
    )

    validation_matrix_path = (
        METRICS_DIRECTORY
        / "baseline_validation_confusion_matrix_v3.csv"
    )

    test_matrix_path = (
        METRICS_DIRECTORY
        / "baseline_test_confusion_matrix_v3.csv"
    )

    joblib.dump(
        model,
        model_path,
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_confusion_matrix(
        validation_metrics["confusion_matrix"],
        validation_matrix_path,
    )

    save_confusion_matrix(
        test_metrics["confusion_matrix"],
        test_matrix_path,
    )

    print("\nBASELINE RESULTS")
    print("=" * 60)

    print(
        "Majority validation:"
        f" accuracy="
        f"{majority_validation_metrics['accuracy']:.4f},"
        f" macro_f1="
        f"{majority_validation_metrics['macro_f1']:.4f}"
    )

    print(
        "Logistic validation:"
        f" accuracy={validation_metrics['accuracy']:.4f},"
        f" balanced_accuracy="
        f"{validation_metrics['balanced_accuracy']:.4f},"
        f" macro_f1={validation_metrics['macro_f1']:.4f}"
    )

    print(
        "Logistic test:"
        f" accuracy={test_metrics['accuracy']:.4f},"
        f" balanced_accuracy="
        f"{test_metrics['balanced_accuracy']:.4f},"
        f" macro_f1={test_metrics['macro_f1']:.4f}"
    )

    print(f"\nModel saved  : {model_path}")
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()