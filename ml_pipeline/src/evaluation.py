from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


CLASS_NAMES = [
    "SELL",
    "HOLD",
    "BUY",
]


def calculate_classification_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, Any]:
    if len(true_labels) == 0:
        raise ValueError("true_labels cannot be empty.")

    if len(true_labels) != len(predicted_labels):
        raise ValueError(
            "True and predicted labels must have equal length."
        )

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1, 2],
    )

    return {
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def create_majority_predictions(
    training_labels: np.ndarray,
    prediction_count: int,
) -> tuple[int, np.ndarray]:
    if len(training_labels) == 0:
        raise ValueError(
            "training_labels cannot be empty."
        )

    if prediction_count < 0:
        raise ValueError(
            "prediction_count cannot be negative."
        )

    majority_class = int(
        np.bincount(
            training_labels,
            minlength=3,
        ).argmax()
    )

    predictions = np.full(
        shape=prediction_count,
        fill_value=majority_class,
        dtype=np.int64,
    )

    return majority_class, predictions