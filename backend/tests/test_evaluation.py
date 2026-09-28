import numpy as np
import pytest

from ml_pipeline.src.evaluation import (
    calculate_classification_metrics,
    create_majority_predictions,
)


def test_classification_metrics_are_calculated() -> None:
    true_labels = np.array(
        [0, 0, 1, 1, 2, 2]
    )

    predicted_labels = np.array(
        [0, 1, 1, 1, 2, 0]
    )

    metrics = calculate_classification_metrics(
        true_labels,
        predicted_labels,
    )

    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["balanced_accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1
    assert len(metrics["confusion_matrix"]) == 3


def test_perfect_predictions_score_one() -> None:
    labels = np.array(
        [0, 1, 2, 0, 1, 2]
    )

    metrics = calculate_classification_metrics(
        labels,
        labels,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_majority_predictions_use_most_common_class() -> None:
    training_labels = np.array(
        [0, 1, 1, 1, 2]
    )

    majority_class, predictions = (
        create_majority_predictions(
            training_labels,
            prediction_count=4,
        )
    )

    assert majority_class == 1
    assert predictions.tolist() == [1, 1, 1, 1]


def test_metrics_reject_different_lengths() -> None:
    with pytest.raises(ValueError):
        calculate_classification_metrics(
            np.array([0, 1]),
            np.array([0]),
        )