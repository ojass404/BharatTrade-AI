from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


@dataclass(frozen=True)
class EnsembleSearchResult:
    weights: tuple[float, float, float]
    macro_f1: float
    accuracy: float


def combine_probabilities(
    probability_matrices: list[np.ndarray],
    weights: tuple[float, ...],
) -> np.ndarray:
    if not probability_matrices:
        raise ValueError("At least one probability matrix is required.")

    if len(probability_matrices) != len(weights):
        raise ValueError(
            "The number of probability matrices and weights must match."
        )

    reference_shape = probability_matrices[0].shape

    if len(reference_shape) != 2:
        raise ValueError("Probability matrices must be two-dimensional.")

    for matrix in probability_matrices:
        if matrix.shape != reference_shape:
            raise ValueError(
                "All probability matrices must have the same shape."
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                "Probability matrices cannot contain non-finite values."
            )

    weight_array = np.asarray(weights, dtype=np.float64)

    if np.any(weight_array < 0):
        raise ValueError("Ensemble weights cannot be negative.")

    if not np.isclose(weight_array.sum(), 1.0):
        raise ValueError("Ensemble weights must sum to 1.")

    combined = np.zeros(reference_shape, dtype=np.float64)

    for matrix, weight in zip(probability_matrices, weight_array):
        combined += matrix * weight

    row_sums = combined.sum(axis=1, keepdims=True)

    if np.any(row_sums <= 0):
        raise ValueError("Combined probabilities contain an invalid row.")

    return combined / row_sums


def search_three_model_weights(
    logistic_probabilities: np.ndarray,
    conv1d_probabilities: np.ndarray,
    lstm_probabilities: np.ndarray,
    y_true: np.ndarray,
    step: float = 0.05,
) -> EnsembleSearchResult:
    if step <= 0 or step > 1:
        raise ValueError("step must be greater than 0 and at most 1.")

    number_of_steps = round(1 / step)

    if not np.isclose(number_of_steps * step, 1.0):
        raise ValueError("step must divide 1.0 exactly.")

    best_result: EnsembleSearchResult | None = None

    matrices = [
        logistic_probabilities,
        conv1d_probabilities,
        lstm_probabilities,
    ]

    for logistic_index in range(number_of_steps + 1):
        for conv1d_index in range(
            number_of_steps - logistic_index + 1
        ):
            lstm_index = (
                number_of_steps
                - logistic_index
                - conv1d_index
            )

            weights = (
                logistic_index / number_of_steps,
                conv1d_index / number_of_steps,
                lstm_index / number_of_steps,
            )

            probabilities = combine_probabilities(
                matrices,
                weights,
            )

            predictions = np.argmax(probabilities, axis=1)

            macro_f1 = float(
                f1_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )

            accuracy = float(
                accuracy_score(y_true, predictions)
            )

            candidate = EnsembleSearchResult(
                weights=weights,
                macro_f1=macro_f1,
                accuracy=accuracy,
            )

            if best_result is None:
                best_result = candidate
                continue

            if candidate.macro_f1 > best_result.macro_f1:
                best_result = candidate
            elif (
                np.isclose(
                    candidate.macro_f1,
                    best_result.macro_f1,
                )
                and candidate.accuracy > best_result.accuracy
            ):
                best_result = candidate

    if best_result is None:
        raise RuntimeError("No ensemble weight combination was evaluated.")

    return best_result