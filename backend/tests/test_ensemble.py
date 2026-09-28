import numpy as np
import pytest

from ml_pipeline.src.ensemble import (
    combine_probabilities,
    search_three_model_weights,
)


def test_combine_probabilities_uses_weights() -> None:
    first = np.array([[0.8, 0.1, 0.1]])
    second = np.array([[0.1, 0.8, 0.1]])

    result = combine_probabilities(
        [first, second],
        (0.75, 0.25),
    )

    expected = np.array([[0.625, 0.275, 0.1]])

    np.testing.assert_allclose(result, expected)


def test_combined_probabilities_sum_to_one() -> None:
    first = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]
    )
    second = np.array(
        [[0.4, 0.4, 0.2], [0.2, 0.3, 0.5]]
    )

    result = combine_probabilities(
        [first, second],
        (0.5, 0.5),
    )

    np.testing.assert_allclose(
        result.sum(axis=1),
        np.ones(2),
    )


def test_combine_probabilities_rejects_invalid_weights() -> None:
    probabilities = np.array([[0.3, 0.4, 0.3]])

    with pytest.raises(ValueError):
        combine_probabilities(
            [probabilities, probabilities],
            (0.8, 0.8),
        )


def test_combine_probabilities_rejects_different_shapes() -> None:
    first = np.array([[0.3, 0.4, 0.3]])
    second = np.array(
        [[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]]
    )

    with pytest.raises(ValueError):
        combine_probabilities(
            [first, second],
            (0.5, 0.5),
        )


def test_weight_search_returns_valid_weights() -> None:
    y_true = np.array([0, 1, 2, 0, 1, 2])

    logistic = np.eye(3)[y_true] * 0.8 + 0.2 / 3
    conv1d = np.full((6, 3), 1 / 3)
    lstm = np.full((6, 3), 1 / 3)

    result = search_three_model_weights(
        logistic,
        conv1d,
        lstm,
        y_true,
        step=0.10,
    )

    assert np.isclose(sum(result.weights), 1.0)
    assert result.macro_f1 == pytest.approx(1.0)