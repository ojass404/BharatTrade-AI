import numpy as np
import pytest
import tensorflow as tf

from ml_pipeline.src.conv1d_model import (
    build_conv1d_model,
    calculate_class_weights,
    set_random_seeds,
)


def test_calculate_class_weights_returns_all_classes() -> None:
    labels = np.array(
        [0, 0, 1, 1, 1, 1, 2, 2],
        dtype=np.int64,
    )

    weights = calculate_class_weights(labels)

    assert set(weights.keys()) == {0, 1, 2}
    assert weights[1] < weights[0]
    assert weights[1] < weights[2]


def test_calculate_class_weights_rejects_empty_array() -> None:
    with pytest.raises(ValueError):
        calculate_class_weights(np.array([], dtype=np.int64))


def test_conv1d_model_has_correct_input_and_output() -> None:
    model = build_conv1d_model(
        input_shape=(60, 19),
        number_of_classes=3,
    )

    assert model.input_shape == (None, 60, 19)
    assert model.output_shape == (None, 3)


def test_conv1d_model_outputs_probabilities() -> None:
    set_random_seeds(42)

    model = build_conv1d_model(
        input_shape=(60, 19),
        number_of_classes=3,
    )

    sample_batch = np.zeros(
        shape=(2, 60, 19),
        dtype=np.float32,
    )

    probabilities = model(
        sample_batch,
        training=False,
    ).numpy()

    assert probabilities.shape == (2, 3)

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(2),
        atol=1e-5,
    )


def test_conv1d_model_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError):
        build_conv1d_model(
            input_shape=(60,),
            number_of_classes=3,
        )