from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def calculate_class_weights(y_train: np.ndarray) -> dict[int, float]:
    """
    Calculate balanced class weights for SELL, HOLD and BUY.

    Minority classes receive higher weights so the model does not
    learn to predict HOLD for almost every sequence.
    """
    labels = np.asarray(y_train, dtype=np.int64)

    if labels.ndim != 1:
        raise ValueError("y_train must be a one-dimensional array.")

    if labels.size == 0:
        raise ValueError("y_train cannot be empty.")

    classes, counts = np.unique(labels, return_counts=True)
    total_samples = labels.size
    class_count = len(classes)

    return {
        int(class_label): float(
            total_samples / (class_count * class_samples)
        )
        for class_label, class_samples in zip(classes, counts)
    }


def build_conv1d_model(
    input_shape: tuple[int, int],
    number_of_classes: int = 3,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Build a compact Conv1D classifier for stock-market sequences.

    Input shape:
        sequence_length × feature_count

    Default project input:
        60 candles × 19 engineered features
    """
    if len(input_shape) != 2:
        raise ValueError(
            "input_shape must contain sequence length and feature count."
        )

    if input_shape[0] <= 0 or input_shape[1] <= 0:
        raise ValueError("Input dimensions must be positive.")

    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least 2.")

    inputs = keras.Input(
        shape=input_shape,
        name="market_sequence",
    )

    x = layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding="causal",
        name="conv1d_block_1",
    )(inputs)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Activation("relu", name="relu_1")(x)
    x = layers.MaxPooling1D(
        pool_size=2,
        name="max_pool_1",
    )(x)
    x = layers.Dropout(0.20, name="dropout_1")(x)

    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="causal",
        name="conv1d_block_2",
    )(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Activation("relu", name="relu_2")(x)
    x = layers.MaxPooling1D(
        pool_size=2,
        name="max_pool_2",
    )(x)
    x = layers.Dropout(0.25, name="dropout_2")(x)

    x = layers.Conv1D(
        filters=128,
        kernel_size=3,
        padding="causal",
        name="conv1d_block_3",
    )(x)
    x = layers.BatchNormalization(name="batch_norm_3")(x)
    x = layers.Activation("relu", name="relu_3")(x)

    x = layers.GlobalAveragePooling1D(
        name="global_average_pooling",
    )(x)

    x = layers.Dense(
        units=64,
        activation="relu",
        name="dense_features",
    )(x)
    x = layers.Dropout(0.30, name="dense_dropout")(x)

    outputs = layers.Dense(
        units=number_of_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="bharattrade_conv1d",
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def set_random_seeds(seed: int = 42) -> None:
    """Set NumPy and TensorFlow random seeds."""
    np.random.seed(seed)
    tf.random.set_seed(seed)