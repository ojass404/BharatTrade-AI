from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers

from ml_pipeline.src.conv1d_model import (
    calculate_class_weights,
    set_random_seeds,
)


def build_lstm_model(
    input_shape: tuple[int, int],
    number_of_classes: int = 3,
    learning_rate: float = 0.001,
) -> keras.Model:
    """
    Build an LSTM model for sequential stock-market classification.

    Default input:
        60 previous candles × 19 engineered features
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

    x = layers.LSTM(
        units=64,
        return_sequences=True,
        dropout=0.20,
        name="lstm_block_1",
    )(inputs)

    x = layers.LayerNormalization(
        name="layer_normalization_1",
    )(x)

    x = layers.LSTM(
        units=32,
        return_sequences=False,
        dropout=0.20,
        name="lstm_block_2",
    )(x)

    x = layers.LayerNormalization(
        name="layer_normalization_2",
    )(x)

    x = layers.Dense(
        units=64,
        activation="relu",
        name="dense_features",
    )(x)

    x = layers.Dropout(
        rate=0.30,
        name="dense_dropout",
    )(x)

    outputs = layers.Dense(
        units=number_of_classes,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="bharattrade_lstm",
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