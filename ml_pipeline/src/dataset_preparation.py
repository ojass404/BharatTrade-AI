from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_pipeline.src.feature_engineering import FEATURE_COLUMNS


@dataclass(frozen=True)
class SplitBoundaries:
    train_end: int
    validation_end: int
    total_rows: int


def calculate_split_boundaries(
    row_count: int,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> SplitBoundaries:
    if row_count <= 0:
        raise ValueError("row_count must be positive.")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    if not 0 < validation_ratio < 1:
        raise ValueError(
            "validation_ratio must be between 0 and 1."
        )

    if train_ratio + validation_ratio >= 1:
        raise ValueError(
            "Training and validation ratios must total less than 1."
        )

    train_end = int(row_count * train_ratio)

    validation_end = train_end + int(
        row_count * validation_ratio
    )

    return SplitBoundaries(
        train_end=train_end,
        validation_end=validation_end,
        total_rows=row_count,
    )


def fit_training_scaler(
    frames: Sequence[pd.DataFrame],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
    train_ratio: float = 0.70,
) -> StandardScaler:
    training_parts: list[np.ndarray] = []

    for frame in frames:
        boundaries = calculate_split_boundaries(
            row_count=len(frame),
            train_ratio=train_ratio,
        )

        training_part = frame.iloc[
            : boundaries.train_end
        ][list(feature_columns)]

        training_parts.append(
            training_part.to_numpy(dtype=np.float64)
        )

    if not training_parts:
        raise ValueError(
            "At least one feature frame is required."
        )

    combined_training_data = np.concatenate(
        training_parts,
        axis=0,
    )

    scaler = StandardScaler()
    scaler.fit(combined_training_data)

    return scaler


def create_sequences(
    scaled_features: np.ndarray,
    labels: np.ndarray,
    target_indices: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    sequence_labels: list[int] = []

    for target_index in target_indices:
        start_index = target_index - sequence_length + 1

        if start_index < 0:
            continue

        sequences.append(
            scaled_features[
                start_index : target_index + 1
            ]
        )

        sequence_labels.append(
            int(labels[target_index])
        )

    if not sequences:
        return (
            np.empty(
                (
                    0,
                    sequence_length,
                    scaled_features.shape[1],
                ),
                dtype=np.float32,
            ),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(sequence_labels, dtype=np.int64),
    )


def prepare_stock_sequences(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    sequence_length: int = 60,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> dict[str, np.ndarray | SplitBoundaries]:
    if sequence_length < 2:
        raise ValueError(
            "sequence_length must be at least 2."
        )

    required_columns = set(FEATURE_COLUMNS).union(
        {
            "timestamp",
            "target_time",
            "target_class",
        }
    )

    missing_columns = required_columns.difference(
        frame.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing dataset columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = frame.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        utc=True,
    )

    data["target_time"] = pd.to_datetime(
        data["target_time"],
        utc=True,
    )

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    boundaries = calculate_split_boundaries(
        row_count=len(data),
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    if boundaries.train_end <= sequence_length:
        raise ValueError(
            "Training split is too small for the sequence length."
        )

    feature_values = data[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    scaled_features = scaler.transform(
        feature_values
    )

    labels = data[
        "target_class"
    ].to_numpy(dtype=np.int64)

    # The final training row is excluded because its target
    # belongs to the validation period.
    train_indices = np.arange(
        sequence_length - 1,
        boundaries.train_end - 1,
    )

    # The final validation row is excluded because its target
    # belongs to the test period.
    validation_indices = np.arange(
        boundaries.train_end,
        boundaries.validation_end - 1,
    )

    test_indices = np.arange(
        boundaries.validation_end,
        boundaries.total_rows,
    )

    x_train, y_train = create_sequences(
        scaled_features,
        labels,
        train_indices,
        sequence_length,
    )

    x_validation, y_validation = create_sequences(
        scaled_features,
        labels,
        validation_indices,
        sequence_length,
    )

    x_test, y_test = create_sequences(
        scaled_features,
        labels,
        test_indices,
        sequence_length,
    )

    target_times = data[
        "target_time"
    ].astype(str).to_numpy()

    return {
        "x_train": x_train,
        "y_train": y_train,
        "train_times": target_times[train_indices],
        "x_validation": x_validation,
        "y_validation": y_validation,
        "validation_times": target_times[
            validation_indices
        ],
        "x_test": x_test,
        "y_test": y_test,
        "test_times": target_times[test_indices],
        "boundaries": boundaries,
    }