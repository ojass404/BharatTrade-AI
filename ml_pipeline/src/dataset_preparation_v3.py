from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_pipeline.src.dataset_preparation import (
    SplitBoundaries,
    calculate_split_boundaries,
    create_sequences,
)
from ml_pipeline.src.feature_engineering import FEATURE_COLUMNS


def prepare_stock_sequences_v3(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    sequence_length: int = 60,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> dict[str, object]:
    """
    Prepare V3 sequences while preventing four-candle targets from
    crossing chronological split boundaries.
    """
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

    boundaries: SplitBoundaries = (
        calculate_split_boundaries(
            row_count=len(data),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
    )

    if boundaries.train_end <= sequence_length:
        raise ValueError(
            "Training split is too small for the sequence length."
        )

    validation_start_time = data.iloc[
        boundaries.train_end
    ]["timestamp"]

    test_start_time = data.iloc[
        boundaries.validation_end
    ]["timestamp"]

    feature_values = data[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float64)

    scaled_features = scaler.transform(
        feature_values
    )

    labels = data[
        "target_class"
    ].to_numpy(dtype=np.int64)

    target_times = data[
        "target_time"
    ]

    train_candidates = np.arange(
        sequence_length - 1,
        boundaries.train_end,
    )

    train_indices = train_candidates[
        target_times.iloc[
            train_candidates
        ].to_numpy()
        < validation_start_time
    ]

    validation_candidates = np.arange(
        boundaries.train_end,
        boundaries.validation_end,
    )

    validation_indices = validation_candidates[
        target_times.iloc[
            validation_candidates
        ].to_numpy()
        < test_start_time
    ]

    test_indices = np.arange(
        boundaries.validation_end,
        boundaries.total_rows,
    )

    x_train, y_train = create_sequences(
        scaled_features=scaled_features,
        labels=labels,
        target_indices=train_indices,
        sequence_length=sequence_length,
    )

    x_validation, y_validation = create_sequences(
        scaled_features=scaled_features,
        labels=labels,
        target_indices=validation_indices,
        sequence_length=sequence_length,
    )

    x_test, y_test = create_sequences(
        scaled_features=scaled_features,
        labels=labels,
        target_indices=test_indices,
        sequence_length=sequence_length,
    )

    timestamp_strings = data[
        "timestamp"
    ].astype(str).to_numpy(dtype="U40")

    target_time_strings = data[
        "target_time"
    ].astype(str).to_numpy(dtype="U40")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "train_times": timestamp_strings[
            train_indices
        ],
        "train_target_times": target_time_strings[
            train_indices
        ],
        "x_validation": x_validation,
        "y_validation": y_validation,
        "validation_times": timestamp_strings[
            validation_indices
        ],
        "validation_target_times": target_time_strings[
            validation_indices
        ],
        "x_test": x_test,
        "y_test": y_test,
        "test_times": timestamp_strings[
            test_indices
        ],
        "test_target_times": target_time_strings[
            test_indices
        ],
        "boundaries": boundaries,
        "validation_start_time": (
            validation_start_time
        ),
        "test_start_time": test_start_time,
    }