from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_pipeline.src.dataset_preparation_v3 import (
    prepare_stock_sequences_v3,
)
from ml_pipeline.src.feature_engineering import FEATURE_COLUMNS


def create_v3_frame(
    row_count: int = 300,
) -> pd.DataFrame:
    start = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    timestamps = [
        start + timedelta(minutes=15 * index)
        for index in range(row_count)
    ]

    frame = pd.DataFrame(
        {
            column: np.linspace(
                0,
                1,
                row_count,
            )
            for column in FEATURE_COLUMNS
        }
    )

    frame["timestamp"] = timestamps

    frame["target_time"] = [
        timestamp + timedelta(minutes=60)
        for timestamp in timestamps
    ]

    frame["target_class"] = np.arange(
        row_count
    ) % 3

    return frame


def create_scaler(
    frame: pd.DataFrame,
) -> StandardScaler:
    scaler = StandardScaler()

    scaler.fit(
        frame[FEATURE_COLUMNS].to_numpy(
            dtype=np.float64
        )
    )

    return scaler


def test_v3_sequence_shapes_are_correct() -> None:
    frame = create_v3_frame()
    scaler = create_scaler(frame)

    prepared = prepare_stock_sequences_v3(
        frame=frame,
        scaler=scaler,
        sequence_length=60,
    )

    assert prepared["x_train"].shape[1:] == (
        60,
        len(FEATURE_COLUMNS),
    )

    assert len(prepared["x_train"]) == len(
        prepared["y_train"]
    )


def test_training_targets_do_not_enter_validation() -> None:
    frame = create_v3_frame()
    scaler = create_scaler(frame)

    prepared = prepare_stock_sequences_v3(
        frame=frame,
        scaler=scaler,
        sequence_length=60,
    )

    train_targets = pd.to_datetime(
        prepared["train_target_times"],
        utc=True,
    )

    assert (
        train_targets
        < prepared["validation_start_time"]
    ).all()


def test_validation_targets_do_not_enter_test() -> None:
    frame = create_v3_frame()
    scaler = create_scaler(frame)

    prepared = prepare_stock_sequences_v3(
        frame=frame,
        scaler=scaler,
        sequence_length=60,
    )

    validation_targets = pd.to_datetime(
        prepared["validation_target_times"],
        utc=True,
    )

    assert (
        validation_targets
        < prepared["test_start_time"]
    ).all()


def test_v3_timestamp_arrays_are_not_object_dtype() -> None:
    frame = create_v3_frame()
    scaler = create_scaler(frame)

    prepared = prepare_stock_sequences_v3(
        frame=frame,
        scaler=scaler,
        sequence_length=60,
    )

    assert (
        prepared["test_times"].dtype.kind
        in {"U", "S"}
    )

    assert (
        prepared["test_target_times"].dtype.kind
        in {"U", "S"}
    )