import numpy as np
import pandas as pd

from ml_pipeline.src.dataset_preparation import (
    calculate_split_boundaries,
    fit_training_scaler,
    prepare_stock_sequences,
)
from ml_pipeline.src.feature_engineering import FEATURE_COLUMNS


def create_feature_frame(
    row_count: int = 500,
) -> pd.DataFrame:
    generator = np.random.default_rng(42)

    timestamps = pd.date_range(
        start="2025-01-01 09:15",
        periods=row_count,
        freq="15min",
        tz="Asia/Kolkata",
    )

    data = {
        column: generator.normal(
            loc=0,
            scale=1,
            size=row_count,
        )
        for column in FEATURE_COLUMNS
    }

    frame = pd.DataFrame(data)

    frame["timestamp"] = timestamps
    frame["target_time"] = timestamps + pd.Timedelta(
        minutes=15
    )

    frame["target_class"] = generator.integers(
        low=0,
        high=3,
        size=row_count,
    )

    return frame


def test_calculate_split_boundaries() -> None:
    boundaries = calculate_split_boundaries(1000)

    assert boundaries.train_end == 700
    assert boundaries.validation_end == 850
    assert boundaries.total_rows == 1000


def test_scaler_is_fitted_only_on_training_rows() -> None:
    frame = create_feature_frame(500)

    scaler = fit_training_scaler([frame])

    expected_training_mean = frame.iloc[:350][
        FEATURE_COLUMNS
    ].mean().to_numpy()

    np.testing.assert_allclose(
        scaler.mean_,
        expected_training_mean,
    )


def test_sequence_shapes_are_correct() -> None:
    frame = create_feature_frame(500)
    scaler = fit_training_scaler([frame])

    result = prepare_stock_sequences(
        frame=frame,
        scaler=scaler,
        sequence_length=60,
    )

    assert result["x_train"].shape[1:] == (
        60,
        len(FEATURE_COLUMNS),
    )

    assert result["x_validation"].shape[1:] == (
        60,
        len(FEATURE_COLUMNS),
    )

    assert result["x_test"].shape[1:] == (
        60,
        len(FEATURE_COLUMNS),
    )


def test_chronological_splits_do_not_overlap() -> None:
    frame = create_feature_frame(500)
    scaler = fit_training_scaler([frame])

    result = prepare_stock_sequences(
        frame=frame,
        scaler=scaler,
        sequence_length=60,
    )

    train_times = pd.to_datetime(
        result["train_times"],
        utc=True,
    )

    validation_times = pd.to_datetime(
        result["validation_times"],
        utc=True,
    )

    test_times = pd.to_datetime(
        result["test_times"],
        utc=True,
    )

    assert train_times.max() < validation_times.min()
    assert validation_times.max() < test_times.min()