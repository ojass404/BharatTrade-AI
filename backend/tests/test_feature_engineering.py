import numpy as np
import pandas as pd

from ml_pipeline.src.feature_engineering import (
    FEATURE_COLUMNS,
    create_features,
)


def create_sample_candles(row_count: int = 150) -> pd.DataFrame:
    timestamps = pd.date_range(
        start="2026-01-01 09:15",
        periods=row_count,
        freq="15min",
        tz="Asia/Kolkata",
    )

    sequence = np.arange(row_count)

    close = (
        100
        + sequence * 0.05
        + np.sin(sequence / 5)
    )

    open_price = close - 0.10
    high = np.maximum(open_price, close) + 0.40
    low = np.minimum(open_price, close) - 0.40
    volume = 1000 + (sequence % 20) * 100

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_create_features_generates_expected_columns() -> None:
    candles = create_sample_candles()

    result = create_features(candles)

    for column in FEATURE_COLUMNS:
        assert column in result.columns

    assert "target_return" in result.columns
    assert "target_class" in result.columns
    assert "target_time" in result.columns


def test_create_features_contains_no_missing_values() -> None:
    candles = create_sample_candles()

    result = create_features(candles)

    checked_columns = FEATURE_COLUMNS + [
        "target_return",
        "target_class",
        "target_time",
    ]

    assert not result[checked_columns].isna().any().any()


def test_target_classes_are_valid() -> None:
    candles = create_sample_candles()

    result = create_features(candles)

    assert set(result["target_class"].unique()).issubset(
        {0, 1, 2}
    )


def test_feature_rows_are_chronological() -> None:
    candles = create_sample_candles()

    result = create_features(candles)

    assert result["timestamp"].is_monotonic_increasing
    assert (
        result["target_time"] > result["timestamp"]
    ).all()