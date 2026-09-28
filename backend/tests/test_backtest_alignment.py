from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from ml_pipeline.src.backtest_alignment import (
    align_target_signals,
    normalize_timestamp,
)


START = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)


def test_target_prediction_is_assigned_to_previous_candle() -> None:
    candle_times = [
        START + timedelta(minutes=15 * index)
        for index in range(4)
    ]

    alignment = align_target_signals(
        candle_timestamps=candle_times,
        target_timestamps=[
            candle_times[1],
            candle_times[2],
            candle_times[3],
        ],
        predictions=[2, 1, 0],
    )

    assert alignment.start_index == 0
    assert alignment.end_index == 3

    np.testing.assert_array_equal(
        alignment.signals,
        np.array([2, 1, 0, 1]),
    )


def test_alignment_uses_previous_trading_candle_across_gap() -> None:
    friday_close = datetime(
        2026, 1, 2, 10, 0, tzinfo=timezone.utc
    )

    monday_open = datetime(
        2026, 1, 5, 3, 45, tzinfo=timezone.utc
    )

    monday_next = datetime(
        2026, 1, 5, 4, 0, tzinfo=timezone.utc
    )

    alignment = align_target_signals(
        candle_timestamps=[
            friday_close,
            monday_open,
            monday_next,
        ],
        target_timestamps=[monday_open],
        predictions=[2],
    )

    assert alignment.start_index == 0
    assert alignment.end_index == 1
    assert alignment.signals.tolist() == [2, 1]


def test_timestamp_strings_are_normalized_to_utc() -> None:
    timestamp = normalize_timestamp(
        "2026-04-21 05:45:00+00:00"
    )

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


def test_missing_target_candle_is_rejected() -> None:
    candle_times = [
        START,
        START + timedelta(minutes=15),
    ]

    with pytest.raises(ValueError):
        align_target_signals(
            candle_timestamps=candle_times,
            target_timestamps=[
                START + timedelta(minutes=30)
            ],
            predictions=[2],
        )


def test_invalid_prediction_is_rejected() -> None:
    candle_times = [
        START,
        START + timedelta(minutes=15),
    ]

    with pytest.raises(ValueError):
        align_target_signals(
            candle_timestamps=candle_times,
            target_timestamps=[candle_times[1]],
            predictions=[99],
        )