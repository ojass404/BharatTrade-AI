from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.src.feature_engineering import (
    create_features,
)


def create_test_candles(
    days: int = 3,
    candles_per_day: int = 25,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    starting_price = 1_000.0

    for day_index in range(days):
        day_start = datetime(
            2026,
            1,
            5 + day_index,
            3,
            45,
            tzinfo=timezone.utc,
        )

        for candle_index in range(candles_per_day):
            index = (
                day_index * candles_per_day
                + candle_index
            )

            close = starting_price + index

            records.append(
                {
                    "timestamp": (
                        day_start
                        + timedelta(
                            minutes=15 * candle_index
                        )
                    ),
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 100_000 + index * 100,
                }
            )

    return pd.DataFrame(records)


def test_four_candle_target_uses_fourth_future_candle() -> None:
    candles = create_test_candles()

    features = create_features(
        candles,
        target_threshold=0.004,
        target_horizon=4,
        same_session_target=True,
    )

    first_row = features.iloc[0]

    original_index = candles.index[
        candles["timestamp"]
        == first_row["timestamp"]
    ][0]

    expected_target_time = candles.iloc[
        original_index + 4
    ]["timestamp"]

    assert (
        first_row["target_time"]
        == expected_target_time
    )


def test_cost_aware_targets_do_not_cross_sessions() -> None:
    features = create_features(
        create_test_candles(),
        target_threshold=0.004,
        target_horizon=4,
        same_session_target=True,
    )

    current_dates = (
        pd.to_datetime(
            features["timestamp"],
            utc=True,
        )
        .dt.tz_convert("Asia/Kolkata")
        .dt.date
    )

    target_dates = (
        pd.to_datetime(
            features["target_time"],
            utc=True,
        )
        .dt.tz_convert("Asia/Kolkata")
        .dt.date
    )

    assert (current_dates == target_dates).all()


def test_final_four_candles_of_session_are_removed() -> None:
    candles = create_test_candles(
        days=3,
        candles_per_day=25,
    )

    features = create_features(
        candles,
        target_threshold=0.004,
        target_horizon=4,
        same_session_target=True,
    )

    feature_times = set(
        pd.to_datetime(
            features["timestamp"],
            utc=True,
        )
    )

    final_day = candles.iloc[-25:]

    for timestamp in final_day[
        "timestamp"
    ].tail(4):
        assert timestamp not in feature_times


def test_target_threshold_is_applied() -> None:
    candles = create_test_candles(
        days=4,
        candles_per_day=25,
    )

    features = create_features(
        candles,
        target_threshold=0.004,
        target_horizon=4,
        same_session_target=True,
    )

    sell_rows = features[
        features["target_return"] < -0.004
    ]

    hold_rows = features[
        features["target_return"].between(
            -0.004,
            0.004,
            inclusive="both",
        )
    ]

    buy_rows = features[
        features["target_return"] > 0.004
    ]

    assert (
        sell_rows["target_class"] == 0
    ).all()

    assert (
        hold_rows["target_class"] == 1
    ).all()

    assert (
        buy_rows["target_class"] == 2
    ).all()


def test_invalid_target_horizon_is_rejected() -> None:
    with pytest.raises(ValueError):
        create_features(
            create_test_candles(),
            target_horizon=0,
        )