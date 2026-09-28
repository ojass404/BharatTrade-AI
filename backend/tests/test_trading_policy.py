import numpy as np
import pytest

from ml_pipeline.src.trading_policy import (
    BUY,
    HOLD,
    SELL,
    TradingPolicy,
    count_signals,
    probabilities_to_signals,
)


def test_high_confidence_predictions_become_signals() -> None:
    probabilities = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.10, 0.20, 0.70],
        ]
    )

    policy = TradingPolicy(
        buy_threshold=0.60,
        sell_threshold=0.60,
        confidence_margin=0.20,
    )

    signals = probabilities_to_signals(
        probabilities,
        policy,
    )

    np.testing.assert_array_equal(
        signals,
        np.array([SELL, BUY]),
    )


def test_low_confidence_predictions_become_hold() -> None:
    probabilities = np.array(
        [
            [0.34, 0.32, 0.34],
            [0.30, 0.35, 0.35],
        ]
    )

    policy = TradingPolicy(
        buy_threshold=0.40,
        sell_threshold=0.40,
        confidence_margin=0.05,
    )

    signals = probabilities_to_signals(
        probabilities,
        policy,
    )

    np.testing.assert_array_equal(
        signals,
        np.array([HOLD, HOLD]),
    )


def test_threshold_without_margin_is_not_enough() -> None:
    probabilities = np.array(
        [
            [0.46, 0.10, 0.44],
        ]
    )

    policy = TradingPolicy(
        buy_threshold=0.40,
        sell_threshold=0.40,
        confidence_margin=0.05,
    )

    signals = probabilities_to_signals(
        probabilities,
        policy,
    )

    assert signals[0] == HOLD


def test_probability_rows_are_normalized() -> None:
    probabilities = np.array(
        [
            [7.0, 2.0, 1.0],
            [1.0, 2.0, 7.0],
        ]
    )

    policy = TradingPolicy(
        buy_threshold=0.60,
        sell_threshold=0.60,
        confidence_margin=0.20,
    )

    signals = probabilities_to_signals(
        probabilities,
        policy,
    )

    np.testing.assert_array_equal(
        signals,
        np.array([SELL, BUY]),
    )


def test_invalid_probability_shape_is_rejected() -> None:
    policy = TradingPolicy(
        buy_threshold=0.50,
        sell_threshold=0.50,
        confidence_margin=0.10,
    )

    with pytest.raises(ValueError):
        probabilities_to_signals(
            np.array([0.2, 0.5, 0.3]),
            policy,
        )


def test_signal_counts_are_calculated() -> None:
    counts = count_signals(
        np.array(
            [SELL, HOLD, BUY, HOLD, HOLD, BUY]
        )
    )

    assert counts == {
        "SELL": 1,
        "HOLD": 3,
        "BUY": 2,
    }