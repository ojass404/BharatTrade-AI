from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

import numpy as np


HOLD = 1


@dataclass(frozen=True)
class SignalAlignment:
    start_index: int
    end_index: int
    signals: np.ndarray


def normalize_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        timestamp = value
    else:
        text = str(value).replace("Z", "+00:00")
        timestamp = datetime.fromisoformat(text)

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)

    return timestamp


def align_target_signals(
    candle_timestamps: Sequence[datetime],
    target_timestamps: Sequence[object],
    predictions: Sequence[int],
) -> SignalAlignment:
    """
    Align predictions with their decision candles.

    target_timestamps represent the candle whose movement is being
    predicted. The signal is assigned to the immediately preceding
    chronological candle so execution occurs at target candle open.
    """
    if len(target_timestamps) != len(predictions):
        raise ValueError(
            "Target timestamp and prediction counts must match."
        )

    if len(candle_timestamps) < 2:
        raise ValueError("At least two candle timestamps are required.")

    if not target_timestamps:
        raise ValueError("At least one target timestamp is required.")

    normalized_candles = [
        normalize_timestamp(value)
        for value in candle_timestamps
    ]

    for index in range(1, len(normalized_candles)):
        if normalized_candles[index] <= normalized_candles[index - 1]:
            raise ValueError(
                "Candle timestamps must be strictly chronological."
            )

    candle_index_by_time = {
        timestamp: index
        for index, timestamp in enumerate(normalized_candles)
    }

    normalized_targets = [
        normalize_timestamp(value)
        for value in target_timestamps
    ]

    target_indices: list[int] = []

    for target_time in normalized_targets:
        if target_time not in candle_index_by_time:
            raise ValueError(
                f"No database candle exists for target {target_time}."
            )

        target_index = candle_index_by_time[target_time]

        if target_index == 0:
            raise ValueError(
                "The first target has no preceding decision candle."
            )

        target_indices.append(target_index)

    if target_indices != sorted(target_indices):
        raise ValueError(
            "Target timestamps must be chronological."
        )

    start_index = target_indices[0] - 1
    end_index = target_indices[-1]

    aligned_signals = np.full(
        end_index - start_index + 1,
        HOLD,
        dtype=np.int64,
    )

    for target_index, prediction in zip(
        target_indices,
        predictions,
    ):
        prediction_value = int(prediction)

        if prediction_value not in {0, 1, 2}:
            raise ValueError(
                "Predictions must be SELL=0, HOLD=1 or BUY=2."
            )

        decision_index = target_index - 1
        local_decision_index = decision_index - start_index

        aligned_signals[local_decision_index] = prediction_value

    return SignalAlignment(
        start_index=start_index,
        end_index=end_index,
        signals=aligned_signals,
    )