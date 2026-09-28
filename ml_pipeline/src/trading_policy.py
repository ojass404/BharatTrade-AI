from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


SELL = 0
HOLD = 1
BUY = 2


@dataclass(frozen=True)
class TradingPolicy:
    buy_threshold: float
    sell_threshold: float
    confidence_margin: float

    def __post_init__(self) -> None:
        for name, value in (
            ("buy_threshold", self.buy_threshold),
            ("sell_threshold", self.sell_threshold),
        ):
            if not 0 < value <= 1:
                raise ValueError(
                    f"{name} must be greater than 0 and at most 1."
                )

        if not 0 <= self.confidence_margin <= 1:
            raise ValueError(
                "confidence_margin must be between 0 and 1."
            )

    def to_dictionary(self) -> dict[str, float]:
        return asdict(self)


def validate_probabilities(
    probabilities: np.ndarray,
) -> np.ndarray:
    values = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            "Probabilities must have shape (samples, 3)."
        )

    if values.shape[0] == 0:
        raise ValueError(
            "At least one probability row is required."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Probabilities cannot contain NaN or infinity."
        )

    if np.any(values < 0):
        raise ValueError(
            "Probabilities cannot be negative."
        )

    row_totals = values.sum(axis=1, keepdims=True)

    if np.any(row_totals <= 0):
        raise ValueError(
            "Every probability row must have a positive total."
        )

    return values / row_totals


def probabilities_to_signals(
    probabilities: np.ndarray,
    policy: TradingPolicy,
) -> np.ndarray:
    """
    Convert SELL/HOLD/BUY probabilities into filtered signals.

    A directional signal is issued only when:

    1. Its probability exceeds the configured threshold.
    2. It exceeds both competing classes by the confidence margin.

    Otherwise, the system returns HOLD.
    """
    values = validate_probabilities(probabilities)

    signals = np.full(
        values.shape[0],
        HOLD,
        dtype=np.int64,
    )

    sell_probabilities = values[:, SELL]
    hold_probabilities = values[:, HOLD]
    buy_probabilities = values[:, BUY]

    strongest_sell_competitor = np.maximum(
        hold_probabilities,
        buy_probabilities,
    )

    strongest_buy_competitor = np.maximum(
        hold_probabilities,
        sell_probabilities,
    )

    sell_mask = (
        (sell_probabilities >= policy.sell_threshold)
        & (
            sell_probabilities
            - strongest_sell_competitor
            >= policy.confidence_margin
        )
    )

    buy_mask = (
        (buy_probabilities >= policy.buy_threshold)
        & (
            buy_probabilities
            - strongest_buy_competitor
            >= policy.confidence_margin
        )
    )

    signals[sell_mask] = SELL
    signals[buy_mask] = BUY

    return signals


def count_signals(
    signals: np.ndarray,
) -> dict[str, int]:
    values = np.asarray(signals, dtype=np.int64)

    return {
        "SELL": int(np.sum(values == SELL)),
        "HOLD": int(np.sum(values == HOLD)),
        "BUY": int(np.sum(values == BUY)),
    }