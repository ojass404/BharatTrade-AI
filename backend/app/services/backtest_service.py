from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import floor, sqrt
from typing import Sequence

import numpy as np


SELL = 0
HOLD = 1
BUY = 2


@dataclass(frozen=True)
class BacktestCandle:
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    position_size_fraction: float = 0.25
    transaction_cost_bps: float = 10.0
    stop_loss_fraction: float = 0.02
    annual_trading_days: int = 252
    candles_per_day: int = 25

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")

        if not 0 < self.position_size_fraction <= 1:
            raise ValueError(
                "position_size_fraction must be between 0 and 1."
            )

        if self.transaction_cost_bps < 0:
            raise ValueError(
                "transaction_cost_bps cannot be negative."
            )

        if not 0 < self.stop_loss_fraction < 1:
            raise ValueError(
                "stop_loss_fraction must be between 0 and 1."
            )


@dataclass(frozen=True)
class BacktestTrade:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    entry_cost: float
    exit_cost: float
    net_profit_loss: float
    exit_reason: str


@dataclass
class BacktestResult:
    initial_capital: float
    final_capital: float
    total_return_percent: float
    sharpe_ratio: float
    maximum_drawdown_percent: float
    win_rate_percent: float
    total_trades: int
    equity_curve: list[float] = field(default_factory=list)
    equity_timestamps: list[datetime] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)


def calculate_maximum_drawdown(
    equity_curve: Sequence[float],
) -> float:
    values = np.asarray(equity_curve, dtype=np.float64)

    if values.size == 0:
        return 0.0

    running_peaks = np.maximum.accumulate(values)

    drawdowns = np.divide(
        running_peaks - values,
        running_peaks,
        out=np.zeros_like(values),
        where=running_peaks > 0,
    )

    return float(np.max(drawdowns) * 100)


def calculate_sharpe_ratio(
    equity_curve: Sequence[float],
    annual_trading_days: int = 252,
    candles_per_day: int = 25,
) -> float:
    values = np.asarray(equity_curve, dtype=np.float64)

    if values.size < 3:
        return 0.0

    returns = np.diff(values) / values[:-1]
    returns = returns[np.isfinite(returns)]

    if returns.size < 2:
        return 0.0

    standard_deviation = float(np.std(returns, ddof=1))

    if np.isclose(standard_deviation, 0):
        return 0.0

    annualisation_factor = sqrt(
        annual_trading_days * candles_per_day
    )

    return float(
        np.mean(returns)
        / standard_deviation
        * annualisation_factor
    )


def _validate_candles(
    candles: Sequence[BacktestCandle],
) -> None:
    if len(candles) < 2:
        raise ValueError("At least two candles are required.")

    previous_timestamp: datetime | None = None

    for candle in candles:
        prices = (
            candle.open_price,
            candle.high_price,
            candle.low_price,
            candle.close_price,
        )

        if any(price <= 0 for price in prices):
            raise ValueError("All candle prices must be positive.")

        if candle.high_price < candle.low_price:
            raise ValueError(
                "Candle high price cannot be below its low price."
            )

        if previous_timestamp is not None:
            if candle.timestamp <= previous_timestamp:
                raise ValueError(
                    "Candles must be strictly chronological."
                )

        previous_timestamp = candle.timestamp


def run_backtest(
    candles: Sequence[BacktestCandle],
    signals: Sequence[int],
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run a long-only, next-candle-execution backtest.

    A signal calculated from candle i is executed using the open price
    of candle i + 1. This prevents executing on information that was
    unavailable when the signal was generated.
    """
    _validate_candles(candles)

    if len(signals) != len(candles):
        raise ValueError(
            "The number of signals must equal the number of candles."
        )

    if any(signal not in {SELL, HOLD, BUY} for signal in signals):
        raise ValueError("Signals must be SELL=0, HOLD=1 or BUY=2.")

    settings = config or BacktestConfig()
    transaction_rate = settings.transaction_cost_bps / 10_000

    cash = float(settings.initial_capital)
    quantity = 0

    entry_time: datetime | None = None
    entry_price = 0.0
    entry_cost = 0.0
    entry_total = 0.0
    stop_price = 0.0

    trades: list[BacktestTrade] = []
    equity_curve = [cash]
    equity_timestamps = [candles[0].timestamp]

    def close_position(
        execution_time: datetime,
        execution_price: float,
        reason: str,
    ) -> None:
        nonlocal cash
        nonlocal quantity
        nonlocal entry_time
        nonlocal entry_price
        nonlocal entry_cost
        nonlocal entry_total
        nonlocal stop_price

        gross_value = quantity * execution_price
        exit_cost = gross_value * transaction_rate
        net_proceeds = gross_value - exit_cost
        net_profit_loss = net_proceeds - entry_total

        cash += net_proceeds

        trades.append(
            BacktestTrade(
                entry_time=entry_time,
                exit_time=execution_time,
                entry_price=entry_price,
                exit_price=execution_price,
                quantity=quantity,
                entry_cost=entry_cost,
                exit_cost=exit_cost,
                net_profit_loss=net_profit_loss,
                exit_reason=reason,
            )
        )

        quantity = 0
        entry_time = None
        entry_price = 0.0
        entry_cost = 0.0
        entry_total = 0.0
        stop_price = 0.0

    for signal_index in range(len(candles) - 1):
        signal = int(signals[signal_index])
        execution_candle = candles[signal_index + 1]
        exited_this_candle = False

        if quantity > 0:
            if execution_candle.open_price <= stop_price:
                close_position(
                    execution_time=execution_candle.timestamp,
                    execution_price=execution_candle.open_price,
                    reason="STOP_LOSS_GAP",
                )
                exited_this_candle = True

            elif execution_candle.low_price <= stop_price:
                close_position(
                    execution_time=execution_candle.timestamp,
                    execution_price=stop_price,
                    reason="STOP_LOSS",
                )
                exited_this_candle = True

            elif signal == SELL:
                close_position(
                    execution_time=execution_candle.timestamp,
                    execution_price=execution_candle.open_price,
                    reason="SELL_SIGNAL",
                )
                exited_this_candle = True

        if (
            quantity == 0
            and signal == BUY
            and not exited_this_candle
        ):
            allocation = cash * settings.position_size_fraction

            unit_cost = (
                execution_candle.open_price
                * (1 + transaction_rate)
            )

            purchasable_quantity = floor(allocation / unit_cost)

            if purchasable_quantity > 0:
                gross_value = (
                    purchasable_quantity
                    * execution_candle.open_price
                )
                calculated_entry_cost = (
                    gross_value * transaction_rate
                )
                total_required = gross_value + calculated_entry_cost

                if total_required <= cash:
                    cash -= total_required
                    quantity = purchasable_quantity

                    entry_time = execution_candle.timestamp
                    entry_price = execution_candle.open_price
                    entry_cost = calculated_entry_cost
                    entry_total = total_required

                    stop_price = entry_price * (
                        1 - settings.stop_loss_fraction
                    )

        marked_equity = (
            cash
            + quantity * execution_candle.close_price
        )

        equity_curve.append(float(marked_equity))
        equity_timestamps.append(execution_candle.timestamp)

    if quantity > 0:
        final_candle = candles[-1]

        close_position(
            execution_time=final_candle.timestamp,
            execution_price=final_candle.close_price,
            reason="END_OF_BACKTEST",
        )

        equity_curve[-1] = float(cash)

    final_capital = float(cash)

    total_return_percent = (
        (final_capital - settings.initial_capital)
        / settings.initial_capital
        * 100
    )

    winning_trades = sum(
        trade.net_profit_loss > 0
        for trade in trades
    )

    win_rate_percent = (
        winning_trades / len(trades) * 100
        if trades
        else 0.0
    )

    return BacktestResult(
        initial_capital=float(settings.initial_capital),
        final_capital=final_capital,
        total_return_percent=float(total_return_percent),
        sharpe_ratio=calculate_sharpe_ratio(
            equity_curve,
            annual_trading_days=settings.annual_trading_days,
            candles_per_day=settings.candles_per_day,
        ),
        maximum_drawdown_percent=calculate_maximum_drawdown(
            equity_curve
        ),
        win_rate_percent=float(win_rate_percent),
        total_trades=len(trades),
        equity_curve=equity_curve,
        equity_timestamps=equity_timestamps,
        trades=trades,
    )