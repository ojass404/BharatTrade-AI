from datetime import datetime, timedelta, timezone

import pytest

from backend.app.services.backtest_service import (
    BUY,
    HOLD,
    SELL,
    BacktestCandle,
    BacktestConfig,
    calculate_maximum_drawdown,
    run_backtest,
)


START = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)


def candle(
    index: int,
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
) -> BacktestCandle:
    return BacktestCandle(
        timestamp=START + timedelta(minutes=15 * index),
        open_price=open_price,
        high_price=high_price,
        low_price=low_price,
        close_price=close_price,
    )


def test_buy_and_sell_execute_on_next_candle_open() -> None:
    candles = [
        candle(0, 100, 102, 99, 101),
        candle(1, 110, 115, 108, 114),
        candle(2, 120, 123, 118, 121),
    ]

    result = run_backtest(
        candles=candles,
        signals=[BUY, SELL, HOLD],
        config=BacktestConfig(
            initial_capital=1_000,
            position_size_fraction=1.0,
            transaction_cost_bps=0,
            stop_loss_fraction=0.10,
        ),
    )

    assert result.total_trades == 1
    assert result.trades[0].entry_price == 110
    assert result.trades[0].exit_price == 120
    assert result.trades[0].quantity == 9
    assert result.final_capital == pytest.approx(1_090)


def test_transaction_cost_reduces_final_capital() -> None:
    candles = [
        candle(0, 100, 101, 99, 100),
        candle(1, 100, 103, 99, 102),
        candle(2, 110, 112, 108, 111),
    ]

    no_cost = run_backtest(
        candles,
        [BUY, SELL, HOLD],
        BacktestConfig(
            initial_capital=10_000,
            position_size_fraction=1.0,
            transaction_cost_bps=0,
            stop_loss_fraction=0.10,
        ),
    )

    with_cost = run_backtest(
        candles,
        [BUY, SELL, HOLD],
        BacktestConfig(
            initial_capital=10_000,
            position_size_fraction=1.0,
            transaction_cost_bps=10,
            stop_loss_fraction=0.10,
        ),
    )

    assert with_cost.final_capital < no_cost.final_capital
    assert with_cost.trades[0].entry_cost > 0
    assert with_cost.trades[0].exit_cost > 0


def test_stop_loss_uses_later_candle_price_range() -> None:
    candles = [
        candle(0, 100, 101, 99, 100),
        candle(1, 100, 103, 99, 102),
        candle(2, 98, 100, 94, 96),
    ]

    result = run_backtest(
        candles,
        [BUY, HOLD, HOLD],
        BacktestConfig(
            initial_capital=10_000,
            position_size_fraction=1.0,
            transaction_cost_bps=0,
            stop_loss_fraction=0.05,
        ),
    )

    assert result.total_trades == 1
    assert result.trades[0].entry_price == 100
    assert result.trades[0].exit_price == 95
    assert result.trades[0].exit_reason == "STOP_LOSS"


def test_open_gap_executes_below_stop_price() -> None:
    candles = [
        candle(0, 100, 101, 99, 100),
        candle(1, 100, 103, 99, 102),
        candle(2, 90, 93, 88, 91),
    ]

    result = run_backtest(
        candles,
        [BUY, HOLD, HOLD],
        BacktestConfig(
            initial_capital=10_000,
            position_size_fraction=1.0,
            transaction_cost_bps=0,
            stop_loss_fraction=0.05,
        ),
    )

    assert result.trades[0].exit_price == 90
    assert result.trades[0].exit_reason == "STOP_LOSS_GAP"


def test_open_position_is_closed_at_end() -> None:
    candles = [
        candle(0, 100, 101, 99, 100),
        candle(1, 100, 106, 99, 105),
    ]

    result = run_backtest(
        candles,
        [BUY, HOLD],
        BacktestConfig(
            initial_capital=1_000,
            position_size_fraction=1.0,
            transaction_cost_bps=0,
            stop_loss_fraction=0.10,
        ),
    )

    assert result.total_trades == 1
    assert result.trades[0].exit_reason == "END_OF_BACKTEST"
    assert result.final_capital == pytest.approx(1_050)


def test_maximum_drawdown_is_calculated() -> None:
    drawdown = calculate_maximum_drawdown(
        [100, 120, 90, 110]
    )

    assert drawdown == pytest.approx(25.0)


def test_invalid_signal_is_rejected() -> None:
    candles = [
        candle(0, 100, 101, 99, 100),
        candle(1, 101, 102, 100, 101),
    ]

    with pytest.raises(ValueError):
        run_backtest(candles, [BUY, 99])