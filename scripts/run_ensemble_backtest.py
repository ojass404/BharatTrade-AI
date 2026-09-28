from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import tensorflow as tf
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import SessionLocal
from backend.app.models import BacktestRun, MarketCandle, Stock
from backend.app.services.backtest_service import (
    BacktestCandle,
    BacktestConfig,
    BacktestResult,
    run_backtest,
)
from ml_pipeline.src.backtest_alignment import (
    align_target_signals,
)
from ml_pipeline.src.ensemble import combine_probabilities


SEQUENCE_DIRECTORY = Path(
    "ml_pipeline/data/processed/sequences"
)

MODEL_DIRECTORY = Path(
    "ml_pipeline/artifacts/models"
)

REPORT_DIRECTORY = Path(
    "ml_pipeline/reports/backtests"
)

LOGISTIC_MODEL_PATH = (
    MODEL_DIRECTORY / "baseline_logistic_regression.joblib"
)

CONV1D_MODEL_PATH = (
    MODEL_DIRECTORY / "conv1d_model.keras"
)

LSTM_MODEL_PATH = (
    MODEL_DIRECTORY / "lstm_model.keras"
)

ENSEMBLE_CONFIG_PATH = (
    MODEL_DIRECTORY / "ensemble_config.json"
)

SUMMARY_PATH = (
    REPORT_DIRECTORY / "ensemble_backtest_summary.json"
)

INITIAL_CAPITAL = 100_000.0
POSITION_SIZE_FRACTION = 0.25
TRANSACTION_COST_BPS = 10.0
STOP_LOSS_FRACTION = 0.02
BATCH_SIZE = 256


def decimal_value(value: float, places: int = 4) -> Decimal:
    return Decimal(str(round(float(value), places)))


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_compatible(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    return value


def load_ensemble_weights() -> tuple[float, float, float]:
    if not ENSEMBLE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Missing ensemble configuration: "
            f"{ENSEMBLE_CONFIG_PATH}"
        )

    configuration = json.loads(
        ENSEMBLE_CONFIG_PATH.read_text(encoding="utf-8")
    )

    stored_weights = configuration["weights"]

    weights = (
        float(stored_weights["logistic_regression"]),
        float(stored_weights["conv1d"]),
        float(stored_weights["lstm"]),
    )

    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Stored ensemble weights do not sum to 1.")

    return weights


def load_stock_candles(
    db: Session,
    symbol: str,
) -> tuple[Stock, list[MarketCandle]]:
    stock = db.scalar(
        select(Stock).where(
            Stock.symbol == symbol,
            Stock.exchange == "NSE",
        )
    )

    if stock is None:
        raise ValueError(
            f"Stock {symbol} does not exist in the database."
        )

    candles = list(
        db.scalars(
            select(MarketCandle)
            .where(
                MarketCandle.stock_id == stock.id,
                MarketCandle.timeframe == "15minute",
            )
            .order_by(MarketCandle.timestamp)
        ).all()
    )

    if len(candles) < 2:
        raise ValueError(
            f"Insufficient candle data for {symbol}."
        )

    return stock, candles


def convert_candles(
    database_candles: list[MarketCandle],
) -> list[BacktestCandle]:
    return [
        BacktestCandle(
            timestamp=candle.timestamp,
            open_price=float(candle.open_price),
            high_price=float(candle.high_price),
            low_price=float(candle.low_price),
            close_price=float(candle.close_price),
        )
        for candle in database_candles
    ]


def create_signal_counts(
    predictions: np.ndarray,
) -> dict[str, int]:
    return {
        "SELL": int(np.sum(predictions == 0)),
        "HOLD": int(np.sum(predictions == 1)),
        "BUY": int(np.sum(predictions == 2)),
    }


def save_trade_report(
    symbol: str,
    result: BacktestResult,
) -> Path:
    output_path = (
        REPORT_DIRECTORY
        / f"{symbol.lower()}_ensemble_trades.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.writer(output_file)

        writer.writerow(
            [
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "quantity",
                "entry_cost",
                "exit_cost",
                "net_profit_loss",
                "exit_reason",
            ]
        )

        for trade in result.trades:
            writer.writerow(
                [
                    trade.entry_time.isoformat(),
                    trade.exit_time.isoformat(),
                    round(trade.entry_price, 4),
                    round(trade.exit_price, 4),
                    trade.quantity,
                    round(trade.entry_cost, 4),
                    round(trade.exit_cost, 4),
                    round(trade.net_profit_loss, 4),
                    trade.exit_reason,
                ]
            )

    return output_path


def save_equity_report(
    symbol: str,
    result: BacktestResult,
) -> Path:
    output_path = (
        REPORT_DIRECTORY
        / f"{symbol.lower()}_ensemble_equity.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.writer(output_file)

        writer.writerow(["timestamp", "equity"])

        for timestamp, equity in zip(
            result.equity_timestamps,
            result.equity_curve,
        ):
            writer.writerow(
                [
                    timestamp.isoformat(),
                    round(equity, 4),
                ]
            )

    return output_path


def save_database_result(
    db: Session,
    stock: Stock,
    symbol: str,
    result: BacktestResult,
    weights: tuple[float, float, float],
    signal_counts: dict[str, int],
) -> int:
    database_run = BacktestRun(
        stock_id=stock.id,
        name=f"{symbol} ensemble test-period backtest",
        model_version="ensemble_v1",
        timeframe="15minute",
        start_date=result.equity_timestamps[0],
        end_date=result.equity_timestamps[-1],
        initial_capital=decimal_value(
            result.initial_capital,
            places=2,
        ),
        final_capital=decimal_value(
            result.final_capital,
            places=2,
        ),
        total_return=decimal_value(
            result.total_return_percent
        ),
        sharpe_ratio=decimal_value(
            result.sharpe_ratio
        ),
        max_drawdown=decimal_value(
            result.maximum_drawdown_percent
        ),
        win_rate=decimal_value(
            result.win_rate_percent
        ),
        total_trades=result.total_trades,
        parameters={
            "position_size_fraction": POSITION_SIZE_FRACTION,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "stop_loss_fraction": STOP_LOSS_FRACTION,
            "execution": "next_candle_open",
            "long_only": True,
            "signal_counts": signal_counts,
            "ensemble_weights": {
                "logistic_regression": weights[0],
                "conv1d": weights[1],
                "lstm": weights[2],
            },
        },
        status="COMPLETED",
        completed_at=datetime.now(timezone.utc),
    )

    db.add(database_run)
    db.flush()

    return int(database_run.id)


def result_to_dictionary(
    symbol: str,
    result: BacktestResult,
    signal_counts: dict[str, int],
    database_run_id: int,
    trade_report_path: Path,
    equity_report_path: Path,
) -> dict[str, Any]:
    profits = [
        trade.net_profit_loss
        for trade in result.trades
    ]

    gross_profit = sum(
        profit for profit in profits if profit > 0
    )

    gross_loss = abs(
        sum(profit for profit in profits if profit < 0)
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else 0.0
    )

    total_transaction_cost = sum(
        trade.entry_cost + trade.exit_cost
        for trade in result.trades
    )

    return {
        "symbol": symbol,
        "database_run_id": database_run_id,
        "period": {
            "start": result.equity_timestamps[0],
            "end": result.equity_timestamps[-1],
        },
        "signal_counts": signal_counts,
        "initial_capital": result.initial_capital,
        "final_capital": result.final_capital,
        "net_profit_loss": (
            result.final_capital
            - result.initial_capital
        ),
        "total_return_percent": (
            result.total_return_percent
        ),
        "sharpe_ratio": result.sharpe_ratio,
        "maximum_drawdown_percent": (
            result.maximum_drawdown_percent
        ),
        "win_rate_percent": result.win_rate_percent,
        "total_trades": result.total_trades,
        "profit_factor": profit_factor,
        "total_transaction_cost": total_transaction_cost,
        "trade_report": str(trade_report_path),
        "equity_report": str(equity_report_path),
    }


def main() -> None:
    required_files = [
        LOGISTIC_MODEL_PATH,
        CONV1D_MODEL_PATH,
        LSTM_MODEL_PATH,
        ENSEMBLE_CONFIG_PATH,
    ]

    for required_path in required_files:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required artifact is missing: {required_path}"
            )

    sequence_files = sorted(
        SEQUENCE_DIRECTORY.glob("*_sequences.npz")
    )

    if not sequence_files:
        raise FileNotFoundError(
            f"No sequence files found in {SEQUENCE_DIRECTORY}."
        )

    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("Loading trained models...")

    logistic_model = joblib.load(LOGISTIC_MODEL_PATH)

    conv1d_model = tf.keras.models.load_model(
        CONV1D_MODEL_PATH,
        compile=False,
    )

    lstm_model = tf.keras.models.load_model(
        LSTM_MODEL_PATH,
        compile=False,
    )

    weights = load_ensemble_weights()

    print(
        "Ensemble weights: "
        f"Logistic={weights[0]:.2f}, "
        f"Conv1D={weights[1]:.2f}, "
        f"LSTM={weights[2]:.2f}"
    )

    settings = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        position_size_fraction=POSITION_SIZE_FRACTION,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        stop_loss_fraction=STOP_LOSS_FRACTION,
    )

    stock_results: list[dict[str, Any]] = []

    with SessionLocal() as db:
        try:
            for sequence_path in sequence_files:
                symbol = sequence_path.stem.replace(
                    "_sequences",
                    "",
                ).upper()

                print(f"\nProcessing {symbol}...")

                with np.load(
                    sequence_path,
                    allow_pickle=True,
                ) as data:
                    x_test = data["x_test"].astype(
                        np.float32
                    )
                    target_times = data["test_times"].tolist()

                logistic_probabilities = (
                    logistic_model.predict_proba(
                        x_test[:, -1, :]
                    )
                )

                conv1d_probabilities = conv1d_model.predict(
                    x_test,
                    batch_size=BATCH_SIZE,
                    verbose=0,
                )

                lstm_probabilities = lstm_model.predict(
                    x_test,
                    batch_size=BATCH_SIZE,
                    verbose=0,
                )

                ensemble_probabilities = combine_probabilities(
                    [
                        logistic_probabilities,
                        conv1d_probabilities,
                        lstm_probabilities,
                    ],
                    weights,
                )

                predictions = np.argmax(
                    ensemble_probabilities,
                    axis=1,
                ).astype(np.int64)

                signal_counts = create_signal_counts(
                    predictions
                )

                stock, database_candles = load_stock_candles(
                    db,
                    symbol,
                )

                alignment = align_target_signals(
                    candle_timestamps=[
                        candle.timestamp
                        for candle in database_candles
                    ],
                    target_timestamps=target_times,
                    predictions=predictions,
                )

                selected_database_candles = database_candles[
                    alignment.start_index :
                    alignment.end_index + 1
                ]

                backtest_candles = convert_candles(
                    selected_database_candles
                )

                result = run_backtest(
                    candles=backtest_candles,
                    signals=alignment.signals.tolist(),
                    config=settings,
                )

                trade_report_path = save_trade_report(
                    symbol,
                    result,
                )

                equity_report_path = save_equity_report(
                    symbol,
                    result,
                )

                database_run_id = save_database_result(
                    db=db,
                    stock=stock,
                    symbol=symbol,
                    result=result,
                    weights=weights,
                    signal_counts=signal_counts,
                )

                stock_result = result_to_dictionary(
                    symbol=symbol,
                    result=result,
                    signal_counts=signal_counts,
                    database_run_id=database_run_id,
                    trade_report_path=trade_report_path,
                    equity_report_path=equity_report_path,
                )

                stock_results.append(stock_result)

                print(
                    f"  Signals: {signal_counts}"
                )
                print(
                    f"  Trades: {result.total_trades}"
                )
                print(
                    f"  Return: "
                    f"{result.total_return_percent:.4f}%"
                )
                print(
                    f"  Sharpe: {result.sharpe_ratio:.4f}"
                )
                print(
                    f"  Max drawdown: "
                    f"{result.maximum_drawdown_percent:.4f}%"
                )
                print(
                    f"  Win rate: "
                    f"{result.win_rate_percent:.2f}%"
                )

            db.commit()

        except Exception:
            db.rollback()
            raise

    total_initial_capital = sum(
        result["initial_capital"]
        for result in stock_results
    )

    total_final_capital = sum(
        result["final_capital"]
        for result in stock_results
    )

    combined_return_percent = (
        (total_final_capital - total_initial_capital)
        / total_initial_capital
        * 100
    )

    summary = {
        "model": "validation_optimized_ensemble_v1",
        "evaluation_scope": (
            "independent long-only backtest per stock"
        ),
        "execution_rule": "next_candle_open",
        "data_split": "unseen chronological test period",
        "ensemble_weights": {
            "logistic_regression": weights[0],
            "conv1d": weights[1],
            "lstm": weights[2],
        },
        "backtest_configuration": {
            "initial_capital_per_stock": INITIAL_CAPITAL,
            "position_size_fraction": POSITION_SIZE_FRACTION,
            "transaction_cost_bps": TRANSACTION_COST_BPS,
            "stop_loss_fraction": STOP_LOSS_FRACTION,
        },
        "combined": {
            "stock_count": len(stock_results),
            "total_initial_capital": total_initial_capital,
            "total_final_capital": total_final_capital,
            "net_profit_loss": (
                total_final_capital
                - total_initial_capital
            ),
            "total_return_percent": combined_return_percent,
            "total_trades": sum(
                result["total_trades"]
                for result in stock_results
            ),
        },
        "stocks": stock_results,
        "generated_at": datetime.now(timezone.utc),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            json_compatible(summary),
            indent=4,
        ),
        encoding="utf-8",
    )

    print("\nCOMBINED BACKTEST")
    print("=" * 60)
    print(
        f"Stocks               : {len(stock_results)}"
    )
    print(
        f"Initial capital      : "
        f"₹{total_initial_capital:,.2f}"
    )
    print(
        f"Final capital        : "
        f"₹{total_final_capital:,.2f}"
    )
    print(
        f"Net profit/loss      : "
        f"₹{total_final_capital - total_initial_capital:,.2f}"
    )
    print(
        f"Combined return      : "
        f"{combined_return_percent:.4f}%"
    )
    print(
        f"Total completed trades: "
        f"{summary['combined']['total_trades']}"
    )
    print(f"Summary saved        : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()