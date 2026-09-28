from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import tensorflow as tf
from sqlalchemy.orm import Session

from backend.app.database import SessionLocal
from backend.app.models import BacktestRun, Stock
from backend.app.services.backtest_service import (
    BacktestConfig,
    BacktestResult,
    run_backtest,
)
from ml_pipeline.src.backtest_alignment import (
    align_target_signals,
)
from ml_pipeline.src.ensemble import combine_probabilities
from ml_pipeline.src.trading_policy import (
    TradingPolicy,
    count_signals,
    probabilities_to_signals,
)
from scripts.run_ensemble_backtest import (
    CONV1D_MODEL_PATH,
    ENSEMBLE_CONFIG_PATH,
    LOGISTIC_MODEL_PATH,
    LSTM_MODEL_PATH,
    SEQUENCE_DIRECTORY,
    convert_candles,
    decimal_value,
    json_compatible,
    load_ensemble_weights,
    load_stock_candles,
)


MODEL_DIRECTORY = Path("ml_pipeline/artifacts/models")
REPORT_DIRECTORY = Path("ml_pipeline/reports/backtests")

POLICY_PATH = MODEL_DIRECTORY / "trading_policy_v2.json"

SUMMARY_PATH = (
    REPORT_DIRECTORY / "v2_backtest_summary.json"
)

BATCH_SIZE = 256

INITIAL_CAPITAL = 100_000.0
POSITION_SIZE_FRACTION = 0.25
TRANSACTION_COST_BPS = 10.0
STOP_LOSS_FRACTION = 0.02


def load_policy() -> tuple[TradingPolicy, dict[str, Any]]:
    if not POLICY_PATH.exists():
        raise FileNotFoundError(
            f"V2 policy does not exist: {POLICY_PATH}"
        )

    configuration = json.loads(
        POLICY_PATH.read_text(encoding="utf-8")
    )

    values = configuration["policy"]

    policy = TradingPolicy(
        buy_threshold=float(
            values["buy_threshold"]
        ),
        sell_threshold=float(
            values["sell_threshold"]
        ),
        confidence_margin=float(
            values["confidence_margin"]
        ),
    )

    return policy, configuration


def save_trade_csv(
    symbol: str,
    result: BacktestResult,
) -> Path:
    path = (
        REPORT_DIRECTORY
        / f"{symbol.lower()}_v2_trades.csv"
    )

    with path.open(
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

    return path


def save_equity_csv(
    symbol: str,
    result: BacktestResult,
) -> Path:
    path = (
        REPORT_DIRECTORY
        / f"{symbol.lower()}_v2_equity.csv"
    )

    with path.open(
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

    return path


def save_database_run(
    db: Session,
    stock: Stock,
    symbol: str,
    result: BacktestResult,
    signal_counts: dict[str, int],
    weights: tuple[float, float, float],
    policy: TradingPolicy,
) -> int:
    record = BacktestRun(
        stock_id=stock.id,
        name=f"{symbol} V2 confidence-filtered backtest",
        model_version="ensemble_v2_policy",
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
            "policy_version": "v2",
            "selection_data": "validation_only",
            "buy_threshold": policy.buy_threshold,
            "sell_threshold": policy.sell_threshold,
            "confidence_margin": (
                policy.confidence_margin
            ),
            "position_size_fraction": (
                POSITION_SIZE_FRACTION
            ),
            "transaction_cost_bps": (
                TRANSACTION_COST_BPS
            ),
            "stop_loss_fraction": (
                STOP_LOSS_FRACTION
            ),
            "execution": "next_candle_open",
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

    db.add(record)
    db.flush()

    return int(record.id)


def create_result_record(
    symbol: str,
    result: BacktestResult,
    signal_counts: dict[str, int],
    database_run_id: int,
    trade_path: Path,
    equity_path: Path,
) -> dict[str, Any]:
    transaction_cost = sum(
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
        "win_rate_percent": (
            result.win_rate_percent
        ),
        "total_trades": result.total_trades,
        "total_transaction_cost": transaction_cost,
        "trade_report": str(trade_path),
        "equity_report": str(equity_path),
    }


def main() -> None:
    required_paths = [
        LOGISTIC_MODEL_PATH,
        CONV1D_MODEL_PATH,
        LSTM_MODEL_PATH,
        ENSEMBLE_CONFIG_PATH,
        POLICY_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required artifact: {path}"
            )

    REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy, policy_configuration = load_policy()
    weights = load_ensemble_weights()

    print("Loading trained models...")

    logistic_model = joblib.load(
        LOGISTIC_MODEL_PATH
    )

    conv1d_model = tf.keras.models.load_model(
        CONV1D_MODEL_PATH,
        compile=False,
    )

    lstm_model = tf.keras.models.load_model(
        LSTM_MODEL_PATH,
        compile=False,
    )

    print("\nFROZEN V2 POLICY")
    print("=" * 60)
    print(
        f"BUY threshold     : "
        f"{policy.buy_threshold:.3f}"
    )
    print(
        f"SELL threshold    : "
        f"{policy.sell_threshold:.3f}"
    )
    print(
        f"Confidence margin : "
        f"{policy.confidence_margin:.3f}"
    )

    backtest_config = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        position_size_fraction=POSITION_SIZE_FRACTION,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        stop_loss_fraction=STOP_LOSS_FRACTION,
    )

    stock_results: list[dict[str, Any]] = []

    sequence_files = sorted(
        SEQUENCE_DIRECTORY.glob("*_sequences.npz")
    )

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
                    x_test = data[
                        "x_test"
                    ].astype(np.float32)

                    target_times = data[
                        "test_times"
                    ].tolist()

                logistic_probabilities = (
                    logistic_model.predict_proba(
                        x_test[:, -1, :]
                    )
                )

                conv1d_probabilities = (
                    conv1d_model.predict(
                        x_test,
                        batch_size=BATCH_SIZE,
                        verbose=0,
                    )
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

                filtered_signals = probabilities_to_signals(
                    ensemble_probabilities,
                    policy,
                )

                signal_counts = count_signals(
                    filtered_signals
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
                    predictions=filtered_signals,
                )

                converted_candles = convert_candles(
                    database_candles
                )

                selected_candles = converted_candles[
                    alignment.start_index :
                    alignment.end_index + 1
                ]

                result = run_backtest(
                    candles=selected_candles,
                    signals=alignment.signals.tolist(),
                    config=backtest_config,
                )

                trade_path = save_trade_csv(
                    symbol,
                    result,
                )

                equity_path = save_equity_csv(
                    symbol,
                    result,
                )

                database_run_id = save_database_run(
                    db=db,
                    stock=stock,
                    symbol=symbol,
                    result=result,
                    signal_counts=signal_counts,
                    weights=weights,
                    policy=policy,
                )

                record = create_result_record(
                    symbol=symbol,
                    result=result,
                    signal_counts=signal_counts,
                    database_run_id=database_run_id,
                    trade_path=trade_path,
                    equity_path=equity_path,
                )

                stock_results.append(record)

                print(f"  Signals: {signal_counts}")
                print(
                    f"  Trades: {result.total_trades}"
                )
                print(
                    f"  Return: "
                    f"{result.total_return_percent:.4f}%"
                )
                print(
                    f"  Drawdown: "
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

    total_initial = sum(
        result["initial_capital"]
        for result in stock_results
    )

    total_final = sum(
        result["final_capital"]
        for result in stock_results
    )

    combined_return = (
        (total_final - total_initial)
        / total_initial
        * 100
    )

    summary = {
        "version": "v2_confidence_filtered",
        "policy_selection": "validation_only",
        "policy": policy.to_dictionary(),
        "validation_calibration": (
            policy_configuration["validation_result"]
        ),
        "ensemble_weights": {
            "logistic_regression": weights[0],
            "conv1d": weights[1],
            "lstm": weights[2],
        },
        "backtest_configuration": {
            "initial_capital_per_stock": (
                INITIAL_CAPITAL
            ),
            "position_size_fraction": (
                POSITION_SIZE_FRACTION
            ),
            "transaction_cost_bps": (
                TRANSACTION_COST_BPS
            ),
            "stop_loss_fraction": (
                STOP_LOSS_FRACTION
            ),
            "execution": "next_candle_open",
        },
        "combined": {
            "stock_count": len(stock_results),
            "total_initial_capital": total_initial,
            "total_final_capital": total_final,
            "net_profit_loss": (
                total_final - total_initial
            ),
            "total_return_percent": combined_return,
            "total_trades": sum(
                result["total_trades"]
                for result in stock_results
            ),
            "total_transaction_cost": sum(
                result["total_transaction_cost"]
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

    print("\nV2 COMBINED TEST RESULT")
    print("=" * 60)
    print(f"Stocks          : {len(stock_results)}")
    print(f"Initial capital : ₹{total_initial:,.2f}")
    print(f"Final capital   : ₹{total_final:,.2f}")
    print(
        f"Net profit/loss : "
        f"₹{total_final - total_initial:,.2f}"
    )
    print(
        f"Combined return : {combined_return:.4f}%"
    )
    print(
        f"Total trades    : "
        f"{summary['combined']['total_trades']}"
    )
    print(
        f"Transaction cost: "
        f"₹{summary['combined']['total_transaction_cost']:,.2f}"
    )
    print(f"Summary saved   : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()