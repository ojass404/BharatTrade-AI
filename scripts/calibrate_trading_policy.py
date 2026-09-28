from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import tensorflow as tf

from backend.app.database import SessionLocal
from backend.app.services.backtest_service import (
    BacktestCandle,
    BacktestConfig,
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
    load_ensemble_weights,
    load_stock_candles,
)


MODEL_DIRECTORY = Path("ml_pipeline/artifacts/models")
REPORT_DIRECTORY = Path("ml_pipeline/reports/backtests")

POLICY_PATH = MODEL_DIRECTORY / "trading_policy_v2.json"

CALIBRATION_RESULTS_PATH = (
    REPORT_DIRECTORY / "policy_calibration_results.csv"
)

CALIBRATION_SUMMARY_PATH = (
    REPORT_DIRECTORY / "policy_calibration_summary.json"
)

BUY_THRESHOLDS = [
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

SELL_THRESHOLDS = [
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

CONFIDENCE_MARGINS = [
    0.00,
    0.025,
    0.05,
    0.075,
    0.10,
]

MINIMUM_TOTAL_TRADES = 25
DRAWDOWN_PENALTY = 0.25
BATCH_SIZE = 256


@dataclass
class StockValidationData:
    symbol: str
    probabilities: np.ndarray
    target_times: list[object]
    database_candles: list
    backtest_candles: list[BacktestCandle]


def prepare_validation_data(
    logistic_model,
    conv1d_model: tf.keras.Model,
    lstm_model: tf.keras.Model,
    weights: tuple[float, float, float],
) -> list[StockValidationData]:
    prepared: list[StockValidationData] = []

    sequence_files = sorted(
        SEQUENCE_DIRECTORY.glob("*_sequences.npz")
    )

    if not sequence_files:
        raise FileNotFoundError(
            "No sequence datasets were found."
        )

    with SessionLocal() as db:
        for sequence_path in sequence_files:
            symbol = sequence_path.stem.replace(
                "_sequences",
                "",
            ).upper()

            print(f"Preparing {symbol} validation data...")

            with np.load(
                sequence_path,
                allow_pickle=True,
            ) as data:
                x_validation = data[
                    "x_validation"
                ].astype(np.float32)

                target_times = data[
                    "validation_times"
                ].tolist()

            logistic_probabilities = (
                logistic_model.predict_proba(
                    x_validation[:, -1, :]
                )
            )

            conv1d_probabilities = conv1d_model.predict(
                x_validation,
                batch_size=BATCH_SIZE,
                verbose=0,
            )

            lstm_probabilities = lstm_model.predict(
                x_validation,
                batch_size=BATCH_SIZE,
                verbose=0,
            )

            probabilities = combine_probabilities(
                [
                    logistic_probabilities,
                    conv1d_probabilities,
                    lstm_probabilities,
                ],
                weights,
            )

            _, database_candles = load_stock_candles(
                db,
                symbol,
            )

            prepared.append(
                StockValidationData(
                    symbol=symbol,
                    probabilities=probabilities,
                    target_times=target_times,
                    database_candles=database_candles,
                    backtest_candles=convert_candles(
                        database_candles
                    ),
                )
            )

    return prepared


def evaluate_policy(
    policy: TradingPolicy,
    stocks: list[StockValidationData],
    config: BacktestConfig,
) -> dict[str, object]:
    stock_results: list[dict[str, object]] = []

    total_initial_capital = 0.0
    total_final_capital = 0.0
    total_trades = 0
    total_drawdown = 0.0

    for stock in stocks:
        signals = probabilities_to_signals(
            stock.probabilities,
            policy,
        )

        alignment = align_target_signals(
            candle_timestamps=[
                candle.timestamp
                for candle in stock.database_candles
            ],
            target_timestamps=stock.target_times,
            predictions=signals,
        )

        selected_candles = stock.backtest_candles[
            alignment.start_index :
            alignment.end_index + 1
        ]

        result = run_backtest(
            candles=selected_candles,
            signals=alignment.signals.tolist(),
            config=config,
        )

        signal_counts = count_signals(signals)

        total_initial_capital += result.initial_capital
        total_final_capital += result.final_capital
        total_trades += result.total_trades
        total_drawdown += (
            result.maximum_drawdown_percent
        )

        stock_results.append(
            {
                "symbol": stock.symbol,
                "return_percent": (
                    result.total_return_percent
                ),
                "maximum_drawdown_percent": (
                    result.maximum_drawdown_percent
                ),
                "sharpe_ratio": result.sharpe_ratio,
                "win_rate_percent": (
                    result.win_rate_percent
                ),
                "total_trades": result.total_trades,
                "signal_counts": signal_counts,
            }
        )

    combined_return_percent = (
        (total_final_capital - total_initial_capital)
        / total_initial_capital
        * 100
    )

    average_drawdown_percent = (
        total_drawdown / len(stocks)
    )

    objective_score = (
        combined_return_percent
        - DRAWDOWN_PENALTY
        * average_drawdown_percent
    )

    return {
        "buy_threshold": policy.buy_threshold,
        "sell_threshold": policy.sell_threshold,
        "confidence_margin": policy.confidence_margin,
        "combined_return_percent": (
            combined_return_percent
        ),
        "average_drawdown_percent": (
            average_drawdown_percent
        ),
        "objective_score": objective_score,
        "total_trades": total_trades,
        "stocks": stock_results,
    }


def save_results(
    results: list[dict[str, object]],
) -> None:
    ordered = sorted(
        results,
        key=lambda item: float(item["objective_score"]),
        reverse=True,
    )

    with CALIBRATION_RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.writer(output_file)

        writer.writerow(
            [
                "rank",
                "buy_threshold",
                "sell_threshold",
                "confidence_margin",
                "combined_return_percent",
                "average_drawdown_percent",
                "objective_score",
                "total_trades",
            ]
        )

        for rank, result in enumerate(
            ordered,
            start=1,
        ):
            writer.writerow(
                [
                    rank,
                    result["buy_threshold"],
                    result["sell_threshold"],
                    result["confidence_margin"],
                    result["combined_return_percent"],
                    result["average_drawdown_percent"],
                    result["objective_score"],
                    result["total_trades"],
                ]
            )


def main() -> None:
    required_paths = [
        LOGISTIC_MODEL_PATH,
        CONV1D_MODEL_PATH,
        LSTM_MODEL_PATH,
        ENSEMBLE_CONFIG_PATH,
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

    weights = load_ensemble_weights()

    stocks = prepare_validation_data(
        logistic_model=logistic_model,
        conv1d_model=conv1d_model,
        lstm_model=lstm_model,
        weights=weights,
    )

    config = BacktestConfig(
        initial_capital=100_000.0,
        position_size_fraction=0.25,
        transaction_cost_bps=10.0,
        stop_loss_fraction=0.02,
    )

    results: list[dict[str, object]] = []
    total_combinations = (
        len(BUY_THRESHOLDS)
        * len(SELL_THRESHOLDS)
        * len(CONFIDENCE_MARGINS)
    )

    print(
        f"\nEvaluating {total_combinations} "
        "policy combinations..."
    )

    completed = 0

    for buy_threshold in BUY_THRESHOLDS:
        for sell_threshold in SELL_THRESHOLDS:
            for confidence_margin in CONFIDENCE_MARGINS:
                policy = TradingPolicy(
                    buy_threshold=buy_threshold,
                    sell_threshold=sell_threshold,
                    confidence_margin=confidence_margin,
                )

                evaluation = evaluate_policy(
                    policy=policy,
                    stocks=stocks,
                    config=config,
                )

                results.append(evaluation)
                completed += 1

                if completed % 20 == 0:
                    print(
                        f"  Completed "
                        f"{completed}/{total_combinations}"
                    )

    eligible_results = [
        result
        for result in results
        if int(result["total_trades"])
        >= MINIMUM_TOTAL_TRADES
    ]

    if not eligible_results:
        raise RuntimeError(
            "No policy generated the minimum number of trades."
        )

    best_result = max(
        eligible_results,
        key=lambda item: (
            float(item["objective_score"]),
            float(item["combined_return_percent"]),
        ),
    )

    best_policy = TradingPolicy(
        buy_threshold=float(
            best_result["buy_threshold"]
        ),
        sell_threshold=float(
            best_result["sell_threshold"]
        ),
        confidence_margin=float(
            best_result["confidence_margin"]
        ),
    )

    policy_configuration = {
        "version": "v2",
        "selection_data": "validation_only",
        "policy": best_policy.to_dictionary(),
        "objective": (
            "combined_return_percent - "
            "0.25 * average_drawdown_percent"
        ),
        "minimum_total_trades": MINIMUM_TOTAL_TRADES,
        "validation_result": best_result,
        "ensemble_weights": {
            "logistic_regression": weights[0],
            "conv1d": weights[1],
            "lstm": weights[2],
        },
        "backtest_configuration": {
            "initial_capital_per_stock": 100_000.0,
            "position_size_fraction": 0.25,
            "transaction_cost_bps": 10.0,
            "stop_loss_fraction": 0.02,
        },
    }

    POLICY_PATH.write_text(
        json.dumps(
            policy_configuration,
            indent=4,
        ),
        encoding="utf-8",
    )

    CALIBRATION_SUMMARY_PATH.write_text(
        json.dumps(
            policy_configuration,
            indent=4,
        ),
        encoding="utf-8",
    )

    save_results(results)

    ordered_results = sorted(
        eligible_results,
        key=lambda item: float(
            item["objective_score"]
        ),
        reverse=True,
    )

    print("\nTOP FIVE VALIDATION POLICIES")
    print("=" * 72)

    for rank, result in enumerate(
        ordered_results[:5],
        start=1,
    ):
        print(
            f"{rank}. "
            f"BUY>={result['buy_threshold']:.3f} | "
            f"SELL>={result['sell_threshold']:.3f} | "
            f"margin={result['confidence_margin']:.3f} | "
            f"return="
            f"{result['combined_return_percent']:.4f}% | "
            f"drawdown="
            f"{result['average_drawdown_percent']:.4f}% | "
            f"trades={result['total_trades']} | "
            f"score={result['objective_score']:.4f}"
        )

    print("\nSELECTED V2 POLICY")
    print("=" * 72)
    print(
        f"BUY threshold     : "
        f"{best_policy.buy_threshold}"
    )
    print(
        f"SELL threshold    : "
        f"{best_policy.sell_threshold}"
    )
    print(
        f"Confidence margin : "
        f"{best_policy.confidence_margin}"
    )
    print(
        f"Validation return : "
        f"{best_result['combined_return_percent']:.4f}%"
    )
    print(
        f"Validation trades : "
        f"{best_result['total_trades']}"
    )
    print(f"Policy saved      : {POLICY_PATH}")


if __name__ == "__main__":
    main()