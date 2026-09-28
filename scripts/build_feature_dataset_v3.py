from pathlib import Path

import pandas as pd
from sqlalchemy import select

from backend.app.database import engine
from backend.app.models.stock import MarketCandle, Stock
from ml_pipeline.src.feature_engineering import (
    FEATURE_COLUMNS,
    create_features,
)


OUTPUT_DIRECTORY = Path(
    "ml_pipeline/data/processed/features_v3"
)

TARGET_HORIZON = 4
TARGET_THRESHOLD = 0.004


def load_stock_candles(stock_id: int) -> pd.DataFrame:
    statement = (
        select(
            MarketCandle.timestamp.label("timestamp"),
            MarketCandle.open_price.label("open"),
            MarketCandle.high_price.label("high"),
            MarketCandle.low_price.label("low"),
            MarketCandle.close_price.label("close"),
            MarketCandle.volume.label("volume"),
        )
        .where(
            MarketCandle.stock_id == stock_id,
            MarketCandle.timeframe == "15minute",
        )
        .order_by(MarketCandle.timestamp)
    )

    with engine.connect() as connection:
        return pd.read_sql(statement, connection)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    with engine.connect() as connection:
        stocks = pd.read_sql(
            select(
                Stock.id,
                Stock.symbol,
            )
            .where(Stock.is_active.is_(True))
            .order_by(Stock.symbol),
            connection,
        )

    if stocks.empty:
        raise SystemExit("No active stocks were found.")

    print("V3 COST-AWARE FEATURE GENERATION")
    print("=" * 64)
    print(f"Feature count    : {len(FEATURE_COLUMNS)}")
    print(f"Target horizon   : {TARGET_HORIZON} candles")
    print(
        f"Target threshold : "
        f"{TARGET_THRESHOLD * 100:.2f}%"
    )
    print("Same-session only: True")

    total_rows = 0

    for stock in stocks.itertuples(index=False):
        candles = load_stock_candles(stock.id)

        if candles.empty:
            print(f"{stock.symbol}: skipped; no candles")
            continue

        features = create_features(
            candles,
            target_threshold=TARGET_THRESHOLD,
            target_horizon=TARGET_HORIZON,
            same_session_target=True,
        )

        features.insert(
            0,
            "symbol",
            stock.symbol,
        )

        output_path = (
            OUTPUT_DIRECTORY
            / f"{stock.symbol.lower()}_features_v3.csv"
        )

        features.to_csv(
            output_path,
            index=False,
        )

        class_counts = (
            features["target_class"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        total_rows += len(features)

        print(
            f"{stock.symbol:<12} "
            f"rows={len(features):>6,} | "
            f"SELL={class_counts.get(0, 0):>5,} | "
            f"HOLD={class_counts.get(1, 0):>5,} | "
            f"BUY={class_counts.get(2, 0):>5,}"
        )

    print("-" * 64)
    print(f"Total feature rows: {total_rows:,}")
    print(f"Output directory  : {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()