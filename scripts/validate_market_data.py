from sqlalchemy import Date, cast, extract, func, select

from backend.app.database import SessionLocal
from backend.app.models.stock import MarketCandle, Stock


EXPECTED_CANDLES_PER_FULL_DAY = 25


def main() -> None:
    hard_failures: list[str] = []
    warnings: list[str] = []

    with SessionLocal() as db:
        total_candles = db.scalar(
            select(func.count(MarketCandle.id))
        ) or 0

        print("MARKET DATA VALIDATION")
        print("=" * 60)
        print(f"Total candles: {total_candles:,}")

        if total_candles == 0:
            raise SystemExit("Validation failed: no candles found.")

        stock_summary = db.execute(
            select(
                Stock.symbol,
                func.count(MarketCandle.id).label("candle_count"),
                func.min(MarketCandle.timestamp).label("first_candle"),
                func.max(MarketCandle.timestamp).label("last_candle"),
            )
            .join(
                MarketCandle,
                MarketCandle.stock_id == Stock.id,
            )
            .where(Stock.is_active.is_(True))
            .group_by(Stock.id, Stock.symbol)
            .order_by(Stock.symbol)
        ).all()

        print("\nACTIVE STOCK COVERAGE")
        print("-" * 60)

        for row in stock_summary:
            print(
                f"{row.symbol:<12} "
                f"{row.candle_count:>8,} candles | "
                f"{row.first_candle} -> {row.last_candle}"
            )

        active_stock_count = db.scalar(
            select(func.count(Stock.id)).where(
                Stock.is_active.is_(True)
            )
        ) or 0

        if len(stock_summary) != active_stock_count:
            hard_failures.append(
                "One or more active stocks have no candle data."
            )

        duplicate_groups = db.scalar(
            select(func.count())
            .select_from(
                select(
                    MarketCandle.stock_id,
                    MarketCandle.timestamp,
                    MarketCandle.timeframe,
                )
                .group_by(
                    MarketCandle.stock_id,
                    MarketCandle.timestamp,
                    MarketCandle.timeframe,
                )
                .having(func.count(MarketCandle.id) > 1)
                .subquery()
            )
        ) or 0

        invalid_price_rows = db.scalar(
            select(func.count(MarketCandle.id)).where(
                (
                    MarketCandle.open_price <= 0
                )
                | (
                    MarketCandle.high_price <= 0
                )
                | (
                    MarketCandle.low_price <= 0
                )
                | (
                    MarketCandle.close_price <= 0
                )
            )
        ) or 0

        invalid_ohlc_rows = db.scalar(
            select(func.count(MarketCandle.id)).where(
                (
                    MarketCandle.high_price
                    < MarketCandle.open_price
                )
                | (
                    MarketCandle.high_price
                    < MarketCandle.close_price
                )
                | (
                    MarketCandle.high_price
                    < MarketCandle.low_price
                )
                | (
                    MarketCandle.low_price
                    > MarketCandle.open_price
                )
                | (
                    MarketCandle.low_price
                    > MarketCandle.close_price
                )
            )
        ) or 0

        negative_volume_rows = db.scalar(
            select(func.count(MarketCandle.id)).where(
                MarketCandle.volume < 0
            )
        ) or 0

        wrong_timeframe_rows = db.scalar(
            select(func.count(MarketCandle.id)).where(
                MarketCandle.timeframe != "15minute"
            )
        ) or 0

        misaligned_rows = db.scalar(
            select(func.count(MarketCandle.id)).where(
                (
                    extract(
                        "minute",
                        MarketCandle.timestamp,
                    )
                    % 15
                )
                != 0
            )
        ) or 0

        non_angel_active_stocks = db.scalar(
            select(func.count(Stock.id)).where(
                Stock.is_active.is_(True),
                ~Stock.instrument_key.startswith("ANGEL_NSE|"),
            )
        ) or 0

        daily_counts = db.execute(
            select(
                Stock.symbol,
                cast(
                    MarketCandle.timestamp,
                    Date,
                ).label("trading_date"),
                func.count(MarketCandle.id).label("candle_count"),
            )
            .join(
                Stock,
                Stock.id == MarketCandle.stock_id,
            )
            .where(Stock.is_active.is_(True))
            .group_by(
                Stock.symbol,
                cast(MarketCandle.timestamp, Date),
            )
        ).all()

        incomplete_days = [
            row
            for row in daily_counts
            if row.candle_count != EXPECTED_CANDLES_PER_FULL_DAY
        ]

        print("\nINTEGRITY CHECKS")
        print("-" * 60)
        print(f"Duplicate candle groups : {duplicate_groups}")
        print(f"Invalid price rows      : {invalid_price_rows}")
        print(f"Invalid OHLC rows       : {invalid_ohlc_rows}")
        print(f"Negative volume rows    : {negative_volume_rows}")
        print(f"Wrong timeframe rows    : {wrong_timeframe_rows}")
        print(f"Misaligned timestamps   : {misaligned_rows}")
        print(f"Non-Angel active stocks : {non_angel_active_stocks}")
        print(f"Incomplete trading days : {len(incomplete_days)}")

        if duplicate_groups:
            hard_failures.append(
                f"{duplicate_groups} duplicate candle groups found."
            )

        if invalid_price_rows:
            hard_failures.append(
                f"{invalid_price_rows} candles contain invalid prices."
            )

        if invalid_ohlc_rows:
            hard_failures.append(
                f"{invalid_ohlc_rows} candles violate OHLC rules."
            )

        if negative_volume_rows:
            hard_failures.append(
                f"{negative_volume_rows} candles have negative volume."
            )

        if wrong_timeframe_rows:
            hard_failures.append(
                f"{wrong_timeframe_rows} candles use another timeframe."
            )

        if misaligned_rows:
            hard_failures.append(
                f"{misaligned_rows} timestamps are not 15-minute aligned."
            )

        if non_angel_active_stocks:
            hard_failures.append(
                f"{non_angel_active_stocks} active stocks lack Angel tokens."
            )

        if incomplete_days:
            warnings.append(
                "Some trading days do not contain exactly 25 candles. "
                "These may be exchange holidays, special sessions, "
                "partial sessions or unavailable broker data."
            )

    print("\nVALIDATION RESULT")
    print("=" * 60)

    if warnings:
        for warning in warnings:
            print(f"WARNING: {warning}")

    if hard_failures:
        for failure in hard_failures:
            print(f"FAILED: {failure}")

        raise SystemExit(1)

    print("PASSED: No critical market-data integrity problems found.")


if __name__ == "__main__":
    main()