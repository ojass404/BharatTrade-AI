import argparse
import time
from datetime import date, datetime, time as datetime_time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import select

from backend.app.database import SessionLocal
from backend.app.models.stock import Stock
from backend.app.services.angel_market_data_service import (
    AngelMarketDataError,
    AngelMarketDataService,
)


INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = datetime_time(hour=9, minute=15)
MARKET_CLOSE = datetime_time(hour=15, minute=30)


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Date must use YYYY-MM-DD format."
        ) from exc


def build_date_windows(
    start_date: date,
    end_date: date,
    window_days: int = 20,
) -> list[tuple[datetime, datetime]]:
    if start_date > end_date:
        raise ValueError(
            "The starting date cannot be after the ending date."
        )

    windows: list[tuple[datetime, datetime]] = []
    current_date = start_date

    while current_date <= end_date:
        current_end_date = min(
            current_date + timedelta(days=window_days - 1),
            end_date,
        )

        from_datetime = datetime.combine(
            current_date,
            MARKET_OPEN,
            tzinfo=INDIA_TIMEZONE,
        )

        to_datetime = datetime.combine(
            current_end_date,
            MARKET_CLOSE,
            tzinfo=INDIA_TIMEZONE,
        )

        windows.append((from_datetime, to_datetime))
        current_date = current_end_date + timedelta(days=1)

    return windows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Angel One 15-minute historical candles "
            "for active NSE stocks."
        )
    )

    parser.add_argument(
        "--from-date",
        required=True,
        type=parse_date,
        help="Starting date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--to-date",
        required=True,
        type=parse_date,
        help="Ending date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--symbols",
        default="",
        help=(
            "Optional comma-separated symbols. If omitted, "
            "all active stocks are processed."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    requested_symbols = {
        symbol.strip().upper()
        for symbol in arguments.symbols.split(",")
        if symbol.strip()
    }

    windows = build_date_windows(
        arguments.from_date,
        arguments.to_date,
    )

    with SessionLocal() as db:
        statement = (
            select(Stock)
            .where(
                Stock.exchange == "NSE",
                Stock.is_active.is_(True),
            )
            .order_by(Stock.symbol)
        )

        if requested_symbols:
            statement = statement.where(
                Stock.symbol.in_(requested_symbols)
            )

        stocks = list(db.scalars(statement).all())

        if not stocks:
            raise SystemExit(
                "No active matching NSE stocks were found."
            )

        print(
            f"Stocks selected: "
            f"{', '.join(stock.symbol for stock in stocks)}"
        )
        print(f"Date windows: {len(windows)}")
        print("Authenticating with Angel One...")

        try:
            market_data_service = AngelMarketDataService()
        except AngelMarketDataError as exc:
            raise SystemExit(str(exc)) from exc

        total_stored = 0
        failures: list[str] = []

        for stock in stocks:
            stock_total = 0

            print(f"\nProcessing {stock.symbol}")

            for from_datetime, to_datetime in windows:
                print(
                    "  "
                    f"{from_datetime.date()} to "
                    f"{to_datetime.date()}",
                    end="",
                )

                try:
                    candles = market_data_service.fetch_candles(
                        stock=stock,
                        from_datetime=from_datetime,
                        to_datetime=to_datetime,
                    )

                    stored = market_data_service.store_candles(
                        db=db,
                        candles=candles,
                    )

                    stock_total += stored
                    total_stored += stored

                    print(f" -> {stored} candles")
                except AngelMarketDataError as exc:
                    error = (
                        f"{stock.symbol} "
                        f"{from_datetime.date()} to "
                        f"{to_datetime.date()}: {exc}"
                    )

                    failures.append(error)
                    print(" -> FAILED")

                time.sleep(0.5)

            print(
                f"Completed {stock.symbol}: "
                f"{stock_total} candles processed"
            )

    print("\nHistorical import summary")
    print(f"Stocks processed : {len(stocks)}")
    print(f"Candles processed: {total_stored}")
    print(f"Failed windows   : {len(failures)}")

    if failures:
        print("\nFailures")

        for failure in failures:
            print(f"- {failure}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()