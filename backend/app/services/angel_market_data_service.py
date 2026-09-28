from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.app.models.stock import MarketCandle, Stock
from backend.app.services.angel_auth_service import (
    AngelAuthService,
    AngelAuthenticationError,
)


INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")
FIFTEEN_MINUTE_INTERVAL = "FIFTEEN_MINUTE"
DATABASE_TIMEFRAME = "15minute"


class AngelMarketDataError(RuntimeError):
    """Raised when Angel One market-data retrieval fails."""


class AngelMarketDataService:
    def __init__(
        self,
        authentication_service: AngelAuthService | None = None,
    ) -> None:
        self.authentication_service = (
            authentication_service or AngelAuthService()
        )

        try:
            self.authentication_service.authenticate()
        except AngelAuthenticationError as exc:
            raise AngelMarketDataError(
                f"Angel One authentication failed: {exc}"
            ) from exc

        self.client = self.authentication_service.client

        if self.client is None:
            raise AngelMarketDataError(
                "Angel One authenticated client was not created."
            )

    @staticmethod
    def extract_symbol_token(stock: Stock) -> str:
        prefix = "ANGEL_NSE|"

        if not stock.instrument_key.startswith(prefix):
            raise AngelMarketDataError(
                f"{stock.symbol} does not have an Angel One token."
            )

        token = stock.instrument_key.removeprefix(prefix).strip()

        if not token:
            raise AngelMarketDataError(
                f"{stock.symbol} has an empty Angel One token."
            )

        return token

    @staticmethod
    def parse_timestamp(value: str) -> datetime:
        normalized_value = value.replace("Z", "+00:00")
        timestamp = datetime.fromisoformat(normalized_value)

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=INDIA_TIMEZONE)

        return timestamp

    def fetch_candles(
        self,
        stock: Stock,
        from_datetime: datetime,
        to_datetime: datetime,
    ) -> list[dict[str, Any]]:
        if from_datetime > to_datetime:
            raise AngelMarketDataError(
                "The starting datetime cannot be after the ending datetime."
            )

        symbol_token = self.extract_symbol_token(stock)

        parameters = {
            "exchange": "NSE",
            "symboltoken": symbol_token,
            "interval": FIFTEEN_MINUTE_INTERVAL,
            "fromdate": from_datetime.strftime("%Y-%m-%d %H:%M"),
            "todate": to_datetime.strftime("%Y-%m-%d %H:%M"),
        }

        try:
            response = self.client.getCandleData(parameters)
        except Exception as exc:
            raise AngelMarketDataError(
                f"Could not download candles for {stock.symbol}."
            ) from exc

        if not response.get("status"):
            message = response.get(
                "message",
                "Angel One rejected the candle-data request.",
            )

            error_code = response.get("errorcode")

            if error_code:
                message = f"{message} ({error_code})"

            raise AngelMarketDataError(
                f"{stock.symbol}: {message}"
            )

        rows = response.get("data") or []
        candles: list[dict[str, Any]] = []

        for row in rows:
            if not isinstance(row, list) or len(row) < 6:
                continue

            candles.append(
                {
                    "stock_id": stock.id,
                    "timestamp": self.parse_timestamp(str(row[0])),
                    "timeframe": DATABASE_TIMEFRAME,
                    "open_price": Decimal(str(row[1])),
                    "high_price": Decimal(str(row[2])),
                    "low_price": Decimal(str(row[3])),
                    "close_price": Decimal(str(row[4])),
                    "volume": int(row[5]),
                    "open_interest": (
                        int(row[6])
                        if len(row) > 6 and row[6] is not None
                        else None
                    ),
                }
            )

        return candles

    @staticmethod
    def store_candles(
        db: Session,
        candles: list[dict[str, Any]],
    ) -> int:
        if not candles:
            return 0

        statement = insert(MarketCandle).values(candles)

        statement = statement.on_conflict_do_update(
            index_elements=[
                "stock_id",
                "timestamp",
                "timeframe",
            ],
            set_={
                "open_price": statement.excluded.open_price,
                "high_price": statement.excluded.high_price,
                "low_price": statement.excluded.low_price,
                "close_price": statement.excluded.close_price,
                "volume": statement.excluded.volume,
                "open_interest": statement.excluded.open_interest,
            },
        )

        db.execute(statement)
        db.commit()

        return len(candles)