from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base

if TYPE_CHECKING:
    from backend.app.models.backtest import BacktestRun
    from backend.app.models.prediction import ModelPrediction
    from backend.app.models.trade import PaperTrade


class Stock(Base):
    __tablename__ = "stocks"

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "exchange",
            name="uq_stocks_symbol_exchange",
        ),
        UniqueConstraint(
            "instrument_key",
            name="uq_stocks_instrument_key",
        ),
        CheckConstraint(
            "char_length(symbol) > 0",
            name="ck_stocks_symbol_not_empty",
        ),
        CheckConstraint(
            "exchange IN ('NSE', 'BSE')",
            name="ck_stocks_valid_exchange",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    symbol: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
    )

    company_name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    exchange: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        default="NSE",
        server_default="NSE",
    )

    instrument_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    sector: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    candles: Mapped[list[MarketCandle]] = relationship(
        back_populates="stock",
        cascade="all, delete-orphan",
    )

    predictions: Mapped[list[ModelPrediction]] = relationship(
        back_populates="stock",
        cascade="all, delete-orphan",
    )

    paper_trades: Mapped[list[PaperTrade]] = relationship(
        back_populates="stock",
    )

    backtest_runs: Mapped[list[BacktestRun]] = relationship(
        back_populates="stock",
    )

    def __repr__(self) -> str:
        return (
            f"Stock(id={self.id!r}, symbol={self.symbol!r}, "
            f"exchange={self.exchange!r})"
        )


class MarketCandle(Base):
    __tablename__ = "market_candles"

    __table_args__ = (
        UniqueConstraint(
            "stock_id",
            "timestamp",
            "timeframe",
            name="uq_market_candle_stock_time_timeframe",
        ),
        CheckConstraint(
            "open_price > 0",
            name="ck_market_candles_open_positive",
        ),
        CheckConstraint(
            "high_price > 0",
            name="ck_market_candles_high_positive",
        ),
        CheckConstraint(
            "low_price > 0",
            name="ck_market_candles_low_positive",
        ),
        CheckConstraint(
            "close_price > 0",
            name="ck_market_candles_close_positive",
        ),
        CheckConstraint(
            "high_price >= low_price",
            name="ck_market_candles_high_gte_low",
        ),
        CheckConstraint(
            "volume >= 0",
            name="ck_market_candles_volume_non_negative",
        ),
        Index(
            "ix_market_candles_stock_timestamp",
            "stock_id",
            "timestamp",
        ),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    stock_id: Mapped[int] = mapped_column(
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    timeframe: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="15minute",
        server_default="15minute",
    )

    open_price: Mapped[Decimal] = mapped_column(
        Numeric(18, 4),
        nullable=False,
    )

    high_price: Mapped[Decimal] = mapped_column(
        Numeric(18, 4),
        nullable=False,
    )

    low_price: Mapped[Decimal] = mapped_column(
        Numeric(18, 4),
        nullable=False,
    )

    close_price: Mapped[Decimal] = mapped_column(
        Numeric(18, 4),
        nullable=False,
    )

    volume: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
        default=0,
        server_default="0",
    )

    open_interest: Mapped[int | None] = mapped_column(
        BigInteger,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    stock: Mapped[Stock] = relationship(
        back_populates="candles",
    )

    def __repr__(self) -> str:
        return (
            f"MarketCandle(id={self.id!r}, stock_id={self.stock_id!r}, "
            f"timestamp={self.timestamp!r})"
        )