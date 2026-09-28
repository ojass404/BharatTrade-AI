from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import CheckConstraint, DateTime, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


class Portfolio(Base):
    """Virtual portfolio used for educational paper trading."""

    __tablename__ = "portfolios"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
    )

    initial_capital: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    available_cash: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    portfolio_value: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    realized_profit_loss: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=0,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="ACTIVE",
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

    paper_trades: Mapped[list["PaperTrade"]] = relationship(
        "PaperTrade",
        back_populates="portfolio",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint(
            "initial_capital > 0",
            name="ck_portfolios_initial_capital_positive",
        ),
        CheckConstraint(
            "available_cash >= 0",
            name="ck_portfolios_cash_non_negative",
        ),
        CheckConstraint(
            "portfolio_value >= 0",
            name="ck_portfolios_value_non_negative",
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'PAUSED', 'CLOSED')",
            name="ck_portfolios_valid_status",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"Portfolio(id={self.id!r}, name={self.name!r}, "
            f"status={self.status!r})"
        )