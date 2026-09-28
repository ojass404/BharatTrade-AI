from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base

if TYPE_CHECKING:
    from backend.app.models.stock import Stock


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    __table_args__ = (
        CheckConstraint(
            "end_date > start_date",
            name="ck_backtest_runs_valid_date_range",
        ),
        CheckConstraint(
            "initial_capital > 0",
            name="ck_backtest_runs_initial_capital_positive",
        ),
        CheckConstraint(
            "final_capital >= 0",
            name="ck_backtest_runs_final_capital_non_negative",
        ),
        CheckConstraint(
            "max_drawdown IS NULL "
            "OR (max_drawdown >= 0 AND max_drawdown <= 100)",
            name="ck_backtest_runs_valid_drawdown",
        ),
        CheckConstraint(
            "win_rate IS NULL OR (win_rate >= 0 AND win_rate <= 100)",
            name="ck_backtest_runs_valid_win_rate",
        ),
        CheckConstraint(
            "total_trades >= 0",
            name="ck_backtest_runs_total_trades_non_negative",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')",
            name="ck_backtest_runs_valid_status",
        ),
        Index(
            "ix_backtest_runs_stock_created_at",
            "stock_id",
            "created_at",
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    stock_id: Mapped[int | None] = mapped_column(
        ForeignKey("stocks.id", ondelete="SET NULL"),
        nullable=True,
    )

    name: Mapped[str] = mapped_column(
        String(150),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    timeframe: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="15minute",
        server_default="15minute",
    )

    start_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    end_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    initial_capital: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    final_capital: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    total_return: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    sharpe_ratio: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    max_drawdown: Mapped[Decimal | None] = mapped_column(
        Numeric(10, 4),
        nullable=True,
    )

    win_rate: Mapped[Decimal | None] = mapped_column(
        Numeric(7, 4),
        nullable=True,
    )

    total_trades: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    parameters: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="PENDING",
        server_default="PENDING",
    )

    error_message: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    stock: Mapped[Stock | None] = relationship(
        back_populates="backtest_runs",
    )

    def __repr__(self) -> str:
        return (
            f"BacktestRun(id={self.id!r}, name={self.name!r}, "
            f"status={self.status!r})"
        )