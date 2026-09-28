from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


class PaperTrade(Base):
    """Simulated Buy or Sell transaction without real-money execution."""

    __tablename__ = "paper_trades"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
    )

    stock_id: Mapped[int] = mapped_column(
        ForeignKey("stocks.id", ondelete="RESTRICT"),
        nullable=False,
    )

    prediction_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_predictions.id", ondelete="SET NULL"),
        nullable=True,
    )

    side: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )

    quantity: Mapped[int] = mapped_column(
        nullable=False,
    )

    execution_price: Mapped[Decimal] = mapped_column(
        Numeric(18, 4),
        nullable=False,
    )

    gross_value: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    transaction_cost: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=0,
    )

    realized_profit_loss: Mapped[Decimal | None] = mapped_column(
        Numeric(18, 2),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="EXECUTED",
    )

    executed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    portfolio: Mapped["Portfolio"] = relationship(
        "Portfolio",
        back_populates="paper_trades",
    )

    stock: Mapped["Stock"] = relationship(
        "Stock",
        back_populates="paper_trades",
    )

    prediction: Mapped["ModelPrediction | None"] = relationship(
        "ModelPrediction",
        back_populates="paper_trades",
    )

    __table_args__ = (
        CheckConstraint(
            "side IN ('BUY', 'SELL')",
            name="ck_paper_trades_valid_side",
        ),
        CheckConstraint(
            "quantity > 0",
            name="ck_paper_trades_quantity_positive",
        ),
        CheckConstraint(
            "execution_price > 0",
            name="ck_paper_trades_price_positive",
        ),
        CheckConstraint(
            "gross_value > 0",
            name="ck_paper_trades_gross_value_positive",
        ),
        CheckConstraint(
            "transaction_cost >= 0",
            name="ck_paper_trades_cost_non_negative",
        ),
        CheckConstraint(
            "status IN ('PENDING', 'EXECUTED', 'REJECTED', 'CANCELLED')",
            name="ck_paper_trades_valid_status",
        ),
        Index(
            "ix_paper_trades_portfolio_execution",
            "portfolio_id",
            "executed_at",
        ),
        Index(
            "ix_paper_trades_stock_execution",
            "stock_id",
            "executed_at",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"PaperTrade(id={self.id!r}, side={self.side!r}, "
            f"quantity={self.quantity!r})"
        )