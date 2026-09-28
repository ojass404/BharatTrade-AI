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
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database import Base


class ModelPrediction(Base):
    """Prediction produced produced by the Conv1D–LSTM ensemble."""

    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    stock_id: Mapped[int] = mapped_column(
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False,
    )

    input_end_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    target_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    timeframe: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="15minute",
    )

    conv1d_probability: Mapped[Decimal] = mapped_column(
        Numeric(7, 6),
        nullable=False,
    )

    lstm_probability: Mapped[Decimal] = mapped_column(
        Numeric(7, 6),
        nullable=False,
    )

    ensemble_probability: Mapped[Decimal] = mapped_column(
        Numeric(7, 6),
        nullable=False,
    )

    signal: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
    )

    confidence: Mapped[Decimal] = mapped_column(
        Numeric(7, 6),
        nullable=False,
    )

    predicted_return: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 8),
        nullable=True,
    )

    actual_return: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 8),
        nullable=True,
    )

    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="1.0.0",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    stock: Mapped["Stock"] = relationship(
        "Stock",
        back_populates="predictions",
    )

    paper_trades: Mapped[list["PaperTrade"]] = relationship(
        "PaperTrade",
        back_populates="prediction",
    )

    __table_args__ = (
        UniqueConstraint(
            "stock_id",
            "input_end_time",
            "target_time",
            "timeframe",
            "model_version",
            name="uq_prediction_stock_target_model",
        ),
        CheckConstraint(
            "target_time > input_end_time",
            name="ck_predictions_target_after_input",
        ),
        CheckConstraint(
            "conv1d_probability BETWEEN 0 AND 1",
            name="ck_predictions_conv1d_probability",
        ),
        CheckConstraint(
            "lstm_probability BETWEEN 0 AND 1",
            name="ck_predictions_lstm_probability",
        ),
        CheckConstraint(
            "ensemble_probability BETWEEN 0 AND 1",
            name="ck_predictions_ensemble_probability",
        ),
        CheckConstraint(
            "confidence BETWEEN 0 AND 1",
            name="ck_predictions_confidence",
        ),
        CheckConstraint(
            "signal IN ('BUY', 'HOLD', 'SELL')",
            name="ck_predictions_valid_signal",
        ),
        Index(
            "ix_predictions_stock_target_time",
            "stock_id",
            "target_time",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"ModelPrediction(stock_id={self.stock_id!r}, "
            f"target_time={self.target_time!r}, "
            f"signal={self.signal!r})"
        )