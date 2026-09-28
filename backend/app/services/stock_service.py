from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from backend.app.models.stock import Stock
from backend.app.schemas.stock import StockCreate, StockUpdate


def get_stock_by_id(db: Session, stock_id: int) -> Stock | None:
    statement = select(Stock).where(Stock.id == stock_id)
    return db.scalar(statement)


def get_stock_by_symbol(db: Session, symbol: str) -> Stock | None:
    statement = select(Stock).where(
        Stock.symbol == symbol.strip().upper()
    )
    return db.scalar(statement)


def find_duplicate_stock(
    db: Session,
    symbol: str,
    instrument_key: str,
    exclude_stock_id: int | None = None,
) -> Stock | None:
    statement = select(Stock).where(
        or_(
            Stock.symbol == symbol.strip().upper(),
            Stock.instrument_key == instrument_key.strip(),
        )
    )

    if exclude_stock_id is not None:
        statement = statement.where(Stock.id != exclude_stock_id)

    return db.scalar(statement)


def list_stocks(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    active_only: bool = False,
) -> list[Stock]:
    statement = select(Stock).order_by(Stock.symbol)

    if active_only:
        statement = statement.where(Stock.is_active.is_(True))

    statement = statement.offset(skip).limit(limit)

    return list(db.scalars(statement).all())


def create_stock(db: Session, stock_data: StockCreate) -> Stock:
    stock = Stock(**stock_data.model_dump())

    db.add(stock)
    db.commit()
    db.refresh(stock)

    return stock


def update_stock(
    db: Session,
    stock: Stock,
    stock_data: StockUpdate,
) -> Stock:
    update_values = stock_data.model_dump(exclude_unset=True)

    for field_name, field_value in update_values.items():
        setattr(stock, field_name, field_value)

    db.commit()
    db.refresh(stock)

    return stock