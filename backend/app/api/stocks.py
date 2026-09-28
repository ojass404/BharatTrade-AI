from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database import get_db
from backend.app.schemas.stock import (
    StockCreate,
    StockResponse,
    StockUpdate,
)
from backend.app.services.stock_service import (
    create_stock,
    find_duplicate_stock,
    get_stock_by_id,
    list_stocks,
    update_stock,
)

router = APIRouter(prefix="/stocks", tags=["Stocks"])


@router.post(
    "",
    response_model=StockResponse,
    status_code=status.HTTP_201_CREATED,
)
def register_stock(
    stock_data: StockCreate,
    db: Session = Depends(get_db),
) -> StockResponse:
    duplicate = find_duplicate_stock(
        db=db,
        symbol=stock_data.symbol,
        instrument_key=stock_data.instrument_key,
    )

    if duplicate is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A stock with this symbol or instrument key already exists.",
        )

    return create_stock(db, stock_data)


@router.get("", response_model=list[StockResponse])
def get_all_stocks(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
    active_only: bool = False,
    db: Session = Depends(get_db),
) -> list[StockResponse]:
    return list_stocks(
        db=db,
        skip=skip,
        limit=limit,
        active_only=active_only,
    )


@router.get("/{stock_id}", response_model=StockResponse)
def get_stock(
    stock_id: int,
    db: Session = Depends(get_db),
) -> StockResponse:
    stock = get_stock_by_id(db, stock_id)

    if stock is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stock not found.",
        )

    return stock


@router.patch("/{stock_id}", response_model=StockResponse)
def modify_stock(
    stock_id: int,
    stock_data: StockUpdate,
    db: Session = Depends(get_db),
) -> StockResponse:
    stock = get_stock_by_id(db, stock_id)

    if stock is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stock not found.",
        )

    if stock_data.instrument_key is not None:
        duplicate = find_duplicate_stock(
            db=db,
            symbol=stock.symbol,
            instrument_key=stock_data.instrument_key,
            exclude_stock_id=stock.id,
        )

        if duplicate is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The instrument key is already assigned to another stock.",
            )

    return update_stock(db, stock, stock_data)