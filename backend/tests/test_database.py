from sqlalchemy import inspect, text

from backend.app import models  # noqa: F401
from backend.app.database import Base, SessionLocal, engine


def test_database_connection() -> None:
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT 1")
        ).scalar_one()

    assert result == 1


def test_required_tables_exist() -> None:
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    required_tables = {
        "stocks",
        "market_candles",
        "model_predictions",
        "portfolios",
        "paper_trades",
        "backtest_runs",
    }

    assert required_tables.issubset(existing_tables)


def test_database_session() -> None:
    database_session = SessionLocal()

    try:
        result = database_session.execute(
            text("SELECT current_database()")
        ).scalar_one()

        assert result == "stock_ai"
    finally:
        database_session.close()