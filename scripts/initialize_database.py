from sqlalchemy import inspect, text

from backend.app import models  # noqa: F401
from backend.app.database import Base, engine


def initialize_database() -> None:
    """Create all registered database tables."""

    print("Connecting to PostgreSQL...")

    with engine.connect() as connection:
        database_name = connection.execute(
            text("SELECT current_database()")
        ).scalar_one()

        database_user = connection.execute(
            text("SELECT current_user")
        ).scalar_one()

    print(f"Database: {database_name}")
    print(f"User: {database_user}")

    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    print("\nCreated/available tables:")

    for table_name in sorted(table_names):
        print(f"  - {table_name}")

    print("\nDatabase initialization completed successfully.")


if __name__ == "__main__":
    initialize_database()