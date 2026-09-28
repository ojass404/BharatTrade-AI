from __future__ import annotations

from sqlalchemy import inspect, text

from backend.app.database import engine


REQUIRED_COLUMNS = {
    "total_return": "NUMERIC(10, 4)",
    "sharpe_ratio": "NUMERIC(10, 4)",
    "max_drawdown": "NUMERIC(10, 4)",
    "win_rate": "NUMERIC(7, 4)",
    "total_trades": (
        "INTEGER NOT NULL DEFAULT 0"
    ),
    "parameters": (
        "JSON NOT NULL DEFAULT '{}'::json"
    ),
    "status": (
        "VARCHAR(20) NOT NULL DEFAULT 'PENDING'"
    ),
    "error_message": "VARCHAR(500)",
    "completed_at": "TIMESTAMP WITH TIME ZONE",
}


def get_existing_columns() -> set[str]:
    inspector = inspect(engine)

    return {
        column["name"]
        for column in inspector.get_columns("backtest_runs")
    }


def main() -> None:
    existing_columns = get_existing_columns()

    print("Existing backtest_runs columns:")
    for column_name in sorted(existing_columns):
        print(f"  - {column_name}")

    missing_columns = {
        name: definition
        for name, definition in REQUIRED_COLUMNS.items()
        if name not in existing_columns
    }

    if not missing_columns:
        print("\nNo migration required.")
        return

    print("\nAdding missing columns:")

    with engine.begin() as connection:
        for column_name, definition in missing_columns.items():
            print(f"  + {column_name}")

            connection.execute(
                text(
                    "ALTER TABLE backtest_runs "
                    f"ADD COLUMN IF NOT EXISTS "
                    f"{column_name} {definition}"
                )
            )

    final_columns = get_existing_columns()

    still_missing = (
        set(REQUIRED_COLUMNS) - final_columns
    )

    if still_missing:
        raise RuntimeError(
            "Migration incomplete. Missing columns: "
            + ", ".join(sorted(still_missing))
        )

    print("\nBacktest schema migration completed successfully.")


if __name__ == "__main__":
    main()