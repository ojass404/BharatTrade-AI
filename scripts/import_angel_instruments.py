import argparse
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import SessionLocal
from backend.app.models.stock import Stock


ANGEL_INSTRUMENT_MASTER_URL = (
    "https://margincalculator.angelbroking.com/"
    "OpenAPI_File/files/OpenAPIScripMaster.json"
)


def download_instrument_master() -> list[dict[str, Any]]:
    print("Downloading Angel One instrument master...")

    response = httpx.get(
        ANGEL_INSTRUMENT_MASTER_URL,
        timeout=120.0,
        follow_redirects=True,
    )

    response.raise_for_status()

    instruments = response.json()

    if not isinstance(instruments, list):
        raise ValueError(
            "Angel One returned an unexpected instrument-master format."
        )

    return instruments


def filter_nse_equities(
    instruments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    equities: dict[str, dict[str, Any]] = {}

    for instrument in instruments:
        exchange = str(
            instrument.get("exch_seg", "")
        ).strip().upper()

        trading_symbol = str(
            instrument.get("symbol", "")
        ).strip().upper()

        token = str(
            instrument.get("token", "")
        ).strip()

        if exchange != "NSE":
            continue

        if not trading_symbol.endswith("-EQ"):
            continue

        if not token:
            continue

        symbol = trading_symbol.removesuffix("-EQ")

        equities[symbol] = instrument

    return list(equities.values())


def synchronise_instruments(
    db: Session,
    instruments: list[dict[str, Any]],
    activate_symbols: set[str],
    dry_run: bool,
) -> dict[str, int]:
    existing_stocks = list(
        db.scalars(
            select(Stock).where(Stock.exchange == "NSE")
        ).all()
    )

    stocks_by_symbol = {
        stock.symbol.upper(): stock
        for stock in existing_stocks
    }

    created = 0
    updated = 0
    unchanged = 0

    for instrument in instruments:
        trading_symbol = str(
            instrument["symbol"]
        ).strip().upper()

        symbol = trading_symbol.removesuffix("-EQ")

        token = str(
            instrument["token"]
        ).strip()

        company_name = str(
            instrument.get("name") or symbol
        ).strip()

        angel_instrument_key = f"ANGEL_NSE|{token}"

        should_activate = symbol in activate_symbols

        stock = stocks_by_symbol.get(symbol)

        if stock is None:
            stock = Stock(
                symbol=symbol,
                company_name=company_name,
                exchange="NSE",
                instrument_key=angel_instrument_key,
                sector=None,
                is_active=should_activate,
            )

            db.add(stock)
            stocks_by_symbol[symbol] = stock
            created += 1
            continue

        changed = False

        if stock.instrument_key != angel_instrument_key:
            stock.instrument_key = angel_instrument_key
            changed = True

        if should_activate and not stock.is_active:
            stock.is_active = True
            changed = True

        if changed:
            updated += 1
        else:
            unchanged += 1

    if dry_run:
        db.rollback()
    else:
        db.commit()

    return {
        "nse_equities": len(instruments),
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import Angel One NSE equity symbol tokens "
            "into PostgreSQL."
        )
    )

    parser.add_argument(
        "--activate",
        default="",
        help=(
            "Comma-separated symbols to activate, for example "
            "RELIANCE,TCS,INFY."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying the database.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    activate_symbols = {
        symbol.strip().upper()
        for symbol in arguments.activate.split(",")
        if symbol.strip()
    }

    all_instruments = download_instrument_master()
    nse_equities = filter_nse_equities(all_instruments)

    print(f"Total instruments downloaded: {len(all_instruments)}")
    print(f"NSE equity instruments selected: {len(nse_equities)}")

    if activate_symbols:
        print(
            "Stocks requested for activation: "
            + ", ".join(sorted(activate_symbols))
        )

    with SessionLocal() as db:
        result = synchronise_instruments(
            db=db,
            instruments=nse_equities,
            activate_symbols=activate_symbols,
            dry_run=arguments.dry_run,
        )

    print("\nSynchronization summary")
    print(f"NSE equities       : {result['nse_equities']}")
    print(f"Created            : {result['created']}")
    print(f"Updated            : {result['updated']}")
    print(f"Unchanged          : {result['unchanged']}")
    print(f"Database modified  : {not arguments.dry_run}")


if __name__ == "__main__":
    main()