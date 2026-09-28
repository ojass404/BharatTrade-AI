import argparse
import gzip
import json
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database import SessionLocal
from backend.app.models.stock import Stock


NSE_INSTRUMENTS_URL = (
    "https://assets.upstox.com/"
    "market-quote/instruments/exchange/NSE.json.gz"
)


def download_instruments() -> list[dict[str, Any]]:
    print("Downloading the latest NSE instrument master...")

    response = httpx.get(
        NSE_INSTRUMENTS_URL,
        timeout=60.0,
        follow_redirects=True,
    )
    response.raise_for_status()

    content = response.content

    # Some servers return decompressed content automatically.
    if content.startswith(b"\x1f\x8b"):
        content = gzip.decompress(content)

    instruments = json.loads(content.decode("utf-8"))

    if not isinstance(instruments, list):
        raise ValueError("Unexpected Upstox instrument-file format.")

    return instruments


def filter_nse_equities(
    instruments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    equities = []

    for instrument in instruments:
        if (
            instrument.get("segment") == "NSE_EQ"
            and instrument.get("instrument_type") == "EQ"
            and instrument.get("instrument_key")
            and instrument.get("trading_symbol")
            and instrument.get("name")
        ):
            equities.append(instrument)

    return equities


def synchronise_stocks(
    db: Session,
    instruments: list[dict[str, Any]],
    activate_symbols: set[str],
    dry_run: bool,
) -> dict[str, int]:
    existing_stocks = list(db.scalars(select(Stock)).all())

    stocks_by_instrument_key = {
        stock.instrument_key: stock
        for stock in existing_stocks
    }

    stocks_by_symbol = {
        stock.symbol: stock
        for stock in existing_stocks
    }

    created = 0
    updated = 0
    unchanged = 0

    for instrument in instruments:
        symbol = instrument["trading_symbol"].strip().upper()
        company_name = instrument["name"].strip()
        instrument_key = instrument["instrument_key"].strip()

        should_activate = symbol in activate_symbols

        stock = stocks_by_instrument_key.get(instrument_key)

        if stock is None:
            stock = stocks_by_symbol.get(symbol)

        if stock is None:
            stock = Stock(
                symbol=symbol,
                company_name=company_name,
                exchange="NSE",
                instrument_key=instrument_key,
                sector=None,
                is_active=should_activate,
            )

            db.add(stock)

            stocks_by_symbol[symbol] = stock
            stocks_by_instrument_key[instrument_key] = stock

            created += 1
            continue

        changed = False

        if stock.company_name != company_name:
            stock.company_name = company_name
            changed = True

        if stock.instrument_key != instrument_key:
            stock.instrument_key = instrument_key
            changed = True

        if stock.exchange != "NSE":
            stock.exchange = "NSE"
            changed = True

        # Activate requested stocks, but never deactivate an existing stock.
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
        "downloaded_equities": len(instruments),
        "created": created,
        "updated": updated,
        "unchanged": unchanged,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import NSE equity instruments from Upstox."
    )

    parser.add_argument(
        "--activate",
        default="",
        help=(
            "Comma-separated stock symbols to activate, "
            "for example RELIANCE,TCS,INFY."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the import without modifying the database.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    activate_symbols = {
        symbol.strip().upper()
        for symbol in arguments.activate.split(",")
        if symbol.strip()
    }

    all_instruments = download_instruments()
    nse_equities = filter_nse_equities(all_instruments)

    print(f"Total instruments downloaded: {len(all_instruments)}")
    print(f"NSE EQ instruments selected: {len(nse_equities)}")

    if activate_symbols:
        print(
            "Stocks requested for activation: "
            + ", ".join(sorted(activate_symbols))
        )

    with SessionLocal() as db:
        result = synchronise_stocks(
            db=db,
            instruments=nse_equities,
            activate_symbols=activate_symbols,
            dry_run=arguments.dry_run,
        )

    print("\nImport summary")
    print(f"Downloaded equities : {result['downloaded_equities']}")
    print(f"Created             : {result['created']}")
    print(f"Updated             : {result['updated']}")
    print(f"Unchanged           : {result['unchanged']}")
    print(f"Database modified   : {not arguments.dry_run}")


if __name__ == "__main__":
    main()