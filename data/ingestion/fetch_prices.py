"""Fetches OHLCV price data from yfinance and computes 3-day forward returns."""
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import yfinance as yf
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import TICKERS
from db.models import PriceRecord

logger = logging.getLogger(__name__)


def fetch_and_store_prices(ticker: str, start_date: str, end_date: str, session: Session) -> int:
    """Pull OHLCV from yfinance and upsert into PriceRecord; return count of new rows inserted."""
    hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
    if hist.empty:
        logger.warning("No price data returned for %s (%s → %s)", ticker, start_date, end_date)
        return 0

    existing_dates = {
        r[0]
        for r in session.query(PriceRecord.date).filter(PriceRecord.ticker == ticker).all()
    }

    new_rows = 0
    for ts, row in hist.iterrows():
        record_date = ts.date()
        if record_date in existing_dates:
            continue
        session.add(PriceRecord(
            ticker=ticker,
            date=record_date,
            open=float(row["Open"]),
            close=float(row["Close"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            volume=float(row["Volume"]),
        ))
        new_rows += 1

    session.commit()
    return new_rows


def compute_and_store_3d_returns(ticker: str, session: Session) -> int:
    """Fill return_3d for NULL PriceRecord rows using trading-day-aware index arithmetic.

    Uses index+3 in the sorted date list so weekends and holidays are skipped correctly.
    Returns count of rows updated.
    """
    all_records = (
        session.query(PriceRecord)
        .filter(PriceRecord.ticker == ticker)
        .order_by(PriceRecord.date)
        .all()
    )
    if len(all_records) < 4:
        return 0

    sorted_dates = [r.date for r in all_records]
    date_to_close = {r.date: r.close for r in all_records}

    updated = 0
    for i, record in enumerate(all_records):
        if record.return_3d is not None:
            continue
        future_idx = i + 3
        if future_idx >= len(sorted_dates):
            continue  # no future data yet — expected for the last 3 trading days
        close_future = date_to_close[sorted_dates[future_idx]]
        record.return_3d = (close_future - record.close) / record.close
        updated += 1

    session.commit()
    return updated


def fetch_all_tickers(start_date: str, end_date: str, session: Session) -> dict[str, dict]:
    """Fetch prices and compute 3d returns for all 25 config tickers; return stats dict."""
    stats: dict[str, dict] = {}
    for ticker in TICKERS:
        inserted = fetch_and_store_prices(ticker, start_date, end_date, session)
        updated = compute_and_store_3d_returns(ticker, session)
        stats[ticker] = {"inserted": inserted, "returns_updated": updated}
        logger.info("%s: prices_inserted=%d  returns_updated=%d", ticker, inserted, updated)
    return stats


if __name__ == "__main__":
    from config import setup_logging
    from db.session import get_session, init_db

    setup_logging()
    init_db()
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=730)).isoformat()
    logger.info("Fetching prices %s → %s for all tickers...", start, end)
    with get_session() as session:
        fetch_all_tickers(start, end, session)
    logger.info("Done.")
