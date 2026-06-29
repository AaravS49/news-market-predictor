"""Fast sanity-check: ingest prices + news for AAPL, MSFT, NVDA and print table counts."""
import time
from datetime import date, timedelta

from config import setup_logging
from db.session import get_session, init_db
from db.models import Article, PriceRecord
from data.ingestion.fetch_prices import fetch_and_store_prices, compute_and_store_3d_returns
from data.ingestion.fetch_news import fetch_and_store_articles, COMPANY_NAMES

SEED_TICKERS = ["AAPL", "MSFT", "NVDA"]


def main() -> None:
    """Insert seed data for 3 tickers and confirm the DB round-trip."""
    setup_logging()
    init_db()
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=730)).isoformat()

    print(f"Seeding prices {start} → {end} for {SEED_TICKERS}...")
    with get_session() as session:
        for ticker in SEED_TICKERS:
            inserted = fetch_and_store_prices(ticker, start, end, session)
            updated = compute_and_store_3d_returns(ticker, session)
            print(f"  {ticker}: {inserted} price rows inserted, {updated} returns computed")

    print("\nFetching news articles...")
    with get_session() as session:
        for ticker in SEED_TICKERS:
            count = fetch_and_store_articles(ticker, COMPANY_NAMES[ticker], session)
            print(f"  {ticker}: {count} articles inserted")
            time.sleep(0.5)

    print("\n--- Row counts ---")
    with get_session() as session:
        for ticker in SEED_TICKERS:
            price_rows = session.query(PriceRecord).filter(PriceRecord.ticker == ticker).count()
            returns_ok = (
                session.query(PriceRecord)
                .filter(PriceRecord.ticker == ticker, PriceRecord.return_3d.isnot(None))
                .count()
            )
            article_rows = session.query(Article).filter(Article.ticker == ticker).count()
            print(
                f"  {ticker}: price_records={price_rows} "
                f"(return_3d filled={returns_ok}), articles={article_rows}"
            )


if __name__ == "__main__":
    main()
