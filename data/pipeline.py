"""Orchestrates the full data ingestion pipeline: prices → 3d returns → news → summary."""
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import TICKERS
from db.models import Article, PriceRecord
from data.ingestion.fetch_news import COMPANY_NAMES, fetch_and_store_articles
from data.ingestion.fetch_prices import compute_and_store_3d_returns, fetch_and_store_prices

logger = logging.getLogger(__name__)


def run_full_pipeline(session: Session) -> None:
    """Run the full ingestion pipeline and print a summary table."""
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=730)).isoformat()

    logger.info("[1/3] Fetching prices %s → %s ...", start, end)
    for ticker in TICKERS:
        fetch_and_store_prices(ticker, start, end, session)

    logger.info("[2/3] Computing 3-day returns...")
    for ticker in TICKERS:
        compute_and_store_3d_returns(ticker, session)

    logger.info("[3/3] Fetching news articles (last 30 days)...")
    for ticker in TICKERS:
        fetch_and_store_articles(ticker, COMPANY_NAMES[ticker], session)
        time.sleep(0.5)

    print("\n" + "=" * 62)
    print(f"{'ticker':<8} {'price_rows':>10} {'articles':>9} {'return_3d_ok':>13}")
    print("-" * 62)
    for ticker in TICKERS:
        price_rows = session.query(PriceRecord).filter(PriceRecord.ticker == ticker).count()
        articles = session.query(Article).filter(Article.ticker == ticker).count()
        returns_ok = (
            session.query(PriceRecord)
            .filter(PriceRecord.ticker == ticker, PriceRecord.return_3d.isnot(None))
            .count()
        )
        print(f"{ticker:<8} {price_rows:>10} {articles:>9} {returns_ok:>13}")
    print("=" * 62)


if __name__ == "__main__":
    from config import setup_logging
    from db.session import get_session, init_db

    setup_logging()
    init_db()
    with get_session() as session:
        run_full_pipeline(session)
