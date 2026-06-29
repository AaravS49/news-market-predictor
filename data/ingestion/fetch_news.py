"""Fetches news articles from NewsAPI and persists them to the database."""
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from newsapi import NewsApiClient
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import NEWSAPI_KEY, TICKERS
from db.models import Article

logger = logging.getLogger(__name__)

COMPANY_NAMES: dict[str, str] = {
    "AAPL":  "Apple Inc",
    "MSFT":  "Microsoft",
    "NVDA":  "Nvidia",
    "META":  "Meta Platforms",
    "GOOGL": "Alphabet Google",
    "JPM":   "JPMorgan Chase",
    "BAC":   "Bank of America",
    "GS":    "Goldman Sachs",
    "MS":    "Morgan Stanley",
    "WFC":   "Wells Fargo",
    "JNJ":   "Johnson Johnson",
    "UNH":   "UnitedHealth",
    "PFE":   "Pfizer",
    "ABBV":  "AbbVie",
    "MRK":   "Merck",
    "XOM":   "ExxonMobil",
    "CVX":   "Chevron",
    "COP":   "ConocoPhillips",
    "SLB":   "SLB Schlumberger",
    "EOG":   "EOG Resources",
    "AMZN":  "Amazon",
    "TSLA":  "Tesla",
    "HD":    "Home Depot",
    "MCD":   "McDonalds",
    "NKE":   "Nike",
}

_client: NewsApiClient | None = None


def _get_client() -> NewsApiClient:
    """Return a cached NewsApiClient instance."""
    global _client
    if _client is None:
        _client = NewsApiClient(api_key=NEWSAPI_KEY)
    return _client


def fetch_and_store_articles(ticker: str, company_name: str, session: Session) -> int:
    """Query NewsAPI for the past 29 days, deduplicate by URL, store to Article.

    Returns count of new rows inserted.
    """
    client = _get_client()

    to_dt = datetime.now(tz=timezone.utc)
    from_dt = to_dt - timedelta(days=29)  # free tier: max 30 days back

    existing_urls = {
        r[0]
        for r in session.query(Article.url).filter(Article.ticker == ticker).all()
    }

    response = client.get_everything(
        q=company_name,
        from_param=from_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        to=to_dt.strftime("%Y-%m-%dT%H:%M:%S"),
        language="en",
        sort_by="publishedAt",
        page_size=100,
        page=1,
    )

    if response.get("status") != "ok":
        logger.error("[%s] NewsAPI error: %s", ticker, response.get("message", "unknown"))
        return 0

    new_rows = 0
    for item in response.get("articles", []):
        url = item.get("url") or ""
        if not url or url in existing_urls:
            continue

        raw_date = item.get("publishedAt", "")
        try:
            published_at = datetime.strptime(raw_date, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            published_at = None

        session.add(Article(
            ticker=ticker,
            headline=item.get("title") or "",
            body=item.get("content") or item.get("description") or "",
            url=url,
            source=item.get("source", {}).get("name") or "",
            published_at=published_at,
        ))
        existing_urls.add(url)
        new_rows += 1

    session.commit()
    return new_rows


def fetch_all_tickers(session: Session) -> dict[str, int]:
    """Fetch articles for all 25 tickers with a small delay between API calls."""
    counts: dict[str, int] = {}
    for ticker in TICKERS:
        company_name = COMPANY_NAMES[ticker]
        count = fetch_and_store_articles(ticker, company_name, session)
        counts[ticker] = count
        logger.info("%s: articles_inserted=%d", ticker, count)
        time.sleep(0.5)
    return counts


if __name__ == "__main__":
    from config import setup_logging
    from db.session import get_session, init_db

    setup_logging()
    init_db()
    logger.info("Fetching news articles for all tickers (last 30 days)...")
    with get_session() as session:
        counts = fetch_all_tickers(session)
    logger.info("Done. Total articles inserted: %d", sum(counts.values()))
