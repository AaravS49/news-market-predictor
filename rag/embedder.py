"""Generates sentence embeddings with all-MiniLM-L6-v2 and upserts them into Chroma."""
import logging
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.models import Article

logger = logging.getLogger(__name__)

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Return the cached SentenceTransformer singleton, loading it on first call."""
    global _MODEL
    if _MODEL is None:
        logger.info("Loading all-MiniLM-L6-v2...")
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def embed_text(text: str) -> list[float]:
    """Encode a text string and return its 384-dim embedding as a plain float list."""
    return _get_model().encode(text, normalize_embeddings=True).tolist()


def embed_and_store_article(article: Article, collection) -> None:
    """Embed an article and upsert it into Chroma; skip if the id is already present."""
    existing = collection.get(ids=[str(article.id)])
    if existing["ids"]:
        return

    body = (article.body or "").strip()
    text = f"{article.headline} [SEP] {body}" if body else article.headline

    embedding = embed_text(text)
    published_str = article.published_at.isoformat() if article.published_at else ""

    collection.upsert(
        ids=[str(article.id)],
        embeddings=[embedding],
        metadatas=[{
            "ticker": article.ticker,
            "published_at": published_str,
            "article_id": article.id,
            "url": article.url or "",
        }],
    )


def embed_all_articles(session: Session, collection) -> int:
    """Embed every article in Postgres into Chroma; log progress every 50. Return total count."""
    articles = session.query(Article).all()
    total = len(articles)
    for i, article in enumerate(articles):
        embed_and_store_article(article, collection)
        if (i + 1) % 50 == 0:
            logger.info("Embedded %d/%d articles...", i + 1, total)
    logger.info("Embedded %d/%d articles.", total, total)
    return total
