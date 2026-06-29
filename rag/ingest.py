"""End-to-end RAG ingestion: embed all Postgres articles into Chroma, then compute RAG features."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.models import Article, RAGFeature
from db.session import get_session, init_db
from rag.embedder import embed_all_articles
from rag.retriever import embed_and_retrieve, get_or_create_collection

logger = logging.getLogger(__name__)


def run_ingest() -> None:
    """Embed all articles into Chroma and compute RAG features for each article."""
    collection = get_or_create_collection()

    logger.info("Step 1: Embedding articles into Chroma...")
    with get_session() as session:
        total_embedded = embed_all_articles(session, collection)
    logger.info("Embedding complete: %d articles processed.", total_embedded)

    logger.info("Step 2: Computing RAG features for all articles...")
    with get_session() as session:
        article_ids = [r[0] for r in session.query(Article.id).all()]

    total = len(article_ids)
    rag_created = 0
    for i, article_id in enumerate(article_ids):
        with get_session() as session:
            article = session.get(Article, article_id)
            if article is None:
                continue
            already_exists = (
                session.query(RAGFeature).filter(RAGFeature.article_id == article_id).first()
            ) is not None
            if already_exists:
                continue
            embed_and_retrieve(article, collection, session)
            rag_created += 1

        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d articles...", i + 1, total)

    logger.info("RAG ingestion complete.")
    logger.info("  Articles embedded into Chroma : %d", total_embedded)
    logger.info("  RAG feature rows created      : %d", rag_created)


if __name__ == "__main__":
    from config import setup_logging

    setup_logging()
    init_db()
    run_ingest()
