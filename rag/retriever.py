"""Retrieves top-K similar past articles from Chroma and computes RAG features."""
import logging
import sys
from pathlib import Path
from statistics import mean, stdev

import chromadb
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RAG_TOP_K
from db.models import Article, PriceRecord, RAGFeature
from rag.embedder import embed_text

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).resolve().parents[1] / "chroma_store"

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    """Return the cached persistent Chroma client, creating it on first call."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_or_create_collection():
    """Return (or create) the persistent Chroma collection for article embeddings."""
    return _get_client().get_or_create_collection(
        name="articles",
        metadata={"hnsw:space": "cosine"},
    )


def retrieve_analogs(
    article: Article,
    collection,
    session: Session,
    top_k: int = RAG_TOP_K,
) -> list[dict]:
    """Return up to top_k past articles for the same ticker with known return_3d values.

    The ticker filter in Chroma prevents cross-company contamination.
    Analogs whose return_3d is NULL are silently skipped.
    """
    body = (article.body or "").strip()
    text = f"{article.headline} [SEP] {body}" if body else article.headline
    embedding = embed_text(text)

    total_in_collection = collection.count()
    if total_in_collection == 0:
        return []

    n_req = min(top_k + 1, total_in_collection)  # +1 to absorb potential self-result

    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_req,
            where={"ticker": article.ticker},
            include=["metadatas", "distances"],
        )
    except Exception:
        logger.debug("Chroma query failed for article %s", article.id, exc_info=True)
        return []

    ids = results["ids"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    analogs: list[dict] = []
    for aid, _, dist in zip(ids, metadatas, distances):
        if int(aid) == article.id:
            continue  # exclude self

        analog_article = session.get(Article, int(aid))
        if analog_article is None or analog_article.published_at is None:
            continue

        pub_date = analog_article.published_at.date()

        # Find the price record on or after publication (nearest trading day forward)
        price_record = (
            session.query(PriceRecord)
            .filter(
                PriceRecord.ticker == analog_article.ticker,
                PriceRecord.date >= pub_date,
            )
            .order_by(PriceRecord.date.asc())
            .first()
        )

        if price_record is None or price_record.return_3d is None:
            continue

        analogs.append({
            "article_id": int(aid),
            "headline": analog_article.headline,
            "published_at": analog_article.published_at,
            "similarity_score": round(1.0 - dist, 4),  # cosine dist → similarity
            "return_3d": price_record.return_3d,
        })

    return analogs[:top_k]


def compute_rag_features(analogs: list[dict]) -> dict:
    """Aggregate analog returns into scalar features for the classifier.

    Returns a dict with keys: mean_analog_return, analog_hit_rate,
    analog_volatility, n_analogs. All zero when analog list is empty.
    """
    if not analogs:
        return {
            "mean_analog_return": 0.0,
            "analog_hit_rate": 0.0,
            "analog_volatility": 0.0,
            "n_analogs": 0,
        }

    returns = [a["return_3d"] for a in analogs]
    return {
        "mean_analog_return": mean(returns),
        "analog_hit_rate": sum(1 for r in returns if r > 0) / len(returns),
        "analog_volatility": stdev(returns) if len(returns) > 1 else 0.0,
        "n_analogs": len(analogs),
    }


def embed_and_retrieve(article: Article, collection, session: Session) -> dict:
    """Retrieve analogs, compute RAG features, persist to RAGFeature table, return features."""
    analogs = retrieve_analogs(article, collection, session)
    features = compute_rag_features(analogs)

    existing = session.query(RAGFeature).filter(RAGFeature.article_id == article.id).first()
    if existing is None:
        session.add(RAGFeature(
            article_id=article.id,
            mean_analog_return=features["mean_analog_return"],
            analog_hit_rate=features["analog_hit_rate"],
            analog_volatility=features["analog_volatility"],
            n_analogs=features["n_analogs"],
        ))
        session.commit()

    return features


if __name__ == "__main__":
    import random
    from config import setup_logging
    from db.session import get_session, init_db

    setup_logging()
    init_db()
    collection = get_or_create_collection()

    with get_session() as session:
        aapl_articles = session.query(Article).filter(Article.ticker == "AAPL").all()
        if not aapl_articles:
            logger.warning("No AAPL articles found — run rag/ingest.py first.")
        else:
            article = random.choice(aapl_articles)
            print(f"Query article : {article.headline}")
            print(f"Published     : {article.published_at}")
            print(f"Ticker        : {article.ticker}\n")

            analogs = retrieve_analogs(article, collection, session)
            features = compute_rag_features(analogs)

            if not analogs:
                print("No analogs found (expected if very few AAPL articles in Chroma).")
            else:
                print(f"Top-{len(analogs)} analogs:")
                for a in analogs:
                    arrow = "↑" if a["return_3d"] > 0 else "↓"
                    print(
                        f"  {arrow} return_3d={a['return_3d']:+.4f}  "
                        f"sim={a['similarity_score']:.4f}  "
                        f"{a['headline'][:70]}"
                    )

            print("\nRAG features:")
            for k, v in features.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
