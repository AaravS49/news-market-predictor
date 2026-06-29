"""Builds the 13-dim feature vector for the classifier: sentiment + RAG + price + sector."""
import logging
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import SECTOR_MAP
from db.models import Article, PriceRecord, RAGFeature, SentimentScore
from rag.retriever import embed_and_retrieve

logger = logging.getLogger(__name__)

SECTOR_ONEHOT: dict[str, int] = {
    "Tech":       0,
    "Finance":    1,
    "Healthcare": 2,
    "Energy":     3,
    "Consumer":   4,
}

FEATURE_NAMES: list[str] = [
    "sentiment_score",
    "mean_analog_return", "analog_hit_rate", "analog_volatility", "n_analogs",
    "momentum_5d", "volume_spike", "price_volatility",
    "sector_tech", "sector_finance", "sector_healthcare", "sector_energy", "sector_consumer",
]

_FINBERT = None


def _get_finbert():
    """Return the cached FinBERT sentiment pipeline, loading it on first call."""
    global _FINBERT
    if _FINBERT is None:
        from transformers import pipeline as hf_pipeline
        logger.info("Loading FinBERT...")
        _FINBERT = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
    return _FINBERT


def get_sentiment_score(article: Article, session) -> float:
    """Return FinBERT score in [-1, 1]; compute once and cache in SentimentScore table."""
    existing = session.query(SentimentScore).filter(
        SentimentScore.article_id == article.id
    ).first()
    if existing:
        return existing.score

    text = f"{article.headline} {article.body or ''}"[:512].strip()
    result = _get_finbert()(text)[0]
    label = result["label"].lower()
    raw = result["score"]
    score = raw if label == "positive" else (-raw if label == "negative" else 0.0)

    session.add(SentimentScore(
        article_id=article.id,
        score=score,
        model_version="finbert-v1",
    ))
    session.commit()
    return score


def get_price_features(ticker: str, published_at: datetime, session) -> dict:
    """Return momentum_5d, volume_spike, price_volatility from PriceRecord history.

    Returns zero defaults if insufficient price history exists.
    """
    pub_date = published_at.date() if hasattr(published_at, "date") else published_at

    records = (
        session.query(PriceRecord)
        .filter(PriceRecord.ticker == ticker, PriceRecord.date <= pub_date)
        .order_by(PriceRecord.date.desc())
        .limit(22)
        .all()
    )
    records = list(reversed(records))  # oldest → newest

    zeros = {"momentum_5d": 0.0, "volume_spike": 1.0, "price_volatility": 0.0}
    if len(records) < 2:
        return zeros

    target = records[-1]

    momentum_5d = (
        (target.close - records[-6].close) / records[-6].close
        if len(records) >= 6 else 0.0
    )

    prior_vols = [r.volume for r in records[:-1]]
    if prior_vols:
        mean_vol = mean(prior_vols[-20:])
        volume_spike = target.volume / mean_vol if mean_vol > 0 else 1.0
    else:
        volume_spike = 1.0

    closes = [r.close for r in records]
    daily_rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
    price_volatility = stdev(daily_rets[-20:]) if len(daily_rets) >= 2 else 0.0

    return {
        "momentum_5d": momentum_5d,
        "volume_spike": volume_spike,
        "price_volatility": price_volatility,
    }


def get_sector_onehot(ticker: str) -> list[float]:
    """Return a 5-dim one-hot vector for the ticker's sector."""
    sector = SECTOR_MAP.get(ticker, "Tech")
    idx = SECTOR_ONEHOT.get(sector, 0)
    vec = [0.0] * 5
    vec[idx] = 1.0
    return vec


def _get_label_and_return(
    article: Article, session
) -> tuple[int | None, float | None]:
    """Look up the 3-day return for an article's ticker/date; return (label, return_3d)."""
    if article.published_at is None:
        return None, None
    pub_date = article.published_at.date()
    pr = (
        session.query(PriceRecord)
        .filter(
            PriceRecord.ticker == article.ticker,
            PriceRecord.date >= pub_date,
        )
        .order_by(PriceRecord.date.asc())
        .first()
    )
    if pr is None or pr.return_3d is None:
        return None, None
    return (1 if pr.return_3d > 0 else 0), pr.return_3d


def build_feature_vector(
    article: Article, session, chroma_collection
) -> tuple[torch.Tensor | None, int | None]:
    """Assemble a 13-dim feature tensor and binary label (1=up, 0=down).

    Returns (None, None) when return_3d is not yet available for this article.
    """
    label, _ = _get_label_and_return(article, session)
    if label is None:
        return None, None

    sentiment = get_sentiment_score(article, session)

    rag_row = session.query(RAGFeature).filter(RAGFeature.article_id == article.id).first()
    if rag_row is None:
        feats = embed_and_retrieve(article, chroma_collection, session)
        mean_ar = feats["mean_analog_return"]
        hit_rate = feats["analog_hit_rate"]
        volatility = feats["analog_volatility"]
        n_analogs = float(feats["n_analogs"])
    else:
        mean_ar = rag_row.mean_analog_return or 0.0
        hit_rate = rag_row.analog_hit_rate or 0.0
        volatility = rag_row.analog_volatility or 0.0
        n_analogs = float(rag_row.n_analogs or 0)

    pf = get_price_features(article.ticker, article.published_at, session)
    sector_vec = get_sector_onehot(article.ticker)

    vec = [
        sentiment,
        mean_ar, hit_rate, volatility, n_analogs,
        pf["momentum_5d"], pf["volume_spike"], pf["price_volatility"],
        *sector_vec,
    ]
    return torch.tensor(vec, dtype=torch.float32), label


def build_dataset(
    session, chroma_collection
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (X, y, returns) tensors for all articles that have a known 3d return.

    Shapes: X=(n,13), y=(n,) binary, returns=(n,) raw return_3d values.
    """
    articles = session.query(Article).all()
    X_list, y_list, ret_list = [], [], []
    skipped = 0

    for i, article in enumerate(articles):
        label, ret = _get_label_and_return(article, session)
        if label is None:
            skipped += 1
            continue
        tensor, _ = build_feature_vector(article, session, chroma_collection)
        if tensor is None:
            skipped += 1
            continue
        X_list.append(tensor)
        y_list.append(float(label))
        ret_list.append(float(ret))
        if (i + 1) % 100 == 0:
            logger.info("Features built for %d/%d articles...", i + 1, len(articles))

    X = torch.stack(X_list)
    y = torch.tensor(y_list, dtype=torch.float32)
    returns = torch.tensor(ret_list, dtype=torch.float32)

    pos = int(y.sum().item())
    neg = len(y) - pos
    logger.info(
        "Dataset: %d samples | pos=%d (%.1f%%) | neg=%d (%.1f%%) | skipped=%d",
        len(y), pos, 100 * pos / len(y), neg, 100 * neg / len(y), skipped,
    )
    return X, y, returns


if __name__ == "__main__":
    from config import setup_logging
    from db.session import get_session, init_db
    from rag.retriever import get_or_create_collection
    from datetime import date, timedelta

    setup_logging()
    init_db()
    collection = get_or_create_collection()
    cutoff = date.today() - timedelta(days=5)

    with get_session() as session:
        articles = (
            session.query(Article)
            .filter(Article.ticker == "AAPL", Article.published_at.isnot(None))
            .all()
        )
        article = next(
            (a for a in articles
             if a.published_at and a.published_at.date() <= cutoff),
            None,
        )
        if not article:
            print("No labeled AAPL article found.")
        else:
            print(f"Article : {article.headline}")
            print(f"Date    : {article.published_at}\n")
            tensor, label = build_feature_vector(article, session, collection)
            if tensor is None:
                print("Label unavailable (return_3d null).")
            else:
                print(f"Label: {label} ({'UP' if label == 1 else 'DOWN'})\n")
                print("13 features:")
                for name, val in zip(FEATURE_NAMES, tensor.tolist()):
                    flag = " <- NaN!" if val != val else ""
                    print(f"  {name:<30s}: {val:>10.6f}{flag}")
