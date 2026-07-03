"""FastAPI application — prediction endpoint and supporting routes."""
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODEL_VERSION, SECTOR_MAP, TICKERS
from db.models import Article, Prediction
from db.session import get_session, init_db
from features.builder import (
    _get_finbert,
    get_price_features,
    get_sector_onehot,
    get_sentiment_score,
)
from model.architecture import NewsMarketClassifier
from rag.retriever import compute_rag_features, get_or_create_collection, retrieve_analogs

logger = logging.getLogger(__name__)

# ── singletons (populated during startup) ────────────────────────────────────
_model: NewsMarketClassifier | None = None
_collection = None
_device: torch.device = torch.device("cpu")

CHECKPOINT = Path(__file__).resolve().parents[1] / "model" / "checkpoints" / "best_model.pt"


# ── Pydantic models ───────────────────────────────────────────────────────────

class ArticleInput(BaseModel):
    ticker: str
    headline: str
    body: str | None = None
    published_at: datetime
    url: str | None = None


class AnalogArticle(BaseModel):
    headline: str
    published_at: datetime
    similarity_score: float
    return_3d: float


class PredictionResponse(BaseModel):
    prediction_id: int | None
    ticker: str
    headline: str
    predicted_prob: float
    predicted_label: int
    confidence: str
    sentiment_score: float
    rag_features: dict
    analog_articles: list[AnalogArticle]
    model_version: str


class HistoryEntry(BaseModel):
    prediction_id: int
    headline: str
    predicted_prob: float
    predicted_label: int
    actual_label: int | None
    created_at: datetime


class FeedbackInput(BaseModel):
    actual_label: int  # 0 or 1


# ── startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy singletons once at startup; fail loudly if anything is missing."""
    global _model, _collection, _device

    import torch as _torch
    from rag.embedder import _get_model as _load_st

    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        force=True,
    )

    _device = _torch.device("mps" if _torch.backends.mps.is_available() else "cpu")

    if not CHECKPOINT.exists():
        raise RuntimeError(
            f"Model checkpoint not found: {CHECKPOINT}. Run model/train.py first."
        )

    _model = NewsMarketClassifier.load(str(CHECKPOINT))
    _model = _model.to(_device)
    _model.eval()

    _collection = get_or_create_collection()

    # Warm up sentence-transformer and FinBERT so first request isn't slow
    _load_st()
    _get_finbert()

    init_db()

    with get_session() as session:
        n_db = session.query(Article).count()

    logger.info(
        "Startup OK: device=%s | chroma=%d | db_articles=%d | model loaded",
        _device, _collection.count(), n_db,
    )
    yield


# ── app ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="News Market Predictor",
    description="Predict 3-day stock return direction from news articles using RAG + PyTorch.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

import api.middleware as _mw  # noqa: E402
from api.middleware import LoggingMiddleware  # noqa: E402

app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _sentiment_inline(headline: str, body: str | None) -> float:
    """Compute FinBERT sentiment without touching the DB (for transient articles)."""
    text = f"{headline} {body or ''}"[:512]
    result = _get_finbert()(text)[0]
    lbl = result["label"].lower()
    raw = result["score"]
    return raw if lbl == "positive" else (-raw if lbl == "negative" else 0.0)


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(payload: ArticleInput):
    """Run the full pipeline on a news article and return a 3-day return direction prediction.

    If `url` is provided and not already in the DB, the article and prediction are persisted.
    Without a `url`, the computation is stateless (nothing written to DB).
    """
    if payload.ticker not in TICKERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Ticker '{payload.ticker}' is not supported. "
                f"Supported: {', '.join(sorted(TICKERS))}"
            ),
        )

    try:
        with get_session() as session:
            # Resolve article: check DB by URL, else create transient object
            article = None
            if payload.url:
                article = session.query(Article).filter(Article.url == payload.url).first()

            if article is None:
                article = Article(
                    ticker=payload.ticker,
                    headline=payload.headline,
                    body=payload.body,
                    url=payload.url,
                    published_at=payload.published_at,
                    source="api-predict",
                )
                if payload.url:
                    session.add(article)
                    session.flush()  # populate article.id without committing yet

            analogs = retrieve_analogs(article, _collection, session)
            rag_feats = compute_rag_features(analogs)

            sentiment = (
                get_sentiment_score(article, session)
                if article.id is not None
                else _sentiment_inline(payload.headline, payload.body)
            )

            price_feats = get_price_features(payload.ticker, payload.published_at, session)

            sector_vec = get_sector_onehot(payload.ticker)
            vec = [
                sentiment,
                rag_feats["mean_analog_return"],
                rag_feats["analog_hit_rate"],
                rag_feats["analog_volatility"],
                float(rag_feats["n_analogs"]),
                price_feats["momentum_5d"],
                price_feats["volume_spike"],
                price_feats["price_volatility"],
                *sector_vec,
            ]
            x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(_device)

            with torch.no_grad():
                prob = float(_model(x).item())

            pred_label = 1 if prob >= 0.5 else 0
            confidence = "high" if prob > 0.7 or prob < 0.3 else "low"

            prediction_id: int | None = None
            if article.id is not None:
                pred_row = Prediction(
                    article_id=article.id,
                    predicted_prob=prob,
                    predicted_label=pred_label,
                )
                session.add(pred_row)
                session.commit()
                prediction_id = pred_row.id

            return PredictionResponse(
                prediction_id=prediction_id,
                ticker=payload.ticker,
                headline=payload.headline,
                predicted_prob=round(prob, 4),
                predicted_label=pred_label,
                confidence=confidence,
                sentiment_score=round(sentiment, 4),
                rag_features=rag_feats,
                analog_articles=[
                    AnalogArticle(
                        headline=a["headline"],
                        published_at=a["published_at"],
                        similarity_score=a["similarity_score"],
                        return_3d=a["return_3d"],
                    )
                    for a in analogs
                ],
                model_version=MODEL_VERSION,
            )

    except HTTPException:
        raise
    except Exception as exc:
        import traceback
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


@app.get("/health", tags=["Meta"])
async def health():
    """Service health check with model and data counts."""
    with get_session() as session:
        n_db = session.query(Article).count()
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_loaded": _model is not None,
        "chroma_article_count": _collection.count() if _collection else 0,
        "db_article_count": n_db,
        "tickers_supported": len(TICKERS),
        "requests_served": _mw.request_count,
    }


@app.get("/tickers", tags=["Meta"])
async def tickers():
    """Return all supported tickers with their sectors."""
    return [{"ticker": t, "sector": SECTOR_MAP[t]} for t in TICKERS]


@app.get("/history/{ticker}", response_model=list[HistoryEntry], tags=["Prediction"])
async def history(ticker: str):
    """Return the 20 most recent predictions for a given ticker."""
    if ticker not in TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' is not supported.")

    with get_session() as session:
        rows = (
            session.query(Prediction, Article)
            .join(Article, Prediction.article_id == Article.id)
            .filter(Article.ticker == ticker)
            .order_by(Prediction.created_at.desc())
            .limit(20)
            .all()
        )
        return [
            HistoryEntry(
                prediction_id=pred.id,
                headline=art.headline,
                predicted_prob=round(pred.predicted_prob, 4),
                predicted_label=pred.predicted_label,
                actual_label=pred.actual_label,
                created_at=pred.created_at,
            )
            for pred, art in rows
        ]


@app.post("/feedback/{prediction_id}", tags=["Prediction"])
async def feedback(prediction_id: int, payload: FeedbackInput):
    """Submit the realized outcome for a stored prediction to close the ML feedback loop."""
    if payload.actual_label not in (0, 1):
        raise HTTPException(status_code=400, detail="actual_label must be 0 or 1.")

    with get_session() as session:
        pred = session.get(Prediction, prediction_id)
        if pred is None:
            raise HTTPException(
                status_code=404, detail=f"Prediction {prediction_id} not found."
            )
        pred.actual_label = payload.actual_label
        session.commit()

    return {
        "status": "ok",
        "prediction_id": prediction_id,
        "actual_label": payload.actual_label,
    }


# ── static frontend (must be mounted last) ───────────────────────────────────
_FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/ui", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")
