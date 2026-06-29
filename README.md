# News Market Predictor

A full-stack ML system that predicts short-term stock price movement
from news sentiment, using RAG-augmented feature engineering and a
PyTorch classifier trained on historical market data.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue)
![ChromaDB](https://img.shields.io/badge/ChromaDB-RAG-purple)

---

## Architecture

```
NewsAPI ──► fetch_news.py ──► PostgreSQL (articles)
yfinance ──► fetch_prices.py ──► PostgreSQL (price_records)
                                        │
                                        ▼
                              features/builder.py
                                   │           │
                         RAG retrieval      Price + Sentiment
                         (Chroma)           features (FinBERT)
                                   │
                                   ▼
                          PyTorch Classifier
                           (13-dim input,
                            3 hidden layers,
                            BCELoss + Adam)
                                   │
                                   ▼
                           FastAPI /predict
```

**Ingestion layer** pulls news articles from NewsAPI and OHLCV price data from yfinance into
PostgreSQL. A second pass computes 3-day forward returns on each price record, generating
the binary training labels (positive return = 1, negative = 0). The pipeline is idempotent —
re-runs skip existing rows via URL and ticker+date deduplication.

**RAG pipeline** embeds every article using `all-MiniLM-L6-v2` (384-dim) and stores vectors
in a persistent ChromaDB collection. At prediction time, the incoming article is embedded and
the 5 most semantically similar *past articles for the same ticker* are retrieved. Their
realized 3-day returns serve as four aggregate features (mean return, hit rate, volatility,
count) — giving the model a memory of how the market reacted to structurally similar news.
The ticker filter in the Chroma query is critical: without it, cross-company contamination
would corrupt the analog signal.

**Model design** uses a 13-feature input vector (1 FinBERT sentiment + 4 RAG analog
statistics + 3 price momentum/volatility features + 5-dim sector one-hot). A 3-layer MLP
with BatchNorm and Dropout is trained with BCE loss, Adam optimizer, and early stopping on
validation AUC. FinBERT is chosen over generic sentiment because it is pre-trained on
financial text (earnings reports, analyst notes) and correctly handles hedged language
("earnings beat but outlook cautious") that generic models mislabel.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Ingestion | NewsAPI, yfinance | Article + price collection |
| Storage | PostgreSQL + SQLAlchemy | Structured data + ORM |
| Migrations | Alembic | Schema version control |
| Vector Store | ChromaDB | Semantic article retrieval |
| Embeddings | all-MiniLM-L6-v2 | Article vectorization |
| Sentiment | FinBERT | Finance-domain NLP |
| Model | PyTorch feedforward | Return direction classifier |
| Serving | FastAPI | REST prediction API |

---

## Quickstart

```bash
git clone https://github.com/yourusername/news-market-predictor
cd news-market-predictor

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # fill in NEWSAPI_KEY and DATABASE_URL
createdb news_market_db
alembic upgrade head

python data/pipeline.py     # ingest 2 years of prices + 30 days of news
python rag/ingest.py        # embed articles into Chroma + compute RAG features
python model/train.py       # train classifier (~100 epochs, early stopping)
uvicorn api.main:app --reload --port 8000
```

> **Note:** `chroma_store/` and `model/checkpoints/` are gitignored (large binary artifacts).
> Run `make ingest` and `make train` to regenerate them locally.

---

## API Reference

### POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "headline": "Apple reports record quarterly revenue driven by iPhone 15 sales",
    "body": "Apple Inc reported its highest ever quarterly revenue...",
    "published_at": "2024-11-01T09:00:00",
    "url": "https://example.com/article"
  }'
```

### GET /health
```bash
curl http://localhost:8000/health
```

### GET /tickers
```bash
curl http://localhost:8000/tickers
```

### GET /history/{ticker}
```bash
curl http://localhost:8000/history/AAPL
```

### POST /feedback/{prediction_id}
```bash
curl -X POST http://localhost:8000/feedback/42 \
  -H "Content-Type: application/json" \
  -d '{"actual_label": 1}'
```

Interactive docs at **http://localhost:8000/docs** (Swagger UI, auto-generated).

---

## Model Performance

> Results on 44-sample holdout set (30 days of news — limited by NewsAPI free tier).
> Metrics will improve significantly with larger historical coverage.

| Metric | Value |
|---|---|
| ROC-AUC | 0.96 |
| F1 (macro) | 0.88 |
| Backtest win rate | 92.9% (14 high-confidence trades) |
| Sharpe ratio | 25.3\* |
| Test set size | 44 samples |

\*Sharpe computed per-trade on 3-day returns — inflated by small N. Would normalize on daily P&L curve with a larger dataset.

---

## Design Decisions

- **RAG features over LLM pass-through**: Passing articles to an LLM at inference time is expensive, slow, and non-deterministic. RAG retrieval is fast, interpretable, and grounds predictions in *actual historical market outcomes* rather than a language model's priors about markets.

- **FinBERT over generic sentiment**: Financial text is full of hedging and domain-specific language. Generic models trained on IMDB/Twitter score "record losses narrowed to $0.02/share" as negative — FinBERT correctly scores it neutral-to-positive in context.

- **PostgreSQL + ChromaDB instead of one store**: Structured data (prices, predictions, sentiment scores) needs relational queries and foreign key integrity — PostgreSQL is the right tool. Semantic similarity search needs a vector index — ChromaDB is the right tool. Unifying them in a single system would sacrifice correctness in at least one dimension.

- **Larger dataset path**: With a paid NewsAPI plan or GDELT (free, public), we could ingest 2+ years of article history per ticker. The pipeline and schema already support this — only the ingestion window changes. Walk-forward backtesting (train on years 1-2, test on year 3) would give a far more reliable AUC estimate.

---

## Limitations & Future Work

- **NewsAPI free tier** limits historical coverage to 30 days — GDELT or Alpaca News API would extend this significantly and produce a much larger labeled dataset.
- **Model trained on limited samples** (436 labeled examples after filtering) — performance estimates have high variance; expect AUC to stabilize with 5k+ samples.
- **Next step**: Add a scheduler (APScheduler or Celery) to run ingestion + inference daily automatically, closing the prediction loop in real-time.
- **Next step**: Fine-tune FinBERT on financial headlines (e.g., using FNSPID dataset) rather than using it zero-shot — this would improve sentiment accuracy on terse wire-service headlines.

---

## Project Structure

```
├── api/            FastAPI app, middleware, test script
├── data/           News + price ingestion pipeline
├── db/             SQLAlchemy models, session, Alembic migrations
├── features/       13-dim feature vector builder
├── model/          PyTorch architecture, training, evaluation
├── notebooks/      Exploratory analysis notebook
├── rag/            Chroma embedder, retriever, ingest script
├── config.py       Central config + logging setup
├── seed.py         Fast 3-ticker sanity check
└── Makefile        One-command workflow
```
