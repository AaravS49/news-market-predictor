"""SQLAlchemy ORM models for all structured data in the news-market-predictor."""
from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    headline = Column(String, nullable=False)
    body = Column(String)
    url = Column(String)
    source = Column(String)
    published_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, ticker={self.ticker!r}, headline={self.headline[:60]!r})>"


class PriceRecord(Base):
    __tablename__ = "price_records"

    id = Column(Integer, primary_key=True)
    ticker = Column(String, nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    close = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Float)
    return_3d = Column(Float, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<PriceRecord(id={self.id}, ticker={self.ticker!r}, "
            f"date={self.date}, return_3d={self.return_3d})>"
        )


class SentimentScore(Base):
    __tablename__ = "sentiment_scores"

    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
    score = Column(Float)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<SentimentScore(id={self.id}, article_id={self.article_id}, score={self.score})>"


class RAGFeature(Base):
    __tablename__ = "rag_features"

    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
    mean_analog_return = Column(Float)
    analog_hit_rate = Column(Float)
    analog_volatility = Column(Float)
    n_analogs = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<RAGFeature(id={self.id}, article_id={self.article_id}, "
            f"mean={self.mean_analog_return})>"
        )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey("articles.id"), nullable=False)
    predicted_prob = Column(Float)
    predicted_label = Column(Integer)
    actual_label = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, article_id={self.article_id}, "
            f"predicted_prob={self.predicted_prob}, predicted_label={self.predicted_label})>"
        )
