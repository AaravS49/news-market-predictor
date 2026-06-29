"""Central configuration — loads env vars and defines project-wide constants."""
import logging
import os
from dotenv import load_dotenv

load_dotenv()

NEWSAPI_KEY: str = os.environ["NEWSAPI_KEY"]
DATABASE_URL: str = os.environ["DATABASE_URL"]

TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "META", "GOOGL",   # Tech
    "JPM",  "BAC",  "GS",   "MS",   "WFC",      # Finance
    "JNJ",  "UNH",  "PFE",  "ABBV", "MRK",      # Healthcare
    "XOM",  "CVX",  "COP",  "SLB",  "EOG",      # Energy
    "AMZN", "TSLA", "HD",   "MCD",  "NKE",      # Consumer
]

SECTOR_MAP: dict[str, str] = {
    "AAPL": "Tech",       "MSFT": "Tech",       "NVDA": "Tech",
    "META": "Tech",       "GOOGL": "Tech",
    "JPM":  "Finance",    "BAC":  "Finance",     "GS":   "Finance",
    "MS":   "Finance",    "WFC":  "Finance",
    "JNJ":  "Healthcare", "UNH":  "Healthcare",  "PFE":  "Healthcare",
    "ABBV": "Healthcare", "MRK":  "Healthcare",
    "XOM":  "Energy",     "CVX":  "Energy",      "COP":  "Energy",
    "SLB":  "Energy",     "EOG":  "Energy",
    "AMZN": "Consumer",   "TSLA": "Consumer",    "HD":   "Consumer",
    "MCD":  "Consumer",   "NKE":  "Consumer",
}

PREDICTION_HORIZON_DAYS: int = 3
RAG_TOP_K: int = 5
MODEL_VERSION: str = "v1"

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with the project-wide format."""
    logging.basicConfig(format=LOG_FORMAT, level=level, force=True)
