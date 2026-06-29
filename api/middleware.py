"""Request logging middleware and in-memory request counter."""
import logging
import time
from datetime import datetime

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger(__name__)

request_count: int = 0


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status code, and duration for every request."""

    async def dispatch(self, request: Request, call_next):
        """Intercept the request, time it, log the result, and increment the counter."""
        global request_count
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        request_count += 1
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            "%s | %s %s | %d | %.1fms",
            ts, request.method, request.url.path, response.status_code, duration_ms,
        )
        return response
