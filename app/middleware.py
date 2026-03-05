"""
Request logging middleware.
Logs every API request with: timestamp, method, endpoint, latency, status code.
Writes to both console and logs/api.log.
"""
import time
import logging
import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

os.makedirs("logs", exist_ok=True)

file_handler = logging.FileHandler("logs/api.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)

api_logger = logging.getLogger("ddi.requests")
api_logger.setLevel(logging.INFO)
api_logger.addHandler(file_handler)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()

        body_info = ""
        if request.method == "POST" and request.url.path in ("/predict", "/explain"):
            try:
                body = await request.body()
                request._body = body
                body_info = f" | body={body.decode('utf-8', errors='replace')[:200]}"
            except Exception:
                body_info = ""

        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000

        api_logger.info(
            f"{request.method} {request.url.path} | "
            f"status={response.status_code} | "
            f"latency={latency_ms:.1f}ms"
            f"{body_info}"
        )

        response.headers["X-Response-Time-ms"] = f"{latency_ms:.1f}"
        return response
