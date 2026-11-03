"""Reusable Fibonacci-backoff retry decorator for HTTP request functions."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger()

T = TypeVar("T")


class HttpError(Exception):
    """HTTP failure carrying a status code; raise this for non-200 responses."""

    def __init__(self, status: int, detail: str = ""):
        self.status = status
        super().__init__(f"HTTP {status}" + (f": {detail}" if detail else ""))


def _is_client_error(e: BaseException) -> bool:
    """True for 4xx responses (urllib HTTPError.code or HttpError.status); those are not retried."""
    for attr in ("status", "code"):
        value = getattr(e, attr, None)
        if isinstance(value, int):
            return 400 <= value < 500
    return False


def _fibonacci_delays(max_retries: int, base_delay: float) -> list[float]:
    a, b = base_delay, base_delay
    delays = []
    for _ in range(max_retries):
        delays.append(a)
        a, b = b, a + b
    return delays


def retry_fibonacci(max_retries: int = 10, base_delay: float = 1.0):
    """Decorator: retry a function on every exception, waiting Fibonacci seconds (1, 1, 2, 3, ...) between attempts.

    Apply only to functions that are a single HTTP request; do not wrap tool execution or
    orchestration. 4xx responses (urllib HTTPError / HttpError with a 400-499 status) are
    definitive client errors and raise immediately without retrying. After max_retries failed
    retries the original exception is raised as-is so the caller's error funnel (e.g.
    main.report_error) converts it into a reportable error. Works with both sync and async.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        delays = _fibonacci_delays(max_retries, base_delay)

        @functools.wraps(func)
        async def awrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if _is_client_error(e):
                        # 4xx is a definitive client error (bad request, not found); retrying won't help.
                        raise
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e!r}")
                        raise
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed ({e!r}); retrying in {delays[attempt]:g}s"
                    )
                    await asyncio.sleep(delays[attempt])

        @functools.wraps(func)
        def swrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if _is_client_error(e):
                        # 4xx is a definitive client error (bad request, not found); retrying won't help.
                        raise
                    if attempt == max_retries:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e!r}")
                        raise
                    logger.warning(
                        f"{func.__name__} attempt {attempt + 1} failed ({e!r}); retrying in {delays[attempt]:g}s"
                    )
                    time.sleep(delays[attempt])

        return awrapper if inspect.iscoroutinefunction(func) else swrapper

    return decorator
