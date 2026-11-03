"""Shared helpers for client modules."""

import asyncio
import time

from bus import Bus, Error, Token


def rate_limit(calls, period=10):
    """Sync rate limiter (blocking sleep) for non-async contexts, e.g. the IRC reactor thread."""

    def decorator(func):
        timestamps = []

        def wrapper(*args, **kwargs):
            now = time.time()
            timestamps[:] = [t for t in timestamps if now - t < period]
            if len(timestamps) >= calls:
                time.sleep(period - (now - timestamps[0]))
            timestamps.append(time.time())
            return func(*args, **kwargs)

        return wrapper

    return decorator


def async_rate_limit(calls, period=10):
    """Async rate limiter for event-loop code."""

    def decorator(func):
        timestamps = []

        async def wrapper(*args, **kwargs):
            now = time.monotonic()
            timestamps[:] = [t for t in timestamps if now - t < period]
            if len(timestamps) >= calls:
                await asyncio.sleep(period - (now - timestamps[0]))
            timestamps.append(time.monotonic())
            return await func(*args, **kwargs)

        return wrapper

    return decorator


async def consume_tokens(bus: Bus, on_complete, on_error=None) -> None:
    """Accumulate streamed Token chunks per user.

    Calls ``await on_complete(user_id, role, text)`` for every complete
    (chunk=False) message, including empty ones. Thinking tokens and
    non-Token events are ignored, except Error events which go to on_error
    when provided.
    """
    pending: dict[str, str] = {}
    while True:
        event = await bus.outbound.get()
        if isinstance(event, Error):
            if on_error is not None:
                await on_error(event.user_id, event.text)
            continue
        if not isinstance(event, Token) or event.role == "thinking":
            continue
        if event.chunk:
            pending[event.user_id] = pending.get(event.user_id, "") + event.text
        else:
            await on_complete(event.user_id, event.role, pending.pop(event.user_id, "") + event.text)
