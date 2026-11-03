"""Typed event bus for client <-> core communication.

Inbound events flow client -> core; outbound events flow core -> clients.
`Bus.post()` is thread-safe (for use from e.g. the IRC reactor thread).
Call `Bus.bind()` once from within the running event loop before posting.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class UserMessage:
    user_id: str
    text: str
    msg_id: str


@dataclass(frozen=True)
class Interrupt:
    user_id: str


@dataclass(frozen=True)
class Shutdown:
    pass


@dataclass(frozen=True)
class Token:
    user_id: str
    role: str
    text: str
    chunk: bool = True


@dataclass(frozen=True)
class TurnDone:
    msg_id: str


@dataclass(frozen=True)
class UsageReport:
    user_id: str
    usage: dict


Event = UserMessage | Interrupt | Shutdown | Token | TurnDone | UsageReport


class Bus:
    def __init__(self) -> None:
        self.inbound = asyncio.Queue()
        self.outbound = asyncio.Queue()
        # Set directly on post (not via the inbound queue) so the tool loop can
        # observe an interrupt mid-turn; the main loop is busy inside
        # handle_message() and cannot drain the queue until the turn ends.
        self.interrupted: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self) -> None:
        """Capture the running event loop; call once from within it."""
        self._loop = asyncio.get_running_loop()

    def post(self, event: Event) -> None:
        """Post an inbound event, safely callable from any thread."""
        if isinstance(event, Interrupt):
            self.interrupted.add(event.user_id)
        if self._loop is None:
            raise RuntimeError("Bus.bind() must be called before post()")
        self._loop.call_soon_threadsafe(self.inbound.put_nowait, event)


class Client(Protocol):
    """Contract every client module fulfils; replaces the old global injection."""

    async def run(self, bus: Bus, context_size: int = 0) -> None: ...

    def stop(self) -> None: ...
