import asyncio
import functools
import logging
import ssl
import sys
import uuid
from typing import Callable

import irc.client

from core.bus import Bus, UserMessage

from .common import consume_tokens, rate_limit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

server_connected = False
reactor: irc.client.Reactor | None = None
connection = None
_bus: Bus | None = None


def on_connect(connection, event):
    global server_connected
    server_connected = True


def on_join(connection, event):
    logger.info(f"joined channel: {event.target}")


@rate_limit(10, 10)
def send_to_irc_rate_limited(connection, target, line):
    connection.privmsg(target, line)


def _send_lines(connection, user_id, text):
    """Runs on the reactor thread; sends one complete message, chunked to IRC limits."""
    for line in text.split("\n"):
        if len(line.encode("utf-8")) <= 400:
            send_to_irc_rate_limited(connection, user_id, line)
        else:
            bytes_line = line.encode("utf-8")
            pos = 0
            while pos < len(bytes_line):
                chunk = bytes_line[pos : pos + 400].decode("utf-8")
                send_to_irc_rate_limited(connection, user_id, chunk)
                pos += 400


async def handle_complete(user_id: str, role: str, text: str) -> None:
    """One complete (non-chunk) message from the bus.

    Sends are handed to the reactor thread (one-shot execute_after) because
    irc.client socket writes are not thread-safe.
    """
    if not server_connected:
        return
    logger.info(f"Sending to {user_id}: {text[:50]}...")
    reactor.scheduler.execute_after(0, functools.partial(_send_lines, connection, user_id, text))


async def handle_error(user_id: str, text: str) -> None:
    """Core failure: prefixed so it stands out in the plain IRC stream."""
    await handle_complete(user_id, "error", f"ERROR: {text}")


def on_disconnect(connection, event):
    logger.info("disconnected")


def on_privmsg(connection, event):
    """Handle incoming private messages with unique user ID (nick)"""
    user_id = event.source.nick  # This is the unique identifier for the user
    message = event.arguments[0]

    logger.info(f"DM from {user_id}: {message}")
    _bus.post(UserMessage(user_id, message, uuid.uuid4().hex[:8]))


async def run(bus: Bus, context_size: int | Callable[[], int | None] = 0) -> None:
    """Connect to IRC and consume outbound events until cancelled."""
    global reactor, connection, _bus
    _bus = bus

    context = ssl.create_default_context()
    wrapper = functools.partial(context.wrap_socket, server_hostname="irc.libera.chat")
    ssl_factory = irc.connection.Factory(wrapper=wrapper)
    reactor = irc.client.Reactor()
    try:
        connection = reactor.server().connect("irc.libera.chat", 6697, "sdfdsf333", connect_factory=ssl_factory)
    except irc.client.ServerConnectionError:
        logger.error(sys.exc_info()[1])
        raise SystemExit(1) from None

    connection.add_global_handler("welcome", on_connect)
    connection.add_global_handler("join", on_join)
    connection.add_global_handler("disconnect", on_disconnect)
    connection.add_global_handler("privmsg", on_privmsg)
    connection.add_global_handler("pubmsg", on_privmsg)

    await asyncio.gather(
        consume_tokens(bus, handle_complete, handle_error),
        asyncio.get_event_loop().run_in_executor(None, reactor.process_forever),
    )


def stop() -> None:
    pass
