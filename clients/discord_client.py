import asyncio
import logging
import os
import uuid
from typing import Callable, Optional

import discord
from discord.ext import commands
from dotenv import load_dotenv

from core.bus import Bus, Interrupt, UserMessage

from .common import async_rate_limit, consume_tokens

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

client = None
_bus: Bus | None = None
user_id_mapping = {}
status_msg_ids: dict[str, str] = {}

STATUS_ROLES = {"tool", "error", "info"}
MAX_STATUS_LEN = 1900


@async_rate_limit(10, 10)
async def send_new_message(user_id: str, message: str) -> Optional[str]:
    if message == "":
        logger.warning("(discord) discarded empty message")
        return None
    global client, user_id_mapping
    channel_id, _ = user_id_mapping[user_id].split(":")
    channel = await client.fetch_channel(channel_id)
    try:
        sent = await channel.send(message)
        return str(sent.id)
    except Exception as e:
        logger.error(f"(discord) error sending message: {e}")
        return None


@async_rate_limit(10, 10)
async def edit_message(user_id: str, message_id: str, text: str) -> Optional[str]:
    global client
    channel_id = user_id_mapping[user_id].split(":")[0]
    channel = await client.fetch_channel(channel_id)
    try:
        original_message = await channel.fetch_message(int(message_id))
        await original_message.edit(content=text)
        return message_id
    except Exception as e:
        logger.error(f"(discord) error editing message: {e}")
        return None


async def post_status(user_id: str, text: str) -> None:
    """Update the rolling status message in place; create a new one if needed."""
    msg_id = status_msg_ids.get(user_id)
    if msg_id is not None:
        if await edit_message(user_id, msg_id, text):
            return
    # No existing status message (or edit failed): post a fresh one
    if len(text) > MAX_STATUS_LEN:
        text = text[-MAX_STATUS_LEN:]
    new_id = await send_new_message(user_id, text)
    if new_id:
        status_msg_ids[user_id] = new_id


def split_message(message: str, chunk_size: int = 1500) -> list[str]:
    """Split a message into chunks that fit within Discord's character limit."""
    chunks = []
    start = 0
    while start < len(message):
        end = min(start + chunk_size, len(message))
        chunks.append(message[start:end])
        start = end
    return chunks


async def handle_complete(user_id: str, role: str, text: str) -> None:
    """One complete (non-chunk) message from the bus."""
    if not text:
        return
    if role in STATUS_ROLES:
        line = text[:200] + ("..." if len(text) > 200 else "")
        await post_status(user_id, line)
    elif role == "assistant":
        for chunk in split_message(text, 1500):
            await send_new_message(user_id, chunk)
        status_msg_ids.pop(user_id, None)


async def handle_error(user_id: str, text: str) -> None:
    """Core failure: sent as a real message so it is not lost in the rolling status."""
    for chunk in split_message(f"ERROR: {text}", 1500):
        await send_new_message(user_id, chunk)


def get_unique_query_id(message):
    message_id = message.id
    channel_id = message.channel.id
    query_id = str(channel_id) + ":" + str(message_id)
    return query_id


class DiscordClient(commands.Bot):
    """Custom Bot class that handles events properly"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def on_ready(self):
        logger.info(f"Discord bot ready as {self.user}")

    async def on_message(self, message):
        if message.author == self.user:
            return

        logger.info(f"DM from {message.author}: {message.content}")
        user_id = str(message.author.id)
        user_id_mapping[user_id] = get_unique_query_id(message)
        if message.content.lower().strip() == "stop":
            _bus.post(Interrupt(user_id))
        else:
            _bus.post(UserMessage(user_id, message.content, uuid.uuid4().hex[:8]))


async def run(bus: Bus, context_size: int | Callable[[], int | None] = 0) -> None:
    """Run the Discord bot and the outbound consumer until cancelled."""
    global client, _bus
    _bus = bus

    # Disable privileged intents to avoid the PrivilegedIntentsRequired error
    intents = discord.Intents.default()
    intents.message_content = False  # Not needed for DMs
    intents.members = False  # Not used in this code

    client = DiscordClient(command_prefix="!", intents=intents)
    await asyncio.gather(
        client.start(os.getenv("DISCORD_TOKEN"), reconnect=True),
        consume_tokens(bus, handle_complete, handle_error),
    )


def stop() -> None:
    global client
    if client and not client.is_closed():
        if asyncio.get_event_loop().is_running():
            asyncio.run_coroutine_threadsafe(client.close(), asyncio.get_event_loop())
        else:
            asyncio.run(client.close())
