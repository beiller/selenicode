#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import base64
import importlib
import inspect
import json
import logging
import random
import string
import sys
import threading
import types
import urllib.request
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import AsyncIterator, Callable, Union, get_type_hints

import aiohttp

from core import config

LOG_FILE = config.DATA_DIR / "agent1.log"

# File-only logging: a console handler would write raw text over the terminal
# client's screen. Everything lands in agent1.log (rotating).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)],
)
logger = logging.getLogger()


# Import tools from tools.py
from core.retry import HttpError, retry_fibonacci
from core.tools import (
    change_working_directory,
    conversation_search,
    edit_file,
    get_loaded_model,
    list_models,
    load_model,
    load_skills,
    make_tool,
    read_file,
    run_bash,
    unload_model,
    write_file,
)

CURRENT_MODEL = config.CURRENT_MODEL
# Context size is mandatory and only ever taken from the server; None until resolved.
CONTEXT_SIZE: int | None = None
ctx_ready = asyncio.Event()
ctx_task: asyncio.Task | None = None
shutdown = asyncio.Event()

SLASH_COMMANDS = [
    ("/compact", "summarize older conversation to free context space"),
    ("/server <url> [key]", "switch to a different model server (eg http(s)://localhost:8080, optional API key)"),
    ("/help", "show this list"),
]


async def _run_daemon(fn, *args):
    """Run blocking fn in a daemon thread; cancelling the awaiter never waits for the work to finish."""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def runner():
        try:
            result = fn(*args)
        except BaseException as e:
            exc, res = e, None
        else:
            exc, res = None, result
        if fut.done():
            return
        try:
            if exc is not None:
                loop.call_soon_threadsafe(fut.set_exception, exc)
            else:
                loop.call_soon_threadsafe(fut.set_result, res)
        except Exception:
            pass  # event loop already closed (awaiter was cancelled and moved on)

    threading.Thread(target=runner, daemon=True).start()
    return await fut


def load_model_and_set(model_name: str):
    global CURRENT_MODEL
    loaded_model = get_loaded_model()
    if loaded_model:
        unload_model(loaded_model)
    load_model(model_name)
    CURRENT_MODEL = model_name


def _fetch_models_once(base_url: str) -> list[dict]:
    """Single-attempt fetch of the model list (the retry loop lives in wait_for_context_size)."""
    req = urllib.request.Request(f"{base_url}/v1/models", method="GET", headers=config.auth_headers())
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8")).get("data", [])


def _parse_context_size(models: list[dict]) -> int | None:
    """Extract the context size from a /v1/models response (llama.cpp meta.n_ctx or vLLM max_model_len)."""
    for preferred in (True, False):
        for model in models:
            if preferred and model.get("id") != CURRENT_MODEL:
                continue
            n_ctx = (model.get("meta") or {}).get("n_ctx", 0) or model.get("max_model_len", 0)
            if n_ctx > 0:
                return int(n_ctx)
    return None


def _parse_vision_supported(models: list[dict]) -> bool:
    """True when the model serves vision (llama-server /v1/models).

    Prefers an exact id match for CURRENT_MODEL, falls back to any loaded model
    (the server may report a different id than the configured name). Handles both
    capability shapes seen in the wild: a list containing "multimodal" and a dict
    with an "image" flag."""
    for preferred in (True, False):
        for model in models:
            if preferred and model.get("id") != CURRENT_MODEL:
                continue
            caps = model.get("capabilities") or []
            if isinstance(caps, dict):
                return bool(caps.get("image"))
            if "multimodal" in caps:
                return True
    return False


async def wait_for_context_size() -> None:
    """Keep trying to get the context size from the server until it succeeds; never gives up.

    Every failed attempt surfaces an error to the user with the next retry delay (capped Fibonacci)."""
    global CONTEXT_SIZE
    bus.outbound.put_nowait(ServerStatus(False))
    attempt = 0
    delay_a, delay_b = 1.0, 1.0
    while CONTEXT_SIZE is None:
        if shutdown.is_set():
            return
        attempt += 1
        try:
            if not config.BASE_URL:
                raise RuntimeError("no server URL configured (set 'server.url' in config.json)")
            models = await _run_daemon(_fetch_models_once, config.BASE_URL)
            n_ctx = _parse_context_size(models)
            if n_ctx is None:
                raise RuntimeError(f"No context size in model list for {CURRENT_MODEL}")
            CONTEXT_SIZE = n_ctx
            config.VISION_SUPPORTED = _parse_vision_supported(models)
        except Exception as e:
            logger.warning(f"Context size check failed (attempt {attempt}): {e!r}; retrying in {delay_a:.0f}s")
            report_error("", f"Model server connection failed (attempt {attempt}): {e} - retrying in {delay_a:.0f}s")
            await asyncio.sleep(delay_a)
            delay_a, delay_b = min(delay_b, 60.0), min(delay_a + delay_b, 60.0)
            continue
        trigger = compaction_trigger(CONTEXT_SIZE)
        target = compaction_target(CONTEXT_SIZE)
        logger.info(f"Context size: {CONTEXT_SIZE}; auto-trigger at {trigger} tokens, shrink-to target {target} tokens")
        if target >= trigger:
            logger.warning("Shrink-to target is not below the auto-trigger; auto-compaction will be a no-op")
        bus.outbound.put_nowait(ServerStatus(True))
        ctx_ready.set()


def _is_unsupported_endpoint(e: BaseException) -> bool:
    """True when a 404/405 means the server lacks the model-management endpoints (not router mode)."""
    for attr in ("status", "code"):
        value = getattr(e, attr, None)
        if isinstance(value, int):
            return value in (404, 405)
    return False


async def startup_load_model() -> None:
    """Load the model in the background; management endpoints may not exist (e.g. vLLM)."""
    if not CURRENT_MODEL:
        logger.info("No model configured; skipping startup load")
        return
    logger.info(f"Loading model {CURRENT_MODEL}")
    try:
        await _run_daemon(load_model_and_set, CURRENT_MODEL)
    except Exception as e:
        if not _is_unsupported_endpoint(e):
            logger.error(f"Model load at startup failed: {e!r}; continuing")
            return
        # The server is not in router mode; it serves whatever model it was started with.
        try:
            loaded = await _run_daemon(get_loaded_model)
        except Exception:
            loaded = None
        if loaded:
            logger.warning(f"Server does not support dynamic model loading; continuing with preloaded model {loaded!r}")
        else:
            logger.error(f"Model load at startup failed: {e!r}; no model is loaded and the server cannot load one")


async def handle_server_command(user_id: UserID, msg_id: str, text: str) -> None:
    """`/server <url> [api_key]`: point this program at a different model server.

    The argument is the full base URL (scheme + host + port), used verbatim;
    the optional api key replaces any configured one (a bare URL clears it).
    Cancels any in-flight fetcher (its stale attempt is discarded by _run_daemon)
    and starts a fresh one, so the context-size check begins immediately against
    the new URL with the usual retry error messages. Resets the context-size gate
    so the next turn blocks until the new server reports its size."""
    global CONTEXT_SIZE, ctx_task
    parts = text.split()
    if len(parts) not in (2, 3):
        report_error(user_id, "Usage: /server <url> [api_key] (eg http(s)://localhost:8080)")
        bus.outbound.put_nowait(TurnDone(msg_id))
        return
    url = parts[1].strip()
    host = url.split("://", 1)[1] if url.startswith(("http://", "https://")) else ""
    if not host:
        report_error(user_id, f"Invalid server URL (need full URL with http/https scheme): {url}")
        bus.outbound.put_nowait(TurnDone(msg_id))
        return
    config.BASE_URL = url
    config.API_KEY = parts[2] if len(parts) == 3 else ""
    if ctx_task is not None and not ctx_task.done():
        ctx_task.cancel()
    CONTEXT_SIZE = None
    ctx_ready.clear()
    ctx_task = asyncio.create_task(wait_for_context_size())
    logger.info(f"/server: switched to {url} (from {user_id})")
    bus.outbound.put_nowait(TurnDone(msg_id))


# ---------------------------------------------------------------------------
# Types – Chat Completions API
# ---------------------------------------------------------------------------

from core.bus import (
    Bus,
    CompactionDone,
    Error,
    History,
    Interrupt,
    ServerStatus,
    Shutdown,
    Token,
    TurnDone,
    UsageReport,
    UserMessage,
)
from core.compaction import compact_messages, compaction_target, compaction_trigger, context_token_count
from core.conversation import (
    approximate_token_count,
    archive_conversation,
    get_last_conversation_session_id,
    read_conversation,
    write_conversation,
)
from core.main_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    Message,
    SessionID,
    Tool,
    ToolCallRef,
    ToolHandler,
    Usage,
    UserID,
)

# ---------------------------------------------------------------------------
# Pure constructors
# ---------------------------------------------------------------------------

_TYPE_MAP = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _resolve_type(t):
    """Collapse Optional/union annotations (e.g. `int | None`) to their non-None member."""
    origin = getattr(t, "__origin__", None)
    if origin in (Union, types.UnionType):
        members = [m for m in t.__args__ if m is not type(None)]
        return members[0] if len(members) == 1 else str
    return t


def tool_from_function(fn: ToolHandler) -> Tool:
    """Build a Tool from a function's signature and docstring."""
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    properties = {}
    required = []
    for name, p in sig.parameters.items():
        properties[name] = {"type": _TYPE_MAP.get(_resolve_type(hints.get(name, str)), "string")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    return make_tool(
        name=fn.__name__,
        description=inspect.getdoc(fn) or "",
        parameters={"type": "object", "properties": properties, "required": required},
    )


def make_user_message(content: str) -> Message:
    return {"role": "user", "content": content}


def make_system_message(content: str) -> Message:
    return {"role": "system", "content": content}


def make_tool_result_message(tool_call_id: str, content: str | list) -> Message:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _tool_output_to_content(output: str) -> str | list:
    """Turn a read_file image marker into multimodal message parts; any other output passes through."""
    try:
        parsed = json.loads(output)
    except (ValueError, TypeError):
        return output
    if isinstance(parsed, dict) and parsed.get("type") == "image" and "path" in parsed and "mime" in parsed:
        return [
            {"type": "text", "text": f"Image loaded from {parsed['path']} ({parsed['mime']})"},
            {"type": "image_ref", "path": parsed["path"], "mime": parsed["mime"]},
        ]
    return output


def _encode_image_ref(part: dict) -> dict:
    """Resolve a stored image_ref part into an OpenAI data-URL part (or a text fallback)."""
    path = part.get("path", "")
    mime = part.get("mime", "image/png")
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return {"type": "text", "text": f"[image unavailable: {path}]"}
    b64 = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


MEDIA_MARKER = "<__media__>"
MEDIA_MARKER_REPLACEMENT = "[media]"


def _scrub_media_marker(text: str) -> str:
    """Replace the marker the server reserves for media parts so text never carries it."""
    return text.replace(MEDIA_MARKER, MEDIA_MARKER_REPLACEMENT) if MEDIA_MARKER in text else text


def _attach_images(content: list) -> list:
    """Encode image_ref parts for the API on a copy; history keeps the compact path form.

    Also scrubs the reserved media marker from text parts (the server 400s a request
    that carries media when any message text contains the marker)."""
    out = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_ref":
            out.append(_encode_image_ref(part))
        elif isinstance(part, dict):
            p = dict(part)
            if isinstance(p.get("text"), str):
                p["text"] = _scrub_media_marker(p["text"])
            out.append(p)
        else:
            out.append(part)
    return out


def _scrub_obj(obj):
    """Recursively replace the reserved marker in every string (on a copy; marker-free values pass through).

    Applied to the whole prepared message so no field can carry the marker (content,
    reasoning_content, tool_calls arguments, ...); the base64 in encoded image parts
    cannot contain it, so data URLs are untouched."""
    if isinstance(obj, str):
        return _scrub_media_marker(obj)
    if isinstance(obj, dict):
        return {k: _scrub_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_obj(v) for v in obj]
    return obj


def _prepare_message(m: Message) -> Message:
    """Strip timestamps; encode image_ref parts and scrub the reserved media marker (on a copy)."""
    out = {k: v for k, v in m.items() if k != "timestamp"}
    content = out.get("content")
    if isinstance(content, list):
        out["content"] = _attach_images(content)
    return _scrub_obj(out)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = config.ROOT_DIR
SYSTEM_MD = ROOT_DIR / "SYSTEM.md"
SKILLS_DIR = ROOT_DIR / "skills"
LOCAL_SKILLS_DIR = ROOT_DIR / "local_skills"


# ---------------------------------------------------------------------------
# HTTP transport
# ---------------------------------------------------------------------------


async def stream_chat_completion(
    base_url: str, messages: list[Message], tools: list[Tool], session_id: SessionID, user_id: UserID
) -> AsyncIterator[ChatCompletionChunk]:
    """Send a streaming chat completion request, yielding SSE chunks asynchronously."""
    system_content = SYSTEM_MD.read_text() if SYSTEM_MD.exists() else ""

    # Add dynamic context to system prompt (keep stable across calls so the server's prompt cache can be reused)
    session_context = f"\n\nActive session: {session_id}"
    system_content = _scrub_media_marker(system_content + session_context)

    msgs = [make_system_message(system_content)] + [_prepare_message(m) for m in messages if m.get("role") != "system"]
    body: ChatCompletionRequest = {
        "model": CURRENT_MODEL,
        "messages": msgs,
        "temperature": 0.7,
        "top_k": 20,
        "top_p": 0.8,
        "repeat_penalty": 1.0,
        "presence_penalty": 1.5,
        "tools": tools,
        "tool_choice": "auto",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # Only the request open is retried; a mid-stream failure propagates (partial output already emitted).
    session, resp = await _open_chat_stream(base_url, body)
    try:
        async for line in resp.content:
            raw_line = line.decode("utf-8").strip()
            if not raw_line or raw_line.startswith(":"):
                continue
            if raw_line == "data: [DONE]":
                break
            if raw_line.startswith("data: "):
                yield json.loads(raw_line[6:])
    finally:
        await session.close()


@retry_fibonacci()
async def _open_chat_stream(
    base_url: str, body: ChatCompletionRequest
) -> tuple[aiohttp.ClientSession, aiohttp.ClientResponse]:
    """Open the streaming chat completion request (one HTTP request, retried on failure)."""
    logger.debug(f"loading: {base_url}/v1/chat/completions")
    session = aiohttp.ClientSession()
    try:
        resp = await session.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json", **config.auth_headers()},
            timeout=aiohttp.ClientTimeout(sock_read=None, sock_connect=5 * 60, connect=5 * 60),
        )
        if resp.status != 200:
            error_body = (await resp.text())[:300]
            await session.close()
            raise HttpError(resp.status, error_body)
        return session, resp
    except BaseException:
        await session.close()
        raise


async def consume_stream(
    chunks: AsyncIterator[ChatCompletionChunk],
    user_id: UserID,
    emit_fn: Callable[[UserID, str, str, bool], None] | None = None,
    interrupted_check: Callable[[], bool] | None = None,
) -> tuple[Message, Usage | None]:
    """Consume streamed chunks, emit tokens, and return the assembled Message and usage.

    If interrupted_check() becomes true mid-stream, stop consuming and return the
    partial message assembled so far (its tool_calls must not be executed)."""
    role = "assistant"
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_accum: dict[int, ToolCallRef] = {}
    usage: Usage | None = None

    async for chunk in chunks:
        if interrupted_check is not None and interrupted_check():
            break
        if chunk.get("usage"):
            usage = chunk["usage"]
        if not chunk.get("choices"):
            continue
        delta = chunk["choices"][0]["delta"]

        if "role" in delta:
            role = delta["role"]

        text = delta.get("content")
        if text:
            content_parts.append(text)
            if emit_fn:
                emit_fn(user_id, role, text, True)
            # hack?
            await asyncio.sleep(0)

        reasoning = delta.get("reasoning_content")
        if reasoning:
            reasoning_parts.append(reasoning)
            if emit_fn:
                emit_fn(user_id, "thinking", reasoning, True)
            await asyncio.sleep(0)

        for dtc in delta.get("tool_calls", []):
            idx = dtc["index"]
            if idx not in tool_calls_accum:
                tool_calls_accum[idx] = {
                    "id": dtc.get("id", ""),
                    "type": dtc.get("type", "function"),
                    "function": {
                        "name": dtc.get("function", {}).get("name", ""),
                        "arguments": dtc.get("function", {}).get("arguments", ""),
                    },
                }
            else:
                entry = tool_calls_accum[idx]
                if "id" in dtc and dtc["id"]:
                    entry["id"] = dtc["id"]
                fn = dtc.get("function", {})
                if fn.get("name"):
                    entry["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    entry["function"]["arguments"] += fn["arguments"]

    msg: Message = {"role": role}
    content = "".join(content_parts)
    if content:
        msg["content"] = content
    else:
        msg["content"] = None
    reasoning = "".join(reasoning_parts)
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tool_calls_accum:
        msg["tool_calls"] = [tool_calls_accum[i] for i in sorted(tool_calls_accum)]
    return msg, usage


# ---------------------------------------------------------------------------
# Tool-call processing
# ---------------------------------------------------------------------------


def emit(user_id: UserID, role: str, content: str, chunk: bool = False) -> None:
    bus.outbound.put_nowait(Token(user_id, role, content, chunk))


def report_error(user_id: UserID, message: str, exc: BaseException | None = None) -> None:
    """Single funnel for core failures: log (with traceback when exc is given) and surface to the client."""
    if exc is not None:
        logger.error(f"{message}: {exc!r}", exc_info=exc)
        bus.outbound.put_nowait(Error(user_id, f"{message}: {exc}"))
    else:
        logger.error(message)
        bus.outbound.put_nowait(Error(user_id, message))


async def execute_tool_calls(
    assistant_msg: Message, registry: dict[str, ToolHandler], user_id: UserID
) -> list[Message]:
    """Return the assistant message + one tool-result message per call."""
    tool_calls: list[ToolCallRef] = assistant_msg.get("tool_calls", [])
    if not tool_calls:
        return []

    results: list[Message] = [assistant_msg]
    for tc in tool_calls:
        name = tc["function"]["name"]
        args: dict[str, str] = json.loads(tc["function"]["arguments"])
        handler = registry.get(name)
        if handler is None:
            output = json.dumps({"error": f"unknown tool: {name}"})
            report_error(user_id, f"Unknown tool: {name}")
        else:
            args_short = ", ".join(f"{k}={v!r}" for k, v in args.items())
            if len(args_short) > 200:
                args_short = args_short[0:200] + "..."
            emit(user_id, "tool", f"{name}({args_short})")
            try:
                if inspect.iscoroutinefunction(handler):
                    output = await handler(**args)
                else:
                    # Offload blocking tools to a worker thread so the event loop
                    # (UI rendering, interrupt polling, other clients) keeps running.
                    output = await asyncio.to_thread(handler, **args)
            except Exception as e:
                report_error(user_id, f"Tool {name} failed", e)
                output = json.dumps({"error": f"Tool {name} raised an exception: {e}"})
        results.append(make_tool_result_message(tc["id"], _tool_output_to_content(output)))
    return results


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------


def build_tools() -> tuple[list[Tool], dict[str, ToolHandler]]:
    """Build the full tools list and registry by reloading skills from disk."""
    tools: list[Tool] = [
        tool_from_function(conversation_search),
        tool_from_function(run_bash),
        tool_from_function(load_model),
        tool_from_function(write_file),
        tool_from_function(edit_file),
        tool_from_function(read_file),
        tool_from_function(list_models),
        tool_from_function(reset_session),
        tool_from_function(compact_session),
        tool_from_function(change_working_directory),
    ]
    registry: dict[str, ToolHandler] = {
        "conversation_search": conversation_search,
        "write_file": write_file,
        "edit_file": edit_file,
        "read_file": read_file,
        "run_bash": run_bash,
        "change_working_directory": change_working_directory,
        "load_model": load_model_and_set,
        "list_models": list_models,
        "reset_session": reset_session,
        "compact_session": compact_session,
    }
    skill_tools, skill_registry = load_skills(SKILLS_DIR)
    tools.extend(skill_tools)
    registry.update(skill_registry)

    local_skill_tools, local_skill_registry = load_skills(LOCAL_SKILLS_DIR)
    tools.extend(local_skill_tools)
    registry.update(local_skill_registry)

    return tools, registry


_tool_cache: tuple[list[Tool], dict[str, ToolHandler]] | None = None
_skill_mtimes: tuple[float, ...] = ()


def _skills_signature() -> tuple[float, ...]:
    sig: list[float] = []
    for d in (SKILLS_DIR, LOCAL_SKILLS_DIR):
        if d.is_dir():
            sig.extend(f.stat().st_mtime for f in sorted(d.glob("*.md")))
    return tuple(sig)


def get_tools() -> tuple[list[Tool], dict[str, ToolHandler]]:
    """Cached build_tools(); rebuilds only when skill files change on disk."""
    global _tool_cache, _skill_mtimes
    sig = _skills_signature()
    if _tool_cache is None or sig != _skill_mtimes:
        _tool_cache = build_tools()
        _skill_mtimes = sig
    return _tool_cache


def _finalize_assistant(msg: Message, messages: list[Message]) -> str:
    """Strip tool_calls, stamp and append the final assistant message; return its text."""
    final = {k: v for k, v in msg.items() if k != "tool_calls"}
    final["content"] = msg.get("content") or ""
    final["timestamp"] = datetime.now().isoformat()
    messages.append(final)
    return final["content"]


async def _wait_interrupt(user_id: UserID) -> None:
    """Resolve once the user's interrupt flag is set (polled; chunks may be stalled)."""
    while user_id not in bus.interrupted:
        await asyncio.sleep(0.1)


async def run_tool_loop(base_url: str, user_id: UserID, messages: list[Message], session_id: SessionID) -> str:
    """Stream messages, handle tool calls in a loop, return final text."""
    max_calls = 200
    tool_counter = 0

    while True:
        if user_id in bus.interrupted:
            bus.interrupted.discard(user_id)
            emit(user_id, "info", "Interrupted. What can I help you with?")
            return ""

        size = context_token_count(messages)
        if size > compaction_trigger(CONTEXT_SIZE):
            logger.info(f"Context {size} tokens exceeded trigger mid-loop, compacting")
            await compact_messages(base_url, messages, session_id, CONTEXT_SIZE, CURRENT_MODEL)
            size = context_token_count(messages)
            if size > CONTEXT_SIZE:
                report_error(user_id, f"Compaction insufficient mid-loop: {size} tokens > {CONTEXT_SIZE}, stopping turn")
                return ""
            bus.outbound.put_nowait(CompactionDone(user_id, size))

        tools, registry = get_tools()
        chunks = stream_chat_completion(base_url, messages, tools, session_id, user_id)
        # Consume in a task so an interrupt can cancel a stalled stream (no chunks
        # flowing); when chunks are flowing, the per-chunk check breaks gracefully
        # and the partial message is kept.
        consume_task = asyncio.create_task(consume_stream(chunks, user_id, emit, lambda: user_id in bus.interrupted))
        interrupt_waiter = asyncio.create_task(_wait_interrupt(user_id))
        done, pending = await asyncio.wait({consume_task, interrupt_waiter}, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        if user_id in bus.interrupted:
            # Interrupted mid-stream: end the turn without executing tools.
            bus.interrupted.discard(user_id)
            emit(user_id, "assistant", "", False)
            if consume_task in done and not consume_task.cancelled():
                assistant_msg, _ = consume_task.result()
                _finalize_assistant(assistant_msg, messages)  # partial kept, tool_calls stripped
            await chunks.aclose()  # close the HTTP stream (no-op if already finished)
            emit(user_id, "info", "Interrupted. What can I help you with?")
            return ""

        assistant_msg, usage = consume_task.result()
        if usage is not None:
            bus.outbound.put_nowait(UsageReport(user_id, usage))
        emit(user_id, "assistant", "", False)  # send a final empty message so the client knows the end of message.
        tool_msgs = await execute_tool_calls(assistant_msg, registry, user_id)

        if not tool_msgs:
            return _finalize_assistant(assistant_msg, messages)

        for tm in tool_msgs:
            stamped = tm.copy()
            stamped["timestamp"] = datetime.now().isoformat()
            messages.append(stamped)

        tool_counter += 1
        if tool_counter > max_calls:
            report_error(user_id, f"Too many tool calls ({max_calls}), giving up on this turn")
            return _finalize_assistant(assistant_msg, messages)


async def handle_message(user_input: str, base_url: str, user_id: UserID, session_id: SessionID, msg_id: str) -> None:
    """Handle a user message event: append it, run the tool loop, emit the reply."""
    compacted_size: int | None = None
    try:
        messages = get_user_messages(user_id)
        # Add timestamp when user types the message
        user_msg = make_user_message(user_input)
        user_msg["timestamp"] = datetime.now().isoformat()
        messages.append(user_msg)
        write_conversation(session_id, messages)

        await run_tool_loop(base_url, user_id, messages, session_id)  # modifies messages

        if context_token_count(messages) > compaction_trigger(CONTEXT_SIZE):
            logger.info("Compacting conversation")
            compacted = await compact_messages(base_url, messages, session_id, CONTEXT_SIZE, CURRENT_MODEL)
            size = approximate_token_count(json.dumps(messages))
            if not compacted and size > compaction_trigger(CONTEXT_SIZE):
                report_error(user_id, f"Compaction failed; context still {size} tokens")
            else:
                logger.info(f"Context size is {size} of {CONTEXT_SIZE}")
                compacted_size = size

        write_conversation(session_id, messages)
    finally:
        bus.outbound.put_nowait(TurnDone(msg_id))
        if compacted_size is not None:
            bus.outbound.put_nowait(CompactionDone(user_id, compacted_size))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
bus = Bus()


# USER DATABASE
user_registry: dict[UserID, list[Message]] = {}


def get_user_messages(user_id: UserID) -> list[Message]:
    if user_id not in user_registry:
        user_registry[user_id] = []
    return user_registry[user_id]


def set_user_messages(user_id: UserID, messages: list[Message]) -> None:
    user_registry[user_id] = messages


def _safe_to_write(session_id: SessionID, messages: list[Message]) -> bool:
    """Refuse to overwrite a non-empty on-disk conversation with an empty list.

    An empty in-memory list usually means the caller passed a user_id that does
    not match the session (e.g. the terminal uses "0"); writing it would wipe
    the saved history."""
    if messages:
        return True
    try:
        on_disk = read_conversation(session_id)
    except (FileNotFoundError, json.JSONDecodeError):
        return True  # nothing on disk to protect
    return len(on_disk) == 0


def reset_session(user_id: UserID = "", session_id: SessionID = ""):
    "Reset the session and chat history. Provide the Current user_id AND the session id"
    logger.info(f"Clear session for user_id {user_id}")
    messages: list[Message] = get_user_messages(user_id)
    if not _safe_to_write(session_id, messages):
        logger.error(f"Refusing to reset: in-memory list for {user_id} is empty but session {session_id} has content on disk")
        return json.dumps({"error": "refusing to overwrite a non-empty session with an empty message list (user_id mismatch?)"})
    filepath: str = archive_conversation(session_id, messages)
    messages.clear()
    set_user_messages(user_id, messages)
    write_conversation(session_id, messages)
    return json.dumps({"status": f"Message history cleared and archived at {filepath}"})


async def compact_session(user_id: UserID = "", session_id: SessionID = ""):
    "Compact the current conversation to free context space: prune old tool outputs, replace older history with a structured summary, keep recent turns verbatim"
    base_url = config.BASE_URL
    messages: list[Message] = get_user_messages(user_id)
    if not _safe_to_write(session_id, messages):
        logger.error(f"Refusing to compact: in-memory list for {user_id} is empty but session {session_id} has content on disk")
        return json.dumps({"error": "refusing to overwrite a non-empty session with an empty message list (user_id mismatch?)"})
    logger.info(f"Compacting session {session_id} on request")
    compacted = await compact_messages(base_url, messages, session_id, CONTEXT_SIZE, CURRENT_MODEL)
    write_conversation(session_id, messages)
    size = approximate_token_count(json.dumps(messages))
    if not compacted and size > compaction_trigger(CONTEXT_SIZE):
        return json.dumps({"error": f"compaction failed; context still {size} tokens"})
    return json.dumps({"status": f"Compaction complete. Context size is {size} of {CONTEXT_SIZE}"})


def _is_gate_command(event) -> bool:
    """Commands that must work while the context-size gate is closed."""
    return isinstance(event, UserMessage) and (
        event.text.lower().startswith("/server") or event.text.lower() == "/help"
    )


async def main(resume: bool = False, session_id: str | None = None) -> None:
    session_resumed = False

    async def _process_event(event) -> bool:
        """Process one inbound event; return False to stop the main loop."""
        nonlocal session_resumed
        match event:
            case Shutdown():
                return False
            case Interrupt(user_id):
                bus.interrupted.add(user_id)  # already set by Bus.post; keeps queue-drain consistent
                return True
            case UserMessage(user_id, text, msg_id):
                bus.interrupted.discard(user_id)  # a stale interrupt must not cancel the new turn
                if not text.strip():
                    bus.outbound.put_nowait(TurnDone(msg_id))
                    return True
                # /server must work before the gate: it is how you fix a wrong/unreachable server.
                if text.lower().startswith("/server"):
                    await handle_server_command(user_id, msg_id, text)
                    return True
                if text.lower() == "/help":
                    emit(user_id, "info", "Available commands:\n" + "\n".join(f"{name:<15} {desc}" for name, desc in SLASH_COMMANDS))
                    bus.outbound.put_nowait(TurnDone(msg_id))
                    return True
                if CONTEXT_SIZE is None:  # defensive; main() gates prompts before dispatch
                    emit(user_id, "info", "Waiting for model server - context size not yet determined")
                    await ctx_ready.wait()

                # load the conversation up once hacky TODO
                if resume and not session_resumed:
                    set_user_messages(user_id, read_conversation(session_id))
                    session_resumed = True

                if text.lower() == "/compact":
                    logger.info(f"/compact command from {user_id}")
                    try:
                        result = json.loads(await compact_session(user_id, session_id))
                    except Exception as e:
                        report_error(user_id, "Compaction failed", e)
                        bus.outbound.put_nowait(TurnDone(msg_id))
                        return True
                    if "error" in result:
                        report_error(user_id, result["error"])
                        bus.outbound.put_nowait(TurnDone(msg_id))
                    else:
                        # Posted after TurnDone so the status line keeps the message (TurnDone resets it to "ready").
                        bus.outbound.put_nowait(TurnDone(msg_id))
                        bus.outbound.put_nowait(CompactionDone(user_id, context_token_count(get_user_messages(user_id))))
                    return True
                try:
                    await handle_message(text, config.BASE_URL, user_id, session_id, msg_id)
                except asyncio.CancelledError:
                    logger.warning("Message handling cancelled")
                    return False
                except Exception as e:
                    # A failed turn must not kill the process; surface it and keep serving.
                    report_error(user_id, "Turn failed", e)
                return True

    async def _wait_for_gate() -> list:
        """Wait for the context-size gate; /server and /help are served while waiting.

        A /server mid-wait restarts the fetcher, so the wait re-arms against the new
        URL. Other events are held (returned) for in-order processing once the gate opens."""
        held = []
        while CONTEXT_SIZE is None:
            get_task = asyncio.create_task(bus.inbound.get())
            ready_task = asyncio.create_task(ctx_ready.wait())
            try:
                await asyncio.wait({get_task, ready_task}, return_when=asyncio.FIRST_COMPLETED)
            except asyncio.CancelledError:
                get_task.cancel()
                ready_task.cancel()
                raise
            if ctx_ready.is_set():
                if get_task.done():  # arrived with the gate opening; process it after the gated event
                    held.append(get_task.result())
                else:
                    get_task.cancel()
                break
            event = get_task.result()
            if _is_gate_command(event):
                await _process_event(event)
            else:
                held.append(event)
        return held

    try:
        while True:
            event = await bus.inbound.get()
            if (
                isinstance(event, UserMessage)
                and event.text.strip()
                and not _is_gate_command(event)
                and CONTEXT_SIZE is None
            ):
                emit(event.user_id, "info", "Waiting for model server - context size not yet determined")
                held = await _wait_for_gate()
                batch = [event, *held]
            else:
                batch = [event]
            for ev in batch:
                if not await _process_event(ev):
                    return
    except asyncio.CancelledError:
        logger.warning("Main loop cancelled")
        raise
    except Exception as e:
        logger.error(f"Main loop error: {e}")
        raise


CLIENTS = {
    "terminal": "clients.terminal",
    "discord_client": "clients.discord_client",
    "irc_client": "clients.irc_client",
}


def _load_client(client_type: str):
    try:
        return importlib.import_module(CLIENTS[client_type])
    except KeyError:
        raise ValueError(f"Unknown client type: {client_type}") from None


async def async_main(client_type: str, session_id: str | None = None) -> None:
    global ctx_task
    client = _load_client(client_type)

    resume = False
    if session_id is not None:
        resume = True
    if session_id and session_id == "true":
        session_id = get_last_conversation_session_id()
        resume = True
    elif session_id is None:
        session_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

    bus.bind()

    if resume and client_type == "terminal":
        try:
            bus.outbound.put_nowait(History(read_conversation(session_id)))
        except (OSError, ValueError) as e:
            logger.warning(f"Could not load history for session {session_id}: {e}")

    # Server-dependent work runs in the background; the UI comes up immediately.
    load_task = asyncio.create_task(startup_load_model())
    ctx_task = asyncio.create_task(wait_for_context_size())
    init_task = asyncio.create_task(client.run(bus, lambda: CONTEXT_SIZE))
    main_task = asyncio.create_task(main(resume=resume, session_id=session_id))
    try:
        await asyncio.wait({init_task, main_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        shutdown.set()  # stop background workers from starting new attempts
        for task in (load_task, ctx_task, init_task, main_task):
            task.cancel()
        try:
            await asyncio.gather(load_task, ctx_task, init_task, main_task, return_exceptions=True)
        except asyncio.CancelledError:
            logger.info("Cancelled during cleanup")
        try:
            client.stop()
        except Exception as e:
            logger.warning(f"Error during client cleanup: {e}")
        if session_id and session_id != "true":
            print(f"Your session id: {session_id}")


def cli() -> None:
    client = config.CLIENT
    if len(sys.argv) > 1:
        client = sys.argv[1]
    session_id: str | None = None
    if len(sys.argv) > 2:
        session_id = sys.argv[2]
    # Run the async function with proper cancellation handling
    asyncio.run(async_main(client, session_id))


if __name__ == "__main__":
    cli()
