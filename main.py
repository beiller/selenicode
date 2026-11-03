#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import pathlib
import random
import string
import subprocess
import sys
import types
import urllib.request
from datetime import datetime
from typing import AsyncIterator, Callable, Union, get_type_hints

import aiohttp

# Load environment variables with priority order
# 1. .env.defaults
# 2. .env (user overrides - if exists)
try:
    from dotenv import load_dotenv

    load_dotenv(".env.defaults", override=False)

    if os.path.exists(".env"):
        load_dotenv(".env", override=True)

    # Set fallback defaults if neither file has them
    if "LLAMA_BASE_URL" not in os.environ:
        os.environ["LLAMA_BASE_URL"] = "http://127.0.0.1:8080"
except ImportError:
    # If python-dotenv is not installed, use hardcoded defaults
    if "LLAMA_BASE_URL" not in os.environ:
        os.environ["LLAMA_BASE_URL"] = "http://127.0.0.1:8080"
import logging
from logging.handlers import RotatingFileHandler

LOG_FILE = pathlib.Path(__file__).parent / "agent1.log"

# File-only logging: a console handler would write raw text over the terminal
# client's screen. Everything lands in agent1.log (rotating).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)],
)
logger = logging.getLogger()


# Import tools from tools.py
from download_model import download_model, search_model
from tools import (
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

CURRENT_MODEL = os.environ.get("CURRENT_MODEL", "Qwen_Qwen3.5-27B-Q4_1")
CONTEXT_SIZE = 16384


def is_llama_running(base_url: str) -> bool:
    """Check if llama-server is already running by making a request to the API."""
    try:
        import urllib.request

        response = urllib.request.urlopen(f"{base_url}/v1/models", timeout=2)
        return response.status == 200
    except Exception:
        return False


def load_model_and_set(model_name: str):
    global CURRENT_MODEL
    loaded_model = get_loaded_model()
    if loaded_model:
        unload_model(loaded_model)
    load_model(model_name)
    CURRENT_MODEL = model_name


def get_context_size() -> int:
    """Get the context size from llama-server or vLLM."""
    base_url = os.getenv("LLAMA_BASE_URL")
    context_size = int(os.environ.get("CTX_SIZE") or "16384")
    try:
        req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
        models = result.get("data", [])
        for preferred in (True, False):
            for model in models:
                if preferred and model.get("id") != CURRENT_MODEL:
                    continue
                # llama.cpp: nested under "meta"; vLLM: top-level field
                n_ctx = (model.get("meta") or {}).get("n_ctx", 0) or model.get("max_model_len", 0)
                if n_ctx > 0:
                    context_size = n_ctx
                    break
            else:
                continue
            break
    except Exception:
        logger.warning(f"Could not read a context size, returning default of {context_size}")
    logger.info(f"Context size: {context_size}")
    return context_size


# ---------------------------------------------------------------------------
# Types – Chat Completions API
# ---------------------------------------------------------------------------

from bus import Bus, History, Interrupt, Shutdown, Token, TurnDone, UsageReport, UserMessage
from compaction import compact_messages, compaction_target, compaction_trigger, context_token_count
from conversation import (
    approximate_token_count,
    archive_conversation,
    get_last_conversation_session_id,
    read_conversation,
    write_conversation,
)
from main_types import (
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


def make_tool_result_message(tool_call_id: str, content: str) -> Message:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR = pathlib.Path(__file__).parent
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
    system_content += session_context

    # Strip timestamps before sending to LLM API
    msgs = [make_system_message(system_content)] + [
        {k: v for k, v in m.items() if k != "timestamp"} for m in messages if m.get("role") != "system"
    ]
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
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(sock_read=None, sock_connect=5 * 60, connect=5 * 60),
        ) as resp:
            logger.debug(f"loading: {base_url}/v1/chat/completions")
            async for line in resp.content:
                raw_line = line.decode("utf-8").strip()
                if not raw_line or raw_line.startswith(":"):
                    continue
                if raw_line == "data: [DONE]":
                    break
                if raw_line.startswith("data: "):
                    yield json.loads(raw_line[6:])


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
            emit(user_id, "error", f"Unknown tool: {name}")
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
                logger.exception(f"Tool {name} failed")
                output = json.dumps({"error": f"Tool {name} raised an exception: {e}"})
        results.append(make_tool_result_message(tc["id"], output))
    return results


# ---------------------------------------------------------------------------
# Conversation loop
# ---------------------------------------------------------------------------


def build_tools() -> tuple[list[Tool], dict[str, ToolHandler]]:
    """Build the full tools list and registry by reloading skills from disk."""
    tools: list[Tool] = [
        tool_from_function(conversation_search),
        tool_from_function(run_bash),
        tool_from_function(download_model),
        tool_from_function(load_model),
        tool_from_function(write_file),
        tool_from_function(edit_file),
        tool_from_function(read_file),
        tool_from_function(list_models),
        tool_from_function(search_model),
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
        "download_model": download_model,
        "load_model": load_model_and_set,
        "list_models": list_models,
        "search_model": search_model,
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
                logger.warning(f"Compaction insufficient mid-loop: {size} tokens > {CONTEXT_SIZE}, stopping turn")
                emit(user_id, "info", f"Context still at {size} tokens after compaction, stopping turn")
                return ""

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
            emit(user_id, "info", "too many tool calls, giving up")
            return _finalize_assistant(assistant_msg, messages)


async def handle_message(user_input: str, base_url: str, user_id: UserID, session_id: SessionID, msg_id: str) -> None:
    """Handle a user message event: append it, run the tool loop, emit the reply."""
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
            await compact_messages(base_url, messages, session_id, CONTEXT_SIZE, CURRENT_MODEL)
            size = approximate_token_count(json.dumps(messages))
            logger.info(f"Context size is {size} of {CONTEXT_SIZE}")

        write_conversation(session_id, messages)
    finally:
        bus.outbound.put_nowait(TurnDone(msg_id))


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
        return json.dumps({"status": "error: refusing to overwrite a non-empty session with an empty message list (user_id mismatch?)"})
    filepath: str = archive_conversation(session_id, messages)
    messages.clear()
    set_user_messages(user_id, messages)
    write_conversation(session_id, messages)
    return json.dumps({"status": f"Message history cleared and archived at {filepath}"})


async def compact_session(user_id: UserID = "", session_id: SessionID = ""):
    "Compact the current conversation to free context space: prune old tool outputs, replace older history with a structured summary, keep recent turns verbatim"
    base_url = os.getenv("LLAMA_BASE_URL")
    messages: list[Message] = get_user_messages(user_id)
    if not _safe_to_write(session_id, messages):
        logger.error(f"Refusing to compact: in-memory list for {user_id} is empty but session {session_id} has content on disk")
        return json.dumps({"status": "error: refusing to overwrite a non-empty session with an empty message list (user_id mismatch?)"})
    logger.info(f"Compacting session {session_id} on request")
    await compact_messages(base_url, messages, session_id, CONTEXT_SIZE, CURRENT_MODEL)
    write_conversation(session_id, messages)
    size = approximate_token_count(json.dumps(messages))
    return json.dumps({"status": f"Compaction complete. Context size is {size} of {CONTEXT_SIZE}"})


async def main_init() -> None:
    global CONTEXT_SIZE
    """Start llama-server if needed, load the model, and read the context size."""
    # Load from environment variables if not provided
    base_url = os.getenv("LLAMA_BASE_URL")

    # Run llama.sh before starting the main server loop
    # Check if llama-server is already running
    global llama_proc
    llama_proc = None

    if is_llama_running(base_url):
        logger.info("llama-server is already running, skipping start")
    else:
        # Try starting llama-server in background with fallback to setup.sh
        logger.info("Attempting to start llama-server via start_llama.sh...")
        llama_proc = subprocess.Popen(
            "./start_llama.sh", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
        )
        await asyncio.sleep(2)  # Wait briefly for it to start/fail without blocking
        if llama_proc.poll() is None:
            logger.info("start_llama.sh started successfully")
        else:
            logger.warning(f"start_llama.sh exited with code {llama_proc.returncode}, trying setup.sh once...")
            subprocess.run(["./setup.sh"], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # messages: list[Message] = []

    # emit("info", f"Chat (Ctrl-C to cancel, twice to quit) - URL: {base_url}")

    logger.info(f"Loading model {CURRENT_MODEL}")
    load_model_and_set(CURRENT_MODEL)
    CONTEXT_SIZE = get_context_size()
    trigger = compaction_trigger(CONTEXT_SIZE)
    target = compaction_target(CONTEXT_SIZE)
    logger.info(f"Compaction: auto-trigger at {trigger} tokens, shrink-to target {target} tokens")
    if target >= trigger:
        logger.warning("Shrink-to target is not below the auto-trigger; auto-compaction will be a no-op")

    await asyncio.sleep(3)


async def main(resume: bool = False, session_id: str | None = None) -> None:
    base_url = os.getenv("LLAMA_BASE_URL")
    session_resumed = False
    try:
        while True:
            event = await bus.inbound.get()
            match event:
                case Shutdown():
                    break
                case Interrupt(user_id):
                    bus.interrupted.add(user_id)  # already set by Bus.post; keeps queue-drain consistent
                    continue
                case UserMessage(user_id, text, msg_id):
                    bus.interrupted.discard(user_id)  # a stale interrupt must not cancel the new turn

                    # load the conversation up once hacky TODO
                    if resume and not session_resumed:
                        set_user_messages(user_id, read_conversation(session_id))
                        session_resumed = True

                    if not text.strip():
                        bus.outbound.put_nowait(TurnDone(msg_id))
                        continue
                    if text.lower() == "/compact":
                        logger.info(f"/compact command from {user_id}")
                        result = await compact_session(user_id, session_id)
                        emit(user_id, "info", json.loads(result)["status"])
                        bus.outbound.put_nowait(TurnDone(msg_id))
                        continue
                    try:
                        await handle_message(text, base_url, user_id, session_id, msg_id)
                    except asyncio.CancelledError:
                        logger.warning("Message handling cancelled")
                        break
    except asyncio.CancelledError:
        logger.warning("Main loop cancelled")
        raise
    except Exception as e:
        logger.error(f"Main loop error: {e}")
        raise
    finally:
        if llama_proc is not None:
            try:
                llama_proc.terminate()
                try:
                    llama_proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    llama_proc.kill()
                logger.info("llama-server stopped")
            except Exception as e:
                logger.error(f"Stopping llama-server: {e}")


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


async def async_main(client_type: str, session_id: str | None = None):
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
    await main_init()

    if resume and client_type == "terminal":
        try:
            bus.outbound.put_nowait(History(read_conversation(session_id)))
        except (OSError, ValueError) as e:
            logger.warning(f"Could not load history for session {session_id}: {e}")

    init_task = asyncio.create_task(client.run(bus, CONTEXT_SIZE))
    main_task = asyncio.create_task(main(resume=resume, session_id=session_id))
    try:
        await asyncio.wait({init_task, main_task}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (init_task, main_task):
            task.cancel()
        try:
            await asyncio.gather(init_task, main_task, return_exceptions=True)
        except asyncio.CancelledError:
            logger.info("Cancelled during cleanup")
        try:
            client.stop()
        except Exception as e:
            logger.warning(f"Error during client cleanup: {e}")
        if session_id and session_id != "true":
            print(f"Your session id: {session_id}")


if __name__ == "__main__":
    exec_dir = os.path.dirname(os.path.abspath(__file__))

    def load_config():
        with open(f"{exec_dir}/config.json") as fh:
            return json.loads(fh.read())

    config = load_config()

    if len(sys.argv) > 1:
        config["client"] = sys.argv[1]
    session_id: str | None = None
    if len(sys.argv) > 2:
        session_id = sys.argv[2]
    # Run the async function with proper cancellation handling
    asyncio.run(async_main(config.get("client", "terminal"), session_id))
