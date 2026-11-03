import json
import logging
import os

import aiohttp

from conversation import approximate_token_count
from main_types import Message, SessionID

logger = logging.getLogger()

DEFAULT_TAIL_TURNS = 2
MIN_PRESERVE_RECENT_TOKENS = 2_000
MAX_PRESERVE_RECENT_TOKENS = 8_000
TOOL_OUTPUT_MAX_CHARS = 2_000
SUMMARY_MAX_TOKENS = 4_096
PRUNE_PROTECT_RATIO = 0.25
PRUNE_MINIMUM_RATIO = 0.10
PRUNED_TOOL_NOTE = "[Old tool result content cleared]"

SUMMARY_SYSTEM_PROMPT = (
    "You are an anchored context summarization assistant for coding sessions. "
    "Summarize only the conversation history you are given. The newest turns may be kept verbatim "
    "outside your summary, so focus on the older context that still matters for continuing the work. "
    "If the prompt includes a <previous-summary> block, treat it as the current anchored summary. "
    "Update it with the new history by preserving still-true details, removing stale details, and merging in new facts. "
    "Always follow the exact output structure requested by the user prompt. Keep every section, preserve exact file paths "
    "and identifiers when known, and prefer terse bullets over paragraphs. Do not answer the conversation itself. "
    "Do not mention that you are summarizing, compacting, or merging context."
)

SUMMARY_TEMPLATE = """Output exactly the Markdown structure shown inside <template> and keep the section order unchanged. Do not include the <template> tags in your response.
<template>
## Goal
- [single-sentence task summary]

## Constraints & Preferences
- [user constraints, preferences, specs, or "(none)"]

## Progress
### Done
- [completed work or "(none)"]

### In Progress
- [current work or "(none)"]

### Blocked
- [blockers or "(none)"]

## Key Decisions
- [decision and why, or "(none)"]

## Next Steps
- [ordered next actions or "(none)"]

## Critical Context
- [important technical facts, errors, open questions, or "(none)"]

## Relevant Files
- [file or directory path: why it matters, or "(none)"]
</template>

Rules:
- Keep every section, even when empty.
- Use terse bullets, not prose paragraphs.
- Preserve exact file paths, commands, error strings, and identifiers when known.
- Do not mention the summary process or that context was compacted."""


def _pct(env_var: str, default: float) -> float:
    raw = os.environ.get(env_var, str(default))
    try:
        pct = float(raw)
    except ValueError:
        logger.warning(f"Invalid {env_var}={raw!r}, using default {default:g}")
        return default
    if not 1 <= pct <= 100:
        logger.warning(f"{env_var}={pct} out of range 1-100, clamping")
        pct = min(max(pct, 1.0), 100.0)
    return pct


def compaction_trigger(context_size: int) -> int:
    """Auto-trigger threshold: compaction starts when context exceeds this (default: 80%)."""
    return int(context_size * _pct("COMPACTION_TRIGGER_PCT", 80.0) / 100)


def compaction_target(context_size: int) -> int:
    """Shrink-to target: compaction stops once context is at or below this (default: 20%)."""
    return int(context_size * _pct("COMPACTION_TARGET_PCT", 20.0) / 100)


def context_token_count(messages: list[Message]) -> int:
    return approximate_token_count(json.dumps(messages))


def preserve_recent_budget(context_size: int) -> int:
    budget = min(MAX_PRESERVE_RECENT_TOKENS, max(MIN_PRESERVE_RECENT_TOKENS, int(context_size * 0.25)))
    return min(budget, compaction_target(context_size))


def _tail_turns() -> int:
    raw = os.environ.get("COMPACTION_TAIL_TURNS", str(DEFAULT_TAIL_TURNS))
    try:
        n = int(raw)
    except ValueError:
        logger.warning(f"Invalid COMPACTION_TAIL_TURNS={raw!r}, using default {DEFAULT_TAIL_TURNS}")
        return DEFAULT_TAIL_TURNS
    return max(0, n)


def turns(messages: list[Message]) -> list[list[int]]:
    result = []
    for i, m in enumerate(messages):
        if m.get("role") != "user" or m.get("compaction_marker"):
            continue
        result.append([i, len(messages)])
    for i in range(len(result) - 1):
        result[i][1] = result[i + 1][0]
    return result


def _split_turn(messages: list[Message], start: int, end: int, budget: int) -> int | None:
    for i in range(start + 1, end):
        if messages[i].get("role") == "tool":
            continue
        if context_token_count(messages[i:end]) <= budget:
            return i
    return None


def select(messages: list[Message], context_size: int) -> tuple[list[Message], list[Message]]:
    """Split history into (head, tail): the newest turns that fit the preserve budget are kept verbatim."""
    limit = _tail_turns()
    if limit <= 0:
        return messages, []
    all_turns = turns(messages)
    if not all_turns:
        return messages, []
    budget = preserve_recent_budget(context_size)
    recent = all_turns[-limit:]
    total = 0
    keep_start: int | None = None
    for i in range(len(recent) - 1, -1, -1):
        t_start, t_end = recent[i]
        size = context_token_count(messages[t_start:t_end])
        if total + size <= budget:
            total += size
            keep_start = t_start
            continue
        split = _split_turn(messages, t_start, t_end, budget - total)
        if split is not None:
            keep_start = split
        break
    if keep_start is None or keep_start == 0:
        return messages, []
    return messages[:keep_start], messages[keep_start:]


def prune(messages: list[Message], context_size: int) -> None:
    """Erase old tool outputs to free space; the two newest user turns are protected."""
    protect = int(context_size * PRUNE_PROTECT_RATIO)
    minimum = int(context_size * PRUNE_MINIMUM_RATIO)
    total = 0
    freed = 0
    to_prune: list[Message] = []
    user_turns = 0
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("role") == "user":
            user_turns += 1
        if user_turns < 2:
            continue
        if m.get("compaction_summary"):
            break
        if m.get("role") != "tool":
            continue
        content = m.get("content") or ""
        if not content or PRUNED_TOOL_NOTE in content:
            continue
        est = approximate_token_count(content)
        total += est
        if total <= protect:
            continue
        freed += est
        to_prune.append(m)
    if freed > minimum:
        for m in to_prune:
            m["content"] = PRUNED_TOOL_NOTE


def _truncate_tool_output(text: str) -> str:
    if len(text) <= TOOL_OUTPUT_MAX_CHARS:
        return text
    return f"{text[:TOOL_OUTPUT_MAX_CHARS]}\n[Tool output truncated for compaction: omitted {len(text) - TOOL_OUTPUT_MAX_CHARS} chars]"


def to_summary_input(m: Message) -> dict:
    out: dict = {"role": m["role"]}
    content = m.get("content")
    if m["role"] == "tool" and content:
        content = _truncate_tool_output(content)
    if content is not None:
        out["content"] = content
    if m["role"] == "assistant" and m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    if m["role"] == "tool" and m.get("tool_call_id"):
        out["tool_call_id"] = m["tool_call_id"]
    return out


def build_prompt(previous_summary: str | None) -> str:
    anchor = (
        "\n".join(
            [
                "Update the anchored summary below using the conversation history above.",
                "Preserve still-true details, remove stale details, and merge in the new facts.",
                "<previous-summary>",
                previous_summary,
                "</previous-summary>",
            ]
        )
        if previous_summary
        else "Create a new anchored summary from the conversation history above."
    )
    return "\n\n".join([anchor, SUMMARY_TEMPLATE])


async def summarize_messages(base_url: str, model: str, system_prompt: str, messages: list[dict]) -> str | None:
    """Send head history plus template prompt to the model; return the summary or None on failure."""
    body: dict = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, *messages],
        "temperature": 0.2,
        "stream": True,
        "max_tokens": SUMMARY_MAX_TOKENS,
    }
    try:
        parts: list[str] = []
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/v1/chat/completions",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(sock_read=None, sock_connect=5 * 60, connect=5 * 60),
            ) as resp:
                async for line in resp.content:
                    raw_line = line.decode("utf-8").strip()
                    if not raw_line or raw_line.startswith(":"):
                        continue
                    if raw_line == "data: [DONE]":
                        break
                    if raw_line.startswith("data: "):
                        chunk = json.loads(raw_line[6:])
                        choices = chunk.get("choices")
                        if not choices:
                            continue
                        content = choices[0].get("delta", {}).get("content")
                        if content:
                            parts.append(content)
        return "".join(parts).strip() or None
    except Exception:
        logger.exception("summarize_messages failed")
        return None


def find_previous_summary(messages: list[Message]) -> tuple[str | None, int | None]:
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("compaction_summary"):
            return messages[i].get("content"), i
    return None, None


async def compact_messages(
    base_url: str, messages: list[Message], session_id: SessionID, context_size: int, model: str
) -> None:
    """Opencode-style compaction: prune old tool outputs, then replace the head with an anchored
    structured summary (marker user message + assistant summary message) and keep the newest turns verbatim."""
    if context_token_count(messages) <= compaction_target(context_size):
        return

    prune(messages, context_size)

    previous_summary, prev_idx = find_previous_summary(messages)
    history = messages[prev_idx + 1 :] if prev_idx is not None else messages
    head, tail = select(history, context_size)
    if not head:
        return

    model_input = [to_summary_input(m) for m in head]
    summary = await summarize_messages(
        base_url, model, SUMMARY_SYSTEM_PROMPT, model_input + [{"role": "user", "content": build_prompt(previous_summary)}]
    )
    if not summary:
        logger.warning("Compaction v2 summarization failed, keeping current context")
        return

    new_messages: list[Message] = [
        {"role": "user", "content": "[compaction] The conversation before this point was summarized.", "compaction_marker": True},
        {"role": "assistant", "content": summary, "compaction_summary": True},
        *tail,
    ]

    last_user: Message | None = None
    for m in reversed(history):
        if m.get("role") == "user" and not m.get("compaction_marker"):
            last_user = m
            break
    if last_user is not None and all(m is not last_user for m in tail):
        new_messages.append(dict(last_user))

    messages.clear()
    messages.extend(new_messages)
