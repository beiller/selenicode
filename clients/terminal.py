"""Bare-bones curses terminal client.

Full-screen TUI with O(visible-lines) render cost: a scrollback viewport, a
pinned status bar and a pinned prompt line. No widgets, no markdown engine -
every message is plain ANSI-styled text, so hundreds of messages cost the same
as dozens.

Layout (H x W):
    rows 0 .. H-3   scrollback viewport (window into all message lines)
    row   H-2       status bar (bold green): spinner/state + token usage
    row   H-1       prompt line: "> " + input + cursor

Keys: Enter submit, Backspace/Left/Right edit, PgUp/PgDn/Home/End/wheel scroll,
Ctrl+C interrupt a running turn (clears input when idle), Ctrl+Q quit.
"""

import asyncio
import curses
import fcntl
import json
import logging
import os
import pathlib
import struct
import sys
import termios
import textwrap
import time
import unicodedata
import uuid
from typing import Callable

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

logger = logging.getLogger()

USER_ID = "0"

# key codes (curses KEY_* values; enter/backspace are raw bytes)
K_ENTER = 10
K_BACKSPACE = 127
K_ESC = 27
K_CTRL_C = 3
K_CTRL_Q = 17
K_CTRL_T = 20
K_UP = 259
K_DOWN = 260
K_LEFT = 261
K_RIGHT = 262
_BLOCK_GAP = 1  # blank rows between scrollback sections (thinking/message/tool)
K_HOME = 270
K_END = 271
K_PPAGE = 339
K_NPAGE = 338

SPINNER = "🌑🌒🌓🌔🌕🌖🌗🌘"

# style ids (mapped to curses attrs in _curses_setup; pure layer stays curses-free)
STY_PLAIN = 0
STY_DIM = 1
STY_YOU = 2
STY_TOOL = 3
STY_INFO = 4
STY_STATUS = 5
STY_ERROR = 6
STY_WARN = 7

_ROLE_STYLE = {
    "thinking": STY_DIM,
    "assistant": STY_PLAIN,
    "tool": STY_TOOL,
    "info": STY_INFO,
    "error": STY_ERROR,
}
_ROLE_PREFIX = {
    "thinking": "(thinking) ",
    "assistant": "\U0001F916: ",
    "user": "You: ",
    "error": "ERROR: ",
}


class Block:
    """One rendered message: role-styled raw text wrapped at render time."""

    __slots__ = ("role", "prefix", "raw", "style")

    def __init__(self, role: str, prefix: str, raw: str, style: int) -> None:
        self.role = role
        self.prefix = prefix
        self.raw = raw
        self.style = style


class UIState:
    """All UI state; every mutator is a pure-ish function testable headless."""

    def __init__(self, context_size: int | Callable[[], int | None] = 0) -> None:
        self.blocks: list[Block] = []
        self.active: Block | None = None
        self.cache: dict[int, tuple[int, list]] = {}  # id(block) -> (width, lines)
        # Absolute index of the viewport's top line; None means pinned to the
        # bottom (auto-follows new content). While set, streaming does not move
        # the view - new lines accumulate below it.
        self.view_top: int | None = None
        self.input_buf = ""
        self.cursor = 0
        self.show_thinking = False  # ctrl+t toggles; collapsed to a placeholder line by default
        self.status = "ready"
        self.status_style = STY_STATUS
        self.busy = False
        self.tokens = 0
        self.context_size = context_size
        self.frame = 0


def wrap_text(text: str, width: int) -> list[str]:
    """Wrap text into lines of at most `width` cells (empty paragraphs kept)."""
    if not text:
        return [""]
    out: list[str] = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, max(1, width)) or [""])
    return out


def _block_lines(state: UIState, block: Block, width: int) -> list[list[tuple[str, int]]]:
    """Wrapped, styled lines for a block; finalized blocks are cached per
    (block, width). The active block is never cached: its raw text grows on
    every token while the object identity stays the same."""
    key = id(block)
    if block is not state.active:
        cached = state.cache.get(key)
        # show_thinking is part of the key: a cached collapsed placeholder
        # must not be served after the user expands (or vice versa).
        if cached is not None and cached[0] == (width, state.show_thinking):
            return cached[1]
    lines: list[list[tuple[str, int]]] = []
    if block.role == "thinking" and not state.show_thinking:
        # Collapsed placeholder: char count + first 100 chars (whitespace
        # collapsed) so hidden thinking is visible at a glance.
        preview = " ".join(block.raw.split())
        if len(preview) > 100:
            preview = preview[:100] + "\u2026"
        lines.append([(f"(thinking) (collapsed) {len(block.raw)} chars: {preview}", block.style)])
    else:
        wrapped = wrap_text(block.raw, max(1, width - 2))
        for i, ln in enumerate(wrapped):
            if i == 0 and block.prefix:
                lines.append([(block.prefix, block.style), (ln, block.style)])
            elif i > 0:
                lines.append([("  " + ln, block.style)])
            else:
                lines.append([(ln, block.style)])
    if block is not state.active:
        state.cache[key] = ((width, state.show_thinking), lines)
    return lines


def _input_block_rows(input_lines: list[str], width: int) -> list[list[tuple[str, int]]]:
    """Rows for the input block; long lines wrap across rows so the cursor
    never runs off the right edge ("> " on the first line, indented after)."""
    chunk_w = max(1, width - 2)
    rows: list[list[tuple[str, int]]] = []
    for i, ln in enumerate(input_lines):
        chunks = [ln[j:j + chunk_w] for j in range(0, len(ln), chunk_w)] or [""]
        for k, chunk in enumerate(chunks):
            prefix = ("> ", STY_YOU) if (i == 0 and k == 0) else ("  ", STY_PLAIN)
            rows.append([prefix, (chunk, STY_PLAIN)])
    return rows


def _input_block_height(input_lines: list[str], width: int) -> int:
    """Rows the input block occupies (long lines wrap)."""
    chunk_w = max(1, width - 2)
    return sum(max(1, (len(ln) + chunk_w - 1) // chunk_w) for ln in input_lines)


def render_frame(state: UIState, height: int, width: int) -> list[list[tuple[str, int]]]:
    """Build the full screen as rows of (text, style) segments. Pure."""
    input_lines = state.input_buf.split("\n") if state.input_buf else [""]
    view_h = max(1, height - 1 - _input_block_height(input_lines, width))
    all_lines = _scrollback_rows(state, width)

    total = len(all_lines)
    bottom = max(0, total - view_h)
    top = bottom if state.view_top is None else min(state.view_top, bottom)

    rows: list[list[tuple[str, int]]] = []
    for i in range(view_h):
        idx = top + i
        rows.append(all_lines[idx] if idx < total else [("", STY_PLAIN)])

    label = f"{SPINNER[state.frame % len(SPINNER)]} thinking" if state.busy else state.status
    cs = state.context_size() if callable(state.context_size) else state.context_size
    if cs:
        pct = int(state.tokens * 100 / cs)
        tok = f" tokens: {state.tokens}/{cs} ({pct}%)"
    else:
        tok = ""
    hidden = sum(1 for b in state.blocks if b.role == "thinking") + (
        1 if state.active is not None and state.active.role == "thinking" else 0
    )
    hid = f" · {hidden} thinking collapsed (ctrl+t)" if hidden and not state.show_thinking else ""
    rows.append([(f"status: {label}{tok}{hid}", STY_STATUS if state.busy else state.status_style)])

    # Input block: one row per line (long lines wrap), "> " on the first,
    # indented after; the block's last row is the screen bottom. The
    # viewport shrinks to fit.
    rows.extend(_input_block_rows(input_lines, width))
    if not state.input_buf:
        rows[-1].append(("  ↑↓/pgup scroll · ctrl+c interrupt · ctrl+q quit · ctrl+t show thinking · /help for commands", STY_DIM))
    return rows


def _finalize_active(state: UIState) -> None:
    if state.active is not None and state.active.raw.strip():
        state.blocks.append(state.active)
    state.active = None


def apply_token(state: UIState, role: str, text: str, chunk: bool) -> None:
    """Feed one outbound Token event into the state."""
    if chunk:
        if state.active is None or state.active.role != role:
            _finalize_active(state)
            style = _ROLE_STYLE.get(role, STY_PLAIN)
            prefix = _ROLE_PREFIX.get(role, "")
            state.active = Block(role, prefix, "", style)
        state.active.raw += text
    elif text:
        # one-shot status line (tool call, info, error)
        _finalize_active(state)
        state.blocks.append(Block(role, "", text, _ROLE_STYLE.get(role, STY_INFO)))
    else:
        # per-LLM-call end marker
        _finalize_active(state)


def submit_message(state: UIState, text: str) -> None:
    """Record the user's prompt in the scrollback (matching how history
    renders it) and start the turn."""
    _finalize_active(state)
    state.blocks.append(Block("user", "You: ", text, STY_YOU))
    state.view_top = None  # follow the new content
    state.busy = True


def apply_key(state: UIState, key, view_h: int = 24, total: int = 0) -> tuple[str, str]:
    """Apply one key event. Returns (action, payload); action is one of
    "", "submit" (payload = text), "interrupt", "quit"."""
    if isinstance(key, tuple):  # ("wheel", delta): positive = older content
        _scroll(state, key[1], total, view_h)
        return "", ""
    if key == K_CTRL_Q:
        return "quit", ""
    if key == K_CTRL_C:
        if state.busy:
            return "interrupt", ""
        state.input_buf = ""
        state.cursor = 0
        return "", ""
    if key == K_CTRL_T:
        state.show_thinking = not state.show_thinking
        return "", ""
    if key in (K_ENTER, 13):
        text = state.input_buf.strip()
        state.input_buf = ""
        state.cursor = 0
        if not text:
            return "", ""
        return "submit", text
    if key == K_ESC:
        state.input_buf = ""
        state.cursor = 0
        return "", ""
    if key in (K_BACKSPACE, 8):
        if state.cursor > 0:
            state.input_buf = state.input_buf[: state.cursor - 1] + state.input_buf[state.cursor :]
            state.cursor -= 1
        return "", ""
    if key == K_UP:
        _scroll(state, 3, total, view_h)
        return "", ""
    if key == K_DOWN:
        _scroll(state, -3, total, view_h)
        return "", ""
    if key == K_LEFT:
        state.cursor = max(0, state.cursor - 1)
        return "", ""
    if key == K_RIGHT:
        state.cursor = min(len(state.input_buf), state.cursor + 1)
        return "", ""
    if key == K_PPAGE:
        _scroll(state, view_h - 1, total, view_h)
        return "", ""
    if key == K_NPAGE:
        _scroll(state, -(view_h - 1), total, view_h)
        return "", ""
    if key == K_HOME:
        state.view_top = 0
        return "", ""
    if key == K_END:
        state.view_top = None
        return "", ""
    if 32 <= key < 127:
        state.input_buf = state.input_buf[: state.cursor] + chr(key) + state.input_buf[state.cursor :]
        state.cursor += 1
        return "", ""
    return "", ""


def _scrollback_rows(state: UIState, width: int) -> list[list[tuple[str, int]]]:
    """All scrollback rows in order, with a blank separator between blocks.
    Single source of truth for both count_lines and render_frame so the two
    can never desync."""
    all_lines: list[list[tuple[str, int]]] = []
    blocks = state.blocks + ([state.active] if state.active is not None else [])
    for block in blocks:
        if all_lines:
            all_lines.extend([[("", STY_PLAIN)] for _ in range(_BLOCK_GAP)])
        all_lines.extend(_block_lines(state, block, width))
    return all_lines


def count_lines(state: UIState, width: int) -> int:
    """Total wrapped content lines (viewport area only, pinned bars excluded)."""
    return len(_scrollback_rows(state, width))


def _scroll(state: UIState, delta: int, total: int, view_h: int) -> None:
    """Scroll the viewport; positive delta moves toward older content.
    Absolute anchor: while detached, streaming does not move the view;
    reaching the bottom re-pins (auto-follow)."""
    bottom = max(0, total - view_h)
    if state.view_top is None:
        state.view_top = bottom  # first scroll up anchors at the current bottom
    state.view_top = max(0, min(state.view_top - delta, bottom))
    if state.view_top == bottom:
        state.view_top = None


def blocks_from_history(messages: list[dict]) -> list[Block]:
    """Convert saved conversation messages into renderable blocks."""
    out: list[Block] = []
    for m in messages:
        role = m.get("role")
        if role == "user":
            if m.get("compaction_marker"):
                out.append(Block("info", "", "[compaction] context summarized", STY_INFO))
            else:
                content = m.get("content") or ""
                if content:
                    out.append(Block("user", "You: ", content, STY_YOU))
        elif role == "assistant":
            reasoning = m.get("reasoning_content")
            if reasoning:
                out.append(Block("thinking", "(thinking) ", reasoning, STY_DIM))
            content = m.get("content")
            if m.get("compaction_summary"):
                if content:
                    out.append(Block("info", "(summary) ", content, STY_INFO))
            elif content:
                out.append(Block("assistant", "\U0001F916: ", content, STY_PLAIN))
            for tc in m.get("tool_calls") or []:
                try:
                    fn = tc["function"]
                    args = json.loads(fn.get("arguments") or "{}")
                    args_short = ", ".join(f"{k}={v!r}" for k, v in args.items())
                except (ValueError, AttributeError, KeyError, TypeError):
                    args_short = str(fn.get("arguments", "")) if isinstance(fn, dict) else str(fn)
                if len(args_short) > 200:
                    args_short = args_short[:200] + "..."
                name = fn.get("name", "?") if isinstance(fn, dict) else "?"
                out.append(Block("tool", "", f"{name}({args_short})", STY_TOOL))
        # tool results are not rendered (parity with the Textual client)
    return out


def apply_event(state: UIState, event) -> None:
    """Feed one outbound bus event into the state."""
    match event:
        case Token(_, role, text, chunk):
            apply_token(state, role, text, chunk)
        case Error(_, text):
            # Errors live on the status line (persistent, can't scroll away); full detail is in agent1.log.
            _finalize_active(state)
            state.status = text
            state.status_style = STY_ERROR
        case TurnDone(_):
            _finalize_active(state)
            state.busy = False
            state.status = "ready"
            state.status_style = STY_STATUS
        case ServerStatus(connected):
            if connected:
                state.status, state.status_style = "ready", STY_STATUS
            else:
                state.status, state.status_style = "waiting for model server...", STY_WARN
        case CompactionDone(_, tokens_after):
            # Posted after TurnDone so this survives the "ready" reset; next turn's TurnDone clears it.
            state.tokens = tokens_after
            state.status = f"compacted → {tokens_after} tokens"
            state.status_style = STY_STATUS
        case UsageReport(_, data):
            state.tokens = int(data.get("prompt_tokens", 0)) + int(data.get("completion_tokens", 0))
        case History(messages):
            for block in blocks_from_history(messages):
                state.blocks.append(block)


# ---------------------------------------------------------------------------
# curses I/O layer
# ---------------------------------------------------------------------------

_STYLE_ATTRS: dict[int, int] = {}


def _diagnose_terminal() -> None:
    """Log exactly which terminal/terminfo we are running under (for debugging)."""
    env_keys = ("TERM", "TERM_PROGRAM", "COLORTERM", "VTE_VERSION", "KONSOLE_VERSION", "KITTY_WINDOW_ID", "WEZTERM_PANE", "ALACRITTY_LOG")
    env = {k: os.environ[k] for k in env_keys if os.environ.get(k)}
    term = os.environ.get("TERM", "")
    first = term[:1]
    db_dirs = [
        pathlib.Path.home() / ".terminfo",
        pathlib.Path("/usr/share/terminfo"),
        pathlib.Path("/lib/terminfo"),
        pathlib.Path("/usr/lib/terminfo"),
        pathlib.Path("/usr/lib64/terminfo"),
    ]
    terminfo_file = next((d / first / term for d in db_dirs if (d / first / term).is_file()), None)
    logger.info("terminal startup: %s", env or "no terminal env vars set")
    logger.info("terminal startup: TERM=%r terminfo file: %s", term, str(terminfo_file) if terminfo_file else "NOT FOUND in plain-text db (may be compiled .db only)")


def _curses_setup() -> "curses._CursesWindow":
    stdscr = curses.initscr()
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_MAGENTA, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_GREEN, -1)
        curses.init_pair(5, curses.COLOR_RED, -1)
        _STYLE_ATTRS.update(
            {
                STY_PLAIN: 0,
                # faded text (thinking, hints)
                STY_DIM: curses.A_DIM,
                STY_YOU: curses.A_BOLD | curses.color_pair(1),
                STY_TOOL: curses.color_pair(2),
                STY_INFO: curses.A_BOLD | curses.color_pair(3),
                # bold green: UI chrome, distinct from the cyan "You:" text.
                STY_STATUS: curses.A_BOLD | curses.color_pair(4),
                STY_ERROR: curses.A_BOLD | curses.color_pair(5),
                # plain yellow (the bold yellow pair is taken by info)
                STY_WARN: curses.color_pair(3),
            }
        )
    except curses.error:
        _STYLE_ATTRS.update({STY_PLAIN: 0, STY_DIM: curses.A_DIM, STY_YOU: curses.A_BOLD, STY_TOOL: 0, STY_INFO: curses.A_BOLD, STY_STATUS: curses.A_BOLD, STY_ERROR: curses.A_BOLD, STY_WARN: 0})
    # raw() (not cbreak) so ISIG is off: Ctrl+C arrives as a 0x03 byte for
    # our decoder instead of raising SIGINT. Input is read from stdin directly;
    # getch is never used.
    curses.raw()
    curses.noecho()
    # NOTE: mouse reporting is deliberately NOT enabled. This system's ncurses
    # enables SGR mouse mode but fails to parse the sequences; _decode_input
    # swallows any mouse sequences that do arrive. Scroll with up/down,
    # PgUp/PgDn, Home/End instead.
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    return stdscr


def _decode_esc_body(buf: bytes):
    """Map a post-ESC byte sequence to a key constant; None = swallow (mouse
    garbage or unknown protocol). Empty buffer = bare ESC."""
    if not buf:
        return K_ESC
    mapping = {
        b"[A": K_UP,
        b"[B": K_DOWN,
        b"[C": K_RIGHT,
        b"[D": K_LEFT,
        b"[5~": K_PPAGE,
        b"[6~": K_NPAGE,
        b"[H": K_HOME,
        b"[F": K_END,
    }
    return mapping.get(buf)


def _decode_input(buf: bytes) -> tuple[list, bytes]:
    """Split raw stdin bytes into complete key events plus any trailing
    partial escape sequence (held back until more bytes arrive or the grace
    period in run() expires). Mouse sequences are swallowed."""
    keys: list = []
    i, n = 0, len(buf)
    while i < n:
        b = buf[i]
        if b == 0x1B:  # ESC
            j = i + 1
            if j >= n or buf[j] not in (0x5B, 0x4D):
                break  # hold back; the grace period resolves a bare ESC
            if buf[j] == 0x5B:  # CSI: ESC [ params final(0x40-0x7E)
                k = j + 1
                while k < n and not (0x40 <= buf[k] <= 0x7E):
                    k += 1
                if k >= n:
                    break  # incomplete CSI, wait for more bytes
                key = _decode_esc_body(buf[i + 1:k + 1])
                if key is not None:
                    keys.append(key)
                i = k + 1
                continue
            if j + 3 >= n:  # X10 mouse: ESC M + 3 bytes, incomplete
                break
            i = j + 4  # swallow the mouse sequence
            continue
        if b in (0x0D, 0x0A):
            keys.append(K_ENTER)
            i += 1
        elif b in (0x7F, 0x08):
            keys.append(K_BACKSPACE)
            i += 1
        elif b == K_CTRL_C:
            keys.append(K_CTRL_C)
            i += 1
        elif b == K_CTRL_Q:
            keys.append(K_CTRL_Q)
            i += 1
        elif b == K_CTRL_T:
            keys.append(K_CTRL_T)
            i += 1
        elif 0x20 <= b < 0x7F:
            keys.append(b)
            i += 1
        else:
            i += 1  # swallow other control bytes
    return keys, buf[i:]


class _InputDecoder:
    """Sits between raw stdin bytes and key events. Understands bracketed
    paste (ESC[200~ ... ESC[201~): pasted text is delivered as a single
    ("paste", text) event instead of being interpreted as keys, so an Enter
    inside a paste does not submit. Terminals without the feature simply
    never send the markers and pastes degrade to the old behavior."""

    def __init__(self) -> None:
        self.buf = b""  # undecoded bytes (partial sequences / mid-paste)
        self.in_paste = False
        self.paste_buf = b""
        self.pend_start = 0.0

    def feed(self, data: bytes, now: float) -> list[tuple[str, object]]:
        """Feed raw bytes; returns ("key", int) and ("paste", str) events."""
        events: list[tuple[str, object]] = []
        self.buf += data
        while True:
            if self.in_paste:
                end = self.buf.find(b"\x1b[201~")
                if end == -1:
                    break  # wait for the closing marker
                self.paste_buf += self.buf[:end]
                self.buf = self.buf[end + 6:]
                self.in_paste = False
                events.append(("paste", _decode_text(self.paste_buf)))
                self.paste_buf = b""
                continue
            i = self.buf.find(b"\x1b[200~")
            if i != -1:
                head, rest = self.buf[:i], self.buf[i + 6:]
                keys, rem = self._decode_keys(head, now)
                events.extend(("key", k) for k in keys)
                self.buf = rem + rest  # hold any partial tail; keep the paste payload
                self.in_paste = True
                continue
            keys, rem = self._decode_keys(self.buf, now)
            events.extend(("key", k) for k in keys)
            self.buf = rem
            break
        return events

    def _decode_keys(self, data: bytes, now: float) -> tuple[list[int], bytes]:
        keys, rem = _decode_input(data)
        if rem:
            if self.pend_start == 0.0:
                self.pend_start = now
            elif now - self.pend_start > 0.03 or len(rem) >= 16:
                # Grace period expired: a bare ESC (or a truncated sequence
                # that will never complete).
                if rem == b"\x1b":
                    keys.append(K_ESC)
                rem = b""
        else:
            self.pend_start = 0.0
        return keys, rem


def _decode_text(data: bytes) -> str:
    """Decode pasted bytes; all line endings become \\n, other control
    characters are dropped (a paste is literal text, not key input)."""
    text = data.decode("utf-8", "replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(ch for ch in text if ch == "\n" or ord(ch) >= 32)


def _insert_text(state: UIState, text: str) -> None:
    """Insert literal text at the cursor (paste / multi-char input)."""
    state.input_buf = state.input_buf[: state.cursor] + text + state.input_buf[state.cursor :]
    state.cursor += len(text)


def _cursor_pos(state: UIState, height: int, width: int) -> tuple[int, int]:
    """(row, col) of the cursor for the current input; the prompt block
    occupies the bottom input-block rows (long lines wrap), "> " on the
    first line."""
    lines = state.input_buf.split("\n") if state.input_buf else [""]
    chunk_w = max(1, width - 2)
    li, start = 0, 0
    for i, ln in enumerate(lines):
        if state.cursor <= start + len(ln):
            li = i
            break
        start += len(ln) + 1
    col_in_line = state.cursor - start
    rows_before = _input_block_height(lines[:li], width)
    row = height - _input_block_height(lines, width) + rows_before + col_in_line // chunk_w
    return row, 2 + col_in_line % chunk_w


def _cell_len(text: str) -> int:
    """Display cells occupied by text (wide/fullwidth chars take 2). ncurses
    advances the cursor by cells, so tracking columns by len() would drift
    after any emoji and overwrite the following character."""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def _render(stdscr, state: UIState) -> None:
    try:
        height, width = stdscr.getmaxyx()
    except Exception:
        return
    rows = render_frame(state, height, width)
    try:
        # Clear the virtual screen first so rows that shrink (or blank out)
        # do not leave stale tails from the previous frame.
        stdscr.erase()
    except curses.error:
        pass
    for r, row in enumerate(rows[:height]):
        col = 0
        for text, style in row:
            if col >= width - 1 or not text:
                continue
            try:
                stdscr.addnstr(r, col, text, (width - 1) - col, _STYLE_ATTRS.get(style, 0))
            except curses.error:
                break
            col += _cell_len(text)
    try:
        row, col = _cursor_pos(state, height, width)
        stdscr.move(row, min(col, width - 1))
        stdscr.refresh()
    except curses.error:
        pass


async def run(bus: Bus, context_size: int | Callable[[], int | None] = 0) -> None:
    """Run the curses UI until cancelled or quit.

    All curses I/O happens on the event-loop thread; key input is read raw
    from stdin via loop.add_reader (no worker thread), so stdscr is never
    touched concurrently and ncurses' screen bookkeeping stays consistent."""
    _diagnose_terminal()
    stdscr = _curses_setup()
    try:
        logger.info("terminal startup: curses window size %sx%s", *stdscr.getmaxyx()[::-1])
    except Exception:
        pass
    state = UIState(context_size=context_size)
    fd = sys.stdin.fileno()
    os.set_blocking(fd, False)
    loop = asyncio.get_running_loop()
    inbox: dict = {"data": b"", "eof": False}

    def _on_readable() -> None:
        try:
            data = os.read(fd, 4096)
        except (BlockingIOError, OSError):
            return
        if data == b"":
            inbox["eof"] = True
            return
        inbox["data"] += data

    loop.add_reader(fd, _on_readable)
    decoder = _InputDecoder()
    last_spin = 0.0
    # Bracketed paste: the terminal wraps pastes in ESC[200~ ... ESC[201~ so
    # an Enter inside a paste is literal text, not a submit. Terminals
    # without the feature never send the markers (paste degrades to typing).
    sys.stdout.write("\x1b[?2004h")
    sys.stdout.flush()
    try:
        while True:
            await asyncio.sleep(0.033)  # tick + await point for cancellation
            if inbox["eof"]:
                break  # stdin closed
            events = decoder.feed(inbox["data"], time.monotonic())
            inbox["data"] = b""
            try:
                h, w = struct.unpack("HHHH", fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8))[:2]
                if (h, w) != stdscr.getmaxyx():
                    curses.resizeterm(h, w)
            except Exception:
                pass
            height, width = stdscr.getmaxyx()
            input_lines = state.input_buf.split("\n") if state.input_buf else [""]
            view_h = max(1, height - 1 - _input_block_height(input_lines, width))
            for kind, value in events:
                if kind == "paste":
                    _insert_text(state, str(value))
                    continue
                action, payload = apply_key(state, int(value), view_h=view_h, total=count_lines(state, width))
                if action == "submit":
                    msg_id = uuid.uuid4().hex[:8]
                    submit_message(state, payload)
                    bus.post(UserMessage(USER_ID, payload, msg_id))
                elif action == "interrupt":
                    bus.post(Interrupt(USER_ID))
                elif action == "quit":
                    bus.post(Shutdown())
                    return
            while not bus.outbound.empty():
                apply_event(state, bus.outbound.get_nowait())
            if state.busy:
                now = time.monotonic()
                if now - last_spin >= 0.2:  # 5fps: a full moon cycle in ~1.6s
                    state.frame += 1
                    last_spin = now
            _render(stdscr, state)
    except asyncio.CancelledError:
        pass
    finally:
        try:
            loop.remove_reader(fd)
        except Exception:
            pass
        try:
            sys.stdout.write("\x1b[?2004l")  # disable bracketed paste
            sys.stdout.flush()
        except Exception:
            pass
        try:
            curses.endwin()
        except Exception:
            pass


def stop() -> None:
    # No worker threads to stop; run()'s finally block restores the terminal.
    pass
