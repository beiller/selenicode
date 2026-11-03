"""Bare-bones curses terminal client (terminal v2).

Full-screen TUI with O(visible-lines) render cost: a scrollback viewport, a
pinned status bar and a pinned prompt line. No widgets, no markdown engine -
every message is plain ANSI-styled text, so hundreds of messages cost the same
as dozens.

Layout (H x W):
    rows 0 .. H-3   scrollback viewport (window into all message lines)
    row   H-2       status bar (dim): spinner/state + token usage
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
import select
import struct
import sys
import termios
import textwrap
import time
import uuid

from bus import Bus, History, Interrupt, Shutdown, Token, TurnDone, UsageReport, UserMessage

logger = logging.getLogger()

USER_ID = "0"

# key codes (curses KEY_* values; enter/backspace are raw bytes)
K_ENTER = 10
K_BACKSPACE = 127
K_ESC = 27
K_CTRL_C = 3
K_CTRL_Q = 17
K_UP = 259
K_DOWN = 260
K_LEFT = 261
K_RIGHT = 262
K_HOME = 270
K_END = 271
K_PPAGE = 339
K_NPAGE = 338

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# style ids (mapped to curses attrs in _curses_setup; pure layer stays curses-free)
STY_PLAIN = 0
STY_DIM = 1
STY_YOU = 2
STY_TOOL = 3
STY_INFO = 4

_ROLE_STYLE = {
    "thinking": STY_DIM,
    "assistant": STY_PLAIN,
    "tool": STY_TOOL,
    "info": STY_INFO,
    "error": STY_INFO,
}
_ROLE_PREFIX = {
    "thinking": "(thinking) ",
    "assistant": "",
    "user": "You: ",
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

    def __init__(self, context_size: int = 0) -> None:
        self.blocks: list[Block] = []
        self.active: Block | None = None
        self.cache: dict[int, tuple[int, list]] = {}  # id(block) -> (width, lines)
        # Distance of the viewport top from the bottom-most position; 0 means
        # pinned to the bottom (auto-follows new content for free).
        self.scroll_up = 0
        self.input_buf = ""
        self.cursor = 0
        self.status = "ready"
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
        if cached is not None and cached[0] == width:
            return cached[1]
    lines: list[list[tuple[str, int]]] = []
    wrapped = wrap_text(block.raw, max(1, width - 2))
    for i, ln in enumerate(wrapped):
        if i == 0 and block.prefix:
            lines.append([(block.prefix, block.style), (ln, block.style)])
        elif i > 0:
            lines.append([("  " + ln, block.style)])
        else:
            lines.append([(ln, block.style)])
    if block is not state.active:
        state.cache[key] = (width, lines)
    return lines


def render_frame(state: UIState, height: int, width: int) -> list[list[tuple[str, int]]]:
    """Build the full screen as rows of (text, style) segments. Pure."""
    view_h = max(1, height - 2)
    all_lines: list[list[tuple[str, int]]] = []
    for block in state.blocks:
        all_lines.extend(_block_lines(state, block, width))
    if state.active is not None:
        all_lines.extend(_block_lines(state, state.active, width))

    total = len(all_lines)
    top = max(0, total - view_h - state.scroll_up)

    rows: list[list[tuple[str, int]]] = []
    for i in range(view_h):
        idx = top + i
        rows.append(all_lines[idx] if idx < total else [("", STY_PLAIN)])

    # DEBUG: status bar is frozen (no spinner, no updates) while display issues are tracked down.
    tok = f" tokens: {state.tokens}/{state.context_size}" if state.context_size else ""
    rows.append([(f"status: {state.status}{tok}", STY_DIM)])

    prompt: list[tuple[str, int]] = [("> ", STY_YOU), (state.input_buf, STY_PLAIN)]
    if not state.input_buf:
        prompt.append(("  ↑↓/pgup scroll · ctrl+c interrupt · ctrl+q quit", STY_DIM))
    rows.append(prompt)
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


def apply_key(state: UIState, key, view_h: int = 24) -> tuple[str, str]:
    """Apply one key event. Returns (action, payload); action is one of
    "", "submit" (payload = text), "interrupt", "quit"."""
    if isinstance(key, tuple):  # ("wheel", delta): positive = older content
        _scroll(state, key[1])
        return "", ""
    if key == K_CTRL_Q:
        return "quit", ""
    if key == K_CTRL_C:
        if state.busy:
            return "interrupt", ""
        state.input_buf = ""
        state.cursor = 0
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
        _scroll(state, 3)
        return "", ""
    if key == K_DOWN:
        _scroll(state, -3)
        return "", ""
    if key == K_LEFT:
        state.cursor = max(0, state.cursor - 1)
        return "", ""
    if key == K_RIGHT:
        state.cursor = min(len(state.input_buf), state.cursor + 1)
        return "", ""
    if key == K_PPAGE:
        _scroll(state, view_h - 1)
        return "", ""
    if key == K_NPAGE:
        _scroll(state, -(view_h - 1))
        return "", ""
    if key == K_HOME:
        state.scroll_up = 10**9  # clamped by the viewport math
        return "", ""
    if key == K_END:
        state.scroll_up = 0
        return "", ""
    if 32 <= key < 127:
        state.input_buf = state.input_buf[: state.cursor] + chr(key) + state.input_buf[state.cursor :]
        state.cursor += 1
        return "", ""
    return "", ""


def _scroll(state: UIState, delta: int) -> None:
    """Scroll the viewport; positive delta moves toward older content.
    Clamped at the bottom, so scrolling down re-pins automatically."""
    state.scroll_up = max(0, state.scroll_up + delta)


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
                out.append(Block("assistant", "", content, STY_PLAIN))
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
        case TurnDone(_):
            _finalize_active(state)
            state.busy = False
        case UsageReport(_, _):
            pass  # DEBUG: token updates to the status bar are frozen while debugging
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
    logger.info("terminal_v2 startup: %s", env or "no terminal env vars set")
    logger.info("terminal_v2 startup: TERM=%r terminfo file: %s", term, str(terminfo_file) if terminfo_file else "NOT FOUND in plain-text db (may be compiled .db only)")


def _curses_setup() -> "curses._CursesWindow":
    stdscr = curses.initscr()
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_MAGENTA, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        _STYLE_ATTRS.update(
            {
                STY_PLAIN: 0,
                STY_DIM: curses.A_DIM,
                STY_YOU: curses.A_BOLD | curses.color_pair(1),
                STY_TOOL: curses.color_pair(2),
                STY_INFO: curses.A_BOLD | curses.color_pair(3),
            }
        )
    except curses.error:
        _STYLE_ATTRS.update({STY_PLAIN: 0, STY_DIM: 0, STY_YOU: curses.A_BOLD, STY_TOOL: 0, STY_INFO: curses.A_BOLD})
    curses.cbreak()
    curses.noecho()
    stdscr.nodelay(True)
    # NOTE: mouse reporting is deliberately NOT enabled. This system's ncurses
    # enables SGR mouse mode but fails to parse the sequences, so every click/
    # wheel event leaks into getch() as raw garbage characters. Scroll with
    # up/down, PgUp/PgDn, Home/End instead.
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
        elif 0x20 <= b < 0x7F:
            keys.append(b)
            i += 1
        else:
            i += 1  # swallow other control bytes
    return keys, buf[i:]


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
            col += len(text)
    try:
        stdscr.move(height - 1, min(2 + state.cursor, width - 1))
        stdscr.refresh()
    except curses.error:
        pass


async def run(bus: Bus, context_size: int = 0) -> None:
    """Run the curses UI until cancelled or quit."""
    global _stop_evt
    _diagnose_terminal()
    stdscr = _curses_setup()
    try:
        logger.info("terminal_v2 startup: curses window size %sx%s", *stdscr.getmaxyx()[::-1])
    except Exception:
        pass
    state = UIState(context_size=context_size)
    key_q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    _stop_evt = threading.Event()
    reader = threading.Thread(target=_key_reader, args=(stdscr, key_q, loop, _stop_evt), daemon=True)
    reader.start()
    try:
        while True:
            try:
                with _curses_lock:
                    height, _width = stdscr.getmaxyx()
            except Exception:
                height = 24
            key = await _next_key(key_q, 0.033)
            if key is not None and key != K_RESIZE:
                action, payload = apply_key(state, key, view_h=max(1, height - 2))
                if action == "submit":
                    msg_id = uuid.uuid4().hex[:8]
                    state.busy = True
                    bus.post(UserMessage(USER_ID, payload, msg_id))
                elif action == "interrupt":
                    bus.post(Interrupt(USER_ID))
                elif action == "quit":
                    bus.post(Shutdown())
                    return
            n = 0
            while not bus.outbound.empty():
                apply_event(state, bus.outbound.get_nowait())
                n += 1
            logger.info("DEBUG tick: events=%d blocks=%d active=%r", n, len(state.blocks), (state.active.raw[:20] if state.active else None))
            _render(stdscr, state)
    except asyncio.CancelledError:
        pass
    finally:
        if _stop_evt is not None:
            _stop_evt.set()
        reader.join(timeout=1)
        try:
            curses.endwin()
        except Exception:
            pass


def stop() -> None:
    if _stop_evt is not None:
        _stop_evt.set()
