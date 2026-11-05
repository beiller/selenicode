"""Bare-bones curses terminal client.

Full-screen TUI with O(visible-lines) render cost: a scrollback viewport, a
pinned status bar and a pinned prompt line. No widgets, no markdown engine -
every message is plain ANSI-styled text, so hundreds of messages cost the same
as dozens.

Layout (H x W):
    rows 0 .. H-3   scrollback viewport (window into all message lines)
    row   H-2       status bar (bold green): spinner/state + token usage
    row   H-1       prompt line: "> " + input + cursor

Keys: Enter submit, Backspace/Delete/Ctrl+Arrow/Left/Right edit, Home/End jump
to line start/end, Up/Down recall user history (scroll when there is none),
PgUp/PgDn scroll, Ctrl+C interrupt a running turn (clears input when idle),
Ctrl+Q quit. Mouse: left-drag selects (highlighted); release copies the
selection to the clipboard (local tool if installed, else OSC 52); middle-
click pastes; wheel scrolls.
"""

import asyncio
import base64
import curses
import fcntl
import json
import logging
import os
import pathlib
import re
import shutil
import struct
import subprocess
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
K_DELETE = 263
K_CTRL_LEFT = 264
K_CTRL_RIGHT = 265
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
# inline markdown (assistant messages only)
STY_BOLD = 8
STY_ITALIC = 9
STY_CODE = 10
STY_NUM = 11  # fenced code line numbers: gray color pair (A_DIM alone is often ignored)

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
        self.hist_index: int | None = None  # position in _user_history while navigating
        self.hist_stash = ""  # the buffer saved when Up started navigation
        self.sel_anchor: tuple[int, int] | None = None  # (row, col) screen coords where the drag started
        self.sel_extent: tuple[int, int] | None = None  # live (row, col) of the drag
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


def _first_line(para: str, w: int) -> tuple[str, str]:
    """Split para into (line of at most w chars on a word boundary, rest)."""
    if len(para) <= w:
        return para, ""
    cut = para.rfind(" ", 0, w + 1)
    first = para[:cut].rstrip() if cut > 0 else ""
    rest = para[cut:].lstrip() if cut > 0 else para
    if not first:  # all-whitespace prefix or no word boundary: hard cut
        return para[:w], para[w:]
    return first, rest


def _wrap_block(text: str, first_w: int, rest_w: int) -> list[str]:
    """Wrap text; the very first line is limited to first_w chars (the block
    prefix sits on it), all following lines to rest_w (2-cell indent)."""
    if not text:
        return [""]
    out: list[str] = []
    for i, para in enumerate(text.split("\n")):
        if i == 0 and para:
            first, rest = _first_line(para, first_w)
            out.append(first)
            if rest:
                out.extend(textwrap.wrap(rest, max(1, rest_w)))
        else:
            out.extend(textwrap.wrap(para, max(1, rest_w)) or [""])
    return out


# --- poor man's markdown (assistant messages only) -------------------------


def _md_edge_alnum(line: str, idx: int) -> bool:
    return 0 <= idx < len(line) and line[idx].isalnum()


def _md_find_spans(line: str) -> list[tuple[int, int, int]]:
    """(start, end, style) marker-inclusive regions of a line. Unbalanced or
    ambiguous markers (e.g. "2*3*4", "O(n * m)", my_var) stay literal."""
    spans: list[tuple[int, int, int]] = []
    for m in re.finditer(r"`([^`\n]+)`", line):
        spans.append((m.start(), m.end(), STY_CODE))
    i = 0
    while True:  # **bold**
        s = line.find("**", i)
        if s == -1:
            break
        e = line.find("**", s + 2)
        if (e != -1
                and not any(a <= s < b or a <= e < b or a >= s and b <= e + 2 for a, b, _ in spans)):
            content = line[s + 2:e]
            if (content and not content[0].isspace() and not content[-1].isspace()
                    and not _md_edge_alnum(line, s - 1) and not _md_edge_alnum(line, e + 2)):
                spans.append((s, e + 2, STY_BOLD))
                i = e + 2
                continue
        i = s + 2
    for ch in ("*", "_"):  # *italic* / _italic_
        i = 0
        while True:
            s = line.find(ch, i)
            if s == -1:
                break
            e = line.find(ch, s + 1)
            if (e != -1
                    and not any(a <= s < b or a <= e < b or a >= s and b <= e + 1 for a, b, _ in spans)):
                content = line[s + 1:e]
                if (content and ch not in content and not content[0].isspace()
                        and not content[-1].isspace()
                        and not _md_edge_alnum(line, s - 1) and not _md_edge_alnum(line, e + 1)):
                    spans.append((s, e + 1, STY_ITALIC))
            i = s + 1
    return sorted(spans)


def _md_inline_words(line: str) -> list[tuple[str, int]] | None:
    """(word, style) pairs for a line containing markdown; None when it has
    no markers (caller falls back to the plain wrap)."""
    spans = _md_find_spans(line)
    if not spans:
        return None
    words: list[tuple[str, int]] = []
    pos = 0
    for start, end, style in spans:
        cut = 2 if style == STY_BOLD else 1
        for w in line[pos:start].split():
            words.append((w, STY_PLAIN))
        for w in line[start + cut:end - cut].split():
            words.append((w, style))
        pos = end
    for w in line[pos:].split():
        words.append((w, STY_PLAIN))
    return words or None


def _split_cells(text: str, width: int) -> list[str]:
    """Split text into chunks of at most `width` cells (wide chars intact)."""
    out: list[str] = []
    cur = ""
    w = 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if cur and w + cw > width:
            out.append(cur)
            cur, w = "", 0
        cur += ch
        w += cw
    if cur:
        out.append(cur)
    return out


def _wrap_tokens(words: list[tuple[str, int]], first_w: int, rest_w: int | None = None) -> list[list[tuple[str, int]]]:
    """Wrap (word, style) pairs: line 0 is limited to first_w, all following
    lines to rest_w (first_w when rest_w is None). Words longer than the line
    are hard-split, keeping their style on every chunk."""
    if rest_w is None:
        rest_w = first_w
    lines: list[list[tuple[str, int]]] = []
    cur: list[tuple[str, int]] = []
    cur_w = 0

    def flush() -> None:
        nonlocal cur, cur_w
        if cur or not lines:
            lines.append(cur)
        cur, cur_w = [], 0

    for word, style in words:
        w = _cell_len(word)
        limit = first_w if not lines else rest_w
        if w > limit:
            if cur_w:
                flush()
            chunks = _split_cells(word, max(1, limit))
            lines.extend([(c, style) for c in chunks[:-1]])
            cur = [(chunks[-1], style)]
            cur_w = _cell_len(chunks[-1])
            continue
        if cur_w and cur_w + 1 + w > limit:
            flush()
        if cur_w:
            cur.append((" ", STY_PLAIN))  # separators stay unstyled (no bg color)
        cur.append((word, style))
        cur_w += (0 if cur_w == 0 else 1) + w
    flush()
    return lines


_MD_LIST = re.compile(r"^(\s*)([-*+]|\d{1,2}[.)])\s+(.*)$")


def _md_list_item(indent: str, marker: str, content: str, w: int) -> list[list[tuple[str, int]]]:
    """One list item as styled lines with a hanging indent (2 cells per
    nesting level, capped)."""
    level = min(len(indent) // 2, 3)
    bullet = "\u2022 " if not marker[0].isdigit() else f"{marker} "
    prefix = "  " * level + bullet
    hang = _cell_len(prefix)
    cw = max(1, w - hang)
    words = _md_inline_words(content)
    if words is not None:
        wrapped = _wrap_tokens(words, cw)
    else:
        wrapped = textwrap.wrap(content, cw) or [""]
    out: list[list[tuple[str, int]]] = []
    for i, ln in enumerate(wrapped):
        lead = (prefix if i == 0 else " " * hang, STY_PLAIN)
        out.append([lead] + ([(ln, STY_PLAIN)] if isinstance(ln, str) else ln))
    return out


def _md_table_cells(ln: str) -> list[str] | None:
    s = ln.strip()
    if not s.startswith("|") or s.count("|") < 2:
        return None
    return [c.strip() for c in s.strip("|").split("|")]


def _md_is_sep(cells: list[str]) -> bool:
    joined = "".join(cells)
    return bool(joined) and set(joined) <= set("-: ") and "-" in joined


def _truncate_cells(text: str, width: int) -> str:
    """Truncate to `width` cells (wide-char safe), ellipsizing."""
    if _cell_len(text) <= width:
        return text
    out = ""
    w = 0
    for ch in text:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > max(1, width - 1):
            break
        out += ch
        w += cw
    return out + "\u2026"


def _md_table(rows: list[list[str]], w: int, cell_style: int = STY_PLAIN, header_style: int = STY_BOLD) -> list[list[tuple[str, int]]] | None:
    """Aligned table (rows[0] = header, separator row already dropped); None
    when it cannot fit even with capped columns (caller renders raw lines)."""
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    col_w = [max(_cell_len(r[i]) for r in rows) for i in range(ncols)]
    if sum(col_w) + 2 * ncols > w:  # "| a | b |" costs sum + 2 per column
        cap = max(1, (w - 2 * ncols) // ncols)
        col_w = [min(cw, cap) for cw in col_w]
        if sum(col_w) + 2 * ncols > w:
            return None
    out: list[list[tuple[str, int]]] = []
    for i, r in enumerate(rows):
        segs: list[tuple[str, int]] = [("| ", cell_style)]
        for j, c in enumerate(r):
            c = _truncate_cells(c, col_w[j])
            style = header_style if i == 0 else cell_style
            segs.append((c, style))
            segs.append((" " * (col_w[j] - _cell_len(c)) + (" | " if j < ncols - 1 else " |"), cell_style))
        out.append(segs)
    return out


def _md_block_lines(raw: str, first_w: int, rest_w: int) -> list[list[tuple[str, int]]] | None:
    """Poor man's markdown for one assistant message: **bold**, *italic*,
    `code`, ``` fences, bullet/numbered lists and |tables|. Returns styled
    lines (line 0 limited to first_w, continuations indented 2 cells), or
    None when nothing markdown-ish is present (caller uses _wrap_block)."""
    out: list[list[tuple[str, int]]] = []
    any_md = False
    lines = raw.split("\n")
    i, n = 0, len(lines)
    while i < n:
        ln = lines[i]
        # fenced code block (no inline parsing inside; unclosed at EOF is fine)
        if ln.strip().startswith("```"):
            any_md = True
            out.append([(ln.strip(), STY_CODE)])
            i += 1
            end = i
            while end < n and not lines[end].strip().startswith("```"):
                end += 1
            num_w = max(3, len(str(end - i)))  # dim line numbers, min width 3
            pad = " " * (num_w + 1)
            cw = max(1, rest_w - num_w - 1)
            num = 0
            while i < end:
                # tables inside fences still get aligned (basic handling), but
                # stay in code style - agents fence data tables all the time
                cells = _md_table_cells(lines[i])
                if cells is not None and i + 1 < end:
                    sep = _md_table_cells(lines[i + 1])
                    if sep is not None and _md_is_sep(sep):
                        rows = [cells]
                        j = i + 2
                        while j < end:
                            rc = _md_table_cells(lines[j])
                            if rc is None:
                                break
                            rows.append(rc)
                            j += 1
                        table = _md_table(rows, cw, STY_CODE, STY_CODE)
                        if table is None:
                            for k in range(i, j):  # too wide: plain code lines
                                num += 1
                                first = True
                                for c in textwrap.wrap(lines[k], cw) or [""]:
                                    lead = (f"{num:>{num_w}} ", STY_NUM) if first else (pad, STY_CODE)
                                    out.append([lead, (c, STY_CODE)])
                                    first = False
                        else:
                            for tln in table:
                                num += 1
                                out.append([(f"{num:>{num_w}} ", STY_NUM)] + tln)
                        i = j
                        continue
                num += 1
                first = True
                for c in textwrap.wrap(lines[i], cw) or [""]:
                    lead = (f"{num:>{num_w}} ", STY_NUM) if first else (pad, STY_CODE)
                    out.append([lead, (c, STY_CODE)])
                    first = False
                i += 1
            if i < n:
                out.append([(lines[i].strip(), STY_CODE)])
                i += 1
            continue
        # table: a |row| followed by a |---| separator row
        cells = _md_table_cells(ln)
        if cells is not None and i + 1 < n:
            sep = _md_table_cells(lines[i + 1])
            if sep is not None and _md_is_sep(sep):
                rows = [cells]
                j = i + 2
                while j < n:
                    rc = _md_table_cells(lines[j])
                    if rc is None:
                        break
                    rows.append(rc)
                    j += 1
                table = _md_table(rows, max(1, first_w if i == 0 else rest_w))
                if table is None:
                    for k in range(i, j):  # too wide: render raw
                        out.append([(lines[k], STY_PLAIN)])
                else:
                    any_md = True
                    out.extend(table)
                i = j
                continue
        m = _MD_LIST.match(ln)
        if m:
            any_md = True
            out.extend(_md_list_item(m.group(1), m.group(2), m.group(3), max(1, first_w if i == 0 else rest_w)))
            i += 1
            continue
        words = _md_inline_words(ln)
        if words is None:
            # plain line: wrap exactly like _wrap_block does
            if i == 0 and ln:
                first, rest = _first_line(ln, max(1, first_w))
                out.append([(first, STY_PLAIN)])
                for c in textwrap.wrap(rest, max(1, rest_w)):
                    out.append([("  " + c, STY_PLAIN)])
            else:
                for c in (textwrap.wrap(ln, max(1, rest_w)) or [""]):
                    out.append([("  " + c, STY_PLAIN)])
        else:
            any_md = True
            wrapped = _wrap_tokens(words, max(1, first_w), max(1, rest_w))
            out.append(wrapped[0])
            for tl in wrapped[1:]:
                out.append([("  ", STY_PLAIN)] + tl)
        i += 1
    return out if any_md else None


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
        text = f"(thinking) (collapsed) {len(block.raw)} chars: {preview}"
        for i, ln in enumerate(wrap_text(text, max(1, width - 3))):
            lines.append([(("  " if i else "") + ln, block.style)])
    else:
        # Usable row width is width-1 cells (the last column is reserved); the
        # first line carries the prefix, continuation lines a 2-cell indent.
        md = None
        if block.role == "assistant":
            md = _md_block_lines(block.raw, max(1, width - 1 - _cell_len(block.prefix)), max(1, width - 3))
        if md is not None:
            for i, segs in enumerate(md):
                if i == 0 and block.prefix:
                    lines.append([(block.prefix, block.style)] + segs)
                else:
                    lines.append(segs)
        else:
            wrapped = _wrap_block(block.raw, max(1, width - 1 - _cell_len(block.prefix)), max(1, width - 3))
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
    chunk_w = max(1, width - 3)
    rows: list[list[tuple[str, int]]] = []
    for i, ln in enumerate(input_lines):
        chunks = [ln[j:j + chunk_w] for j in range(0, len(ln), chunk_w)] or [""]
        for k, chunk in enumerate(chunks):
            prefix = ("> ", STY_YOU) if (i == 0 and k == 0) else ("  ", STY_PLAIN)
            rows.append([prefix, (chunk, STY_PLAIN)])
    return rows


def _input_block_height(input_lines: list[str], width: int) -> int:
    """Rows the input block occupies (long lines wrap)."""
    chunk_w = max(1, width - 3)
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
        tok = f" · tokens: {state.tokens}/{cs} ({pct}%)"
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
        rows[-1].append(("  ↑↓ history · drag copies · mid-click pastes · pgup/pgdn scroll · ctrl+c interrupt · ctrl+q quit · ctrl+t show thinking · /help for commands", STY_DIM))
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


def _line_bounds(buf: str, cursor: int) -> tuple[int, int]:
    """(start, end) of the line containing cursor in buf."""
    start = buf.rfind("\n", 0, cursor) + 1
    end = buf.find("\n", cursor)
    return start, len(buf) if end == -1 else end


def _user_history(state: UIState) -> list[str]:
    """User-submitted messages in order (for Up/Down recall)."""
    return [b.raw for b in state.blocks if b.role == "user"]


def _end_hist_nav(state: UIState) -> None:
    """Leave history navigation (the line was edited or cleared)."""
    if state.hist_index is not None:
        state.hist_index = None
        state.hist_stash = ""


def _word_jump(buf: str, cursor: int, forward: bool) -> int:
    """Move to the next/previous word boundary (readline semantics)."""
    def is_word(c: str) -> bool:
        return c.isalnum() or c == "_"
    n = len(buf)
    if forward:
        i = cursor
        while i < n and not is_word(buf[i]):
            i += 1
        while i < n and is_word(buf[i]):
            i += 1
        return i
    i = cursor
    while i > 0 and not is_word(buf[i - 1]):
        i -= 1
    while i > 0 and is_word(buf[i - 1]):
        i -= 1
    return i


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
        _end_hist_nav(state)
        return "", ""
    if key == K_CTRL_T:
        state.show_thinking = not state.show_thinking
        return "", ""
    if key in (K_ENTER, 13):
        text = state.input_buf.strip()
        state.input_buf = ""
        state.cursor = 0
        _end_hist_nav(state)
        if not text:
            return "", ""
        return "submit", text
    if key == K_ESC:
        state.input_buf = ""
        state.cursor = 0
        _end_hist_nav(state)
        return "", ""
    if key in (K_BACKSPACE, 8):
        _end_hist_nav(state)
        if state.cursor > 0:
            state.input_buf = state.input_buf[: state.cursor - 1] + state.input_buf[state.cursor :]
            state.cursor -= 1
        return "", ""
    if key == K_UP:
        hist = _user_history(state)
        if hist:
            if state.hist_index is None:
                state.hist_stash = state.input_buf
                state.hist_index = len(hist) - 1
            else:
                state.hist_index = max(0, state.hist_index - 1)
            state.input_buf = hist[state.hist_index]
            state.cursor = len(state.input_buf)
        else:
            _scroll(state, 3, total, view_h)  # no history yet: scroll as before
        return "", ""
    if key == K_DOWN:
        hist = _user_history(state)
        if state.hist_index is not None and hist:
            state.hist_index += 1
            if state.hist_index >= len(hist):
                state.input_buf = state.hist_stash
                state.cursor = len(state.input_buf)
                state.hist_index = None
                state.hist_stash = ""
            else:
                state.input_buf = hist[state.hist_index]
                state.cursor = len(state.input_buf)
        else:
            _scroll(state, -3, total, view_h)  # not navigating: scroll as before
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
        state.cursor = _line_bounds(state.input_buf, state.cursor)[0]
        return "", ""
    if key == K_END:
        start, end = _line_bounds(state.input_buf, state.cursor)
        if 0 <= state.cursor < len(state.input_buf) and state.input_buf[state.cursor] == "\n":
            end = min(len(state.input_buf), state.cursor + 1)
        state.cursor = end
        return "", ""
    if key == K_CTRL_LEFT:
        state.cursor = _word_jump(state.input_buf, state.cursor, False)
        return "", ""
    if key == K_CTRL_RIGHT:
        state.cursor = _word_jump(state.input_buf, state.cursor, True)
        return "", ""
    if key == K_DELETE:
        _end_hist_nav(state)
        if state.cursor < len(state.input_buf):
            state.input_buf = state.input_buf[: state.cursor] + state.input_buf[state.cursor + 1 :]
        return "", ""
    if 32 <= key < 127:
        _end_hist_nav(state)
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
    _clear_selection(state)  # screen coords would drift with the content
    bottom = max(0, total - view_h)
    if state.view_top is None:
        state.view_top = bottom  # first scroll up anchors at the current bottom
    state.view_top = max(0, min(state.view_top - delta, bottom))
    if state.view_top == bottom:
        state.view_top = None


def _clear_selection(state: UIState) -> None:
    state.sel_anchor = None
    state.sel_extent = None


def _selection_range(state: UIState) -> tuple[int, int, int, int] | None:
    """Normalized (r1, r2, c1, c2) of the active drag; None while not
    dragging (a press alone is not a selection yet)."""
    if state.sel_anchor is None or state.sel_extent is None:
        return None
    ar, ac = state.sel_anchor
    er, ec = state.sel_extent
    return (min(ar, er), max(ar, er), min(ac, ec), max(ac, ec))


def _apply_mouse(state: UIState, event: str, col1: int, row1: int, height: int, width: int) -> None:
    """Mouse event in SGR 1-based screen coords. Left press starts a
    selection, drag extends it (highlighted by _render), release copies the
    region to the clipboard; a click without drag cancels. Middle-button
    press pastes the system clipboard at the cursor."""
    col = max(0, min(col1 - 1, width - 2))
    row = max(0, min(row1 - 1, height - 1))
    if event == "middle":
        _end_hist_nav(state)
        text = _read_clipboard()
        if text:
            _insert_text(state, text)
        return
    if event == "press":
        state.sel_anchor = (row, col)
        state.sel_extent = None
    elif event == "drag" and state.sel_anchor is not None:
        state.sel_extent = (row, col)
    elif event == "release":
        if state.sel_anchor is not None and state.sel_extent is not None:
            _copy_selection(state, height, width)
        _clear_selection(state)


def _row_cells(segs: list[tuple[str, int]], width: int) -> list[tuple[int, str]]:
    """(start_col, char) of the visible cells of a row, clipped exactly like
    _render (last column reserved; a wide char straddling the edge drops)."""
    out: list[tuple[int, str]] = []
    col = 0
    for text, _ in segs:
        if col >= width - 1 or not text:
            continue
        for ch in text:
            wch = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if col + wch > width - 1:
                break
            out.append((col, ch))
            col += wch
    return out


def _cells_slice(cells: list[tuple[int, str]], c1: int, c2: int) -> str:
    """Chars whose cells overlap [c1, c2); a wide char is included whole when
    a boundary lands on it."""
    out: list[str] = []
    for start, ch in cells:
        end = start + (2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1)
        if (c1 <= start < c2) or (start < c1 < end):
            out.append(ch)
    return "".join(out)


def _sel_row_bounds(state: UIState, r: int) -> tuple[int, int] | None:
    """(c1, c2) of the selection on screen row r; None when the row is not
    selected. Line-aware (terminal/tmux semantics): a multi-line drag selects
    the top line from its column to end-of-line, the bottom line from
    start-of-line to its column, and middle lines in full."""
    if state.sel_anchor is None or state.sel_extent is None:
        return None
    ar, ac = state.sel_anchor
    er, ec = state.sel_extent
    r1, r2 = min(ar, er), max(ar, er)
    if r < r1 or r > r2:
        return None
    if r1 == r2:
        return (min(ac, ec), max(ac, ec))
    top_col = ac if ar == r1 else ec
    bot_col = ec if er == r2 else ac
    if r == r1:
        return (top_col, 9999)
    if r == r2:
        return (0, bot_col)
    return (0, 9999)


def _extract_selection(state: UIState, height: int, width: int) -> str:
    """Text of the selected screen region using line-aware semantics (see
    _sel_row_bounds); lines are joined with newlines."""
    if state.sel_anchor is None or state.sel_extent is None:
        return ""
    r1 = min(state.sel_anchor[0], state.sel_extent[0])
    r2 = max(state.sel_anchor[0], state.sel_extent[0])
    rows = render_frame(state, height, width)[:height]
    lines: list[str] = []
    for r in range(r1, r2 + 1):
        cells = _row_cells(rows[r], width) if r < len(rows) else []
        c1, c2 = _sel_row_bounds(state, r)
        lines.append(_cells_slice(cells, c1, c2).rstrip())
    return "\n".join(lines)


def _copy_osc52(text: str) -> str:
    """OSC 52 clipboard-write sequence (the tmux approach); terminals that
    restrict writes simply ignore it, so this can silently no-op."""
    return f"\x1b]52;c;{base64.b64encode(text.encode('utf-8')).decode('ascii')}\x1b\\"


_CLIP_CMDS: list[list[str]] = []  # set once at startup by _pick_clipboard_cmds
_CLIP_READ_CMDS: list[list[str]] = []  # set once at startup by _pick_clipboard_read_cmds


def _pick_clipboard_cmds() -> list[list[str]]:
    """All installed local clipboard tools (each reads the text from stdin),
    in preference order. A session may only support some of them (e.g. an X11
    session with wl-clipboard installed), so copying tries each in turn."""
    candidates = (["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard"], ["pbcopy"])
    return [cmd for cmd in candidates if shutil.which(cmd[0])]


def _pick_clipboard_read_cmds() -> list[list[str]]:
    """All installed local tools that print the clipboard to stdout (for
    middle-click paste), in preference order."""
    candidates = (["wl-paste"], ["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard"], ["pbpaste"])
    return [cmd for cmd in candidates if shutil.which(cmd[0])]


def _read_clipboard() -> str | None:
    """Read the system clipboard via the first tool that succeeds."""
    for cmd in _CLIP_READ_CMDS:
        try:
            proc = subprocess.run(cmd, timeout=1, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout.decode("utf-8", "replace")
        except Exception:
            pass
    return None


def _copy_via_tool(text: str) -> bool:
    """Write text to the clipboard via the first tool that succeeds."""
    for cmd in _CLIP_CMDS:
        try:
            proc = subprocess.run(cmd, input=text.encode("utf-8"), timeout=2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if proc.returncode == 0:
                return True
        except Exception:
            pass
    return False


def _copy_selection(state: UIState, height: int, width: int) -> str | None:
    """Copy the active selection and note it in the status bar; returns the
    copied text (None when there was nothing to copy). Prefers a local
    clipboard tool (reliable where terminals restrict OSC 52 writes, e.g.
    GNOME Terminal), falling back to OSC 52."""
    text = _extract_selection(state, height, width)
    if not text:
        return None
    if not _copy_via_tool(text):
        sys.stdout.write(_copy_osc52(text))
        sys.stdout.flush()
    state.status = f"copied {len(text)} chars"
    return text


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
        curses.init_pair(6, curses.COLOR_WHITE, -1)  # soft gray (palette white) for line numbers
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
                STY_BOLD: curses.A_BOLD,
                STY_ITALIC: curses.A_ITALIC,
                # magenta like tool output: "code-ish"
                STY_CODE: curses.color_pair(2),
                # gray + faint where the terminal honors it (VTE ignores A_DIM alone)
                STY_NUM: curses.color_pair(6) | curses.A_DIM,
            }
        )
    except curses.error:
        _STYLE_ATTRS.update({STY_PLAIN: 0, STY_DIM: curses.A_DIM, STY_YOU: curses.A_BOLD, STY_TOOL: 0, STY_INFO: curses.A_BOLD, STY_STATUS: curses.A_BOLD, STY_ERROR: curses.A_BOLD, STY_WARN: 0, STY_BOLD: curses.A_BOLD, STY_ITALIC: curses.A_ITALIC, STY_CODE: 0, STY_NUM: curses.A_DIM})
    # raw() (not cbreak) so ISIG is off: Ctrl+C arrives as a 0x03 byte for
    # our decoder instead of raising SIGINT. Input is read from stdin directly;
    # getch is never used.
    curses.raw()
    curses.noecho()
    # Mouse reporting is fully enabled in run() (?1000h ?1002h ?1006h): the
    # wheel arrives as a distinct SGR sequence (button 64/65) rather than
    # being translated into arrow keys, and left-button press/drag/release
    # drive our own selection (highlighted; on release the region is copied
    # to the clipboard - local tool if installed, else OSC 52). Other mouse
    # sequences are swallowed by the decoder;
    # Up/Down recall user history (scroll when there is none).
    try:
        curses.curs_set(1)
    except curses.error:
        pass
    return stdscr


def _decode_sgr_mouse(body: bytes):
    """SGR mouse <button;col;row M/m: wheel up=64, down=65; left-button
    press (0), drag motion (32) and release (m 0) become ("mouse", ...)
    events for selection; other buttons are swallowed."""
    parts = body[2:-1].split(b";")
    if len(parts) != 3:
        return None
    try:
        button, col, row = (int(p) for p in parts)
    except ValueError:
        return None
    event = body[-1]
    if button == 64:
        return ("wheel", 1)
    if button == 65:
        return ("wheel", -1)
    if button == 0 and event == ord("M"):
        return ("mouse", ("press", col, row))
    if button == 32 and event == ord("M"):
        return ("mouse", ("drag", col, row))
    if button == 0 and event == ord("m"):
        return ("mouse", ("release", col, row))
    if button == 1 and event == ord("M"):
        return ("mouse", ("middle", col, row))
    return None


def _decode_esc_body(buf: bytes):
    """Map a post-ESC byte sequence to a key constant, a ("wheel", delta)
    event or a ("mouse", ...) event; None = swallow (unknown protocol).
    Empty buffer = bare ESC."""
    if not buf:
        return K_ESC
    if buf.startswith(b"[<"):  # SGR mouse
        return _decode_sgr_mouse(buf)
    mapping = {
        b"[A": K_UP,
        b"[B": K_DOWN,
        b"[C": K_RIGHT,
        b"[D": K_LEFT,
        b"[5~": K_PPAGE,
        b"[6~": K_NPAGE,
        b"[H": K_HOME,
        b"[F": K_END,
        b"[1~": K_HOME,
        b"[4~": K_END,
        b"[3~": K_DELETE,
        b"[1;5D": K_CTRL_LEFT,
        b"[1;5C": K_CTRL_RIGHT,
    }
    return mapping.get(buf)


def _decode_input(buf: bytes) -> tuple[list, bytes]:
    """Split raw stdin bytes into complete key events (ints, plus
    ("wheel", delta) and ("mouse", ...) tuples from SGR mouse) plus any
    trailing partial escape sequence (held back until more bytes arrive or
    the grace period in run() expires). Non-left-button mouse sequences are
    swallowed."""
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
        """Feed raw bytes; returns ("key", int), ("wheel", int) and
        ("paste", str) events."""
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
                self._add_keys(events, keys)
                self.buf = rem + rest  # hold any partial tail; keep the paste payload
                self.in_paste = True
                continue
            keys, rem = self._decode_keys(self.buf, now)
            self._add_keys(events, keys)
            self.buf = rem
            break
        return events

    def _add_keys(self, events: list[tuple[str, object]], keys: list) -> None:
        for k in keys:
            if isinstance(k, tuple):  # ("wheel", delta) passes through as-is
                events.append(k)
            else:
                events.append(("key", k))

    def _decode_keys(self, data: bytes, now: float) -> tuple[list, bytes]:
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
    chunk_w = max(1, width - 3)
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


def _draw_row_sel(stdscr, r: int, row: list[tuple[str, int]], width: int, c1: int, c2: int) -> None:
    """Draw a row with A_REVERSE on the cells overlapping [c1, c2); runs of
    the same selection state are drawn in one addnstr."""
    col = 0
    for text, style in row:
        if col >= width - 1 or not text:
            continue
        base = _STYLE_ATTRS.get(style, 0)
        i = 0
        while i < len(text):
            wch = 2 if unicodedata.east_asian_width(text[i]) in ("W", "F") else 1
            if col + wch > width - 1:
                return
            sel = col < c2 and col + wch > c1
            j = i + 1
            while j < len(text):
                ncol = col + _cell_len(text[i:j])
                w2 = 2 if unicodedata.east_asian_width(text[j]) in ("W", "F") else 1
                if (ncol < c2 and ncol + w2 > c1) != sel:
                    break
                j += 1
            chunk = text[i:j]
            try:
                stdscr.addnstr(r, col, chunk, (width - 1) - col, base | (curses.A_REVERSE if sel else 0))
            except curses.error:
                return
            col += _cell_len(chunk)
            i = j


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
        bounds = _sel_row_bounds(state, r)
        if bounds is not None:
            _draw_row_sel(stdscr, r, row, width, bounds[0], bounds[1])
            continue
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
    # ?1000h: normal mouse tracking (wheel + press/release); ?1002h: button
    # event tracking (drag motion - we own the mouse and implement selection
    # ourselves, tmux-style); ?1006h: SGR encoding.
    sys.stdout.write("\x1b[?2004h\x1b[?1000h\x1b[?1002h\x1b[?1006h")
    global _CLIP_CMDS, _CLIP_READ_CMDS
    _CLIP_CMDS = _pick_clipboard_cmds()  # copy via local tool(s) if installed
    _CLIP_READ_CMDS = _pick_clipboard_read_cmds()  # middle-click paste source
    logger.info("clipboard tools: %s", [c[0] for c in _CLIP_CMDS] or ["none (OSC 52 only)"])
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
                    _clear_selection(state)  # screen coords no longer map
            except Exception:
                pass
            height, width = stdscr.getmaxyx()
            input_lines = state.input_buf.split("\n") if state.input_buf else [""]
            view_h = max(1, height - 1 - _input_block_height(input_lines, width))
            for kind, value in events:
                if kind == "paste":
                    _insert_text(state, str(value))
                    continue
                if kind == "mouse":  # (event, col1, row1) from SGR left button
                    ev, col, row = value
                    _apply_mouse(state, ev, col, row, height, width)
                    continue
                key = ("wheel", int(value)) if kind == "wheel" else int(value)
                action, payload = apply_key(state, key, view_h=view_h, total=count_lines(state, width))
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
            sys.stdout.write("\x1b[?2004l\x1b[?1000l\x1b[?1002l")  # disable paste + mouse modes
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
