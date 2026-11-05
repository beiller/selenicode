import contextlib
import io
import os
import sys
import unicodedata
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clients.terminal as t2
from core.bus import CompactionDone, Error, History, ServerStatus, Token, TurnDone, UsageReport


def fill(state: t2.UIState, n: int) -> None:
    """Give the state n one-line blocks."""
    for i in range(n):
        state.blocks.append(t2.Block("assistant", "", f"line {i}", t2.STY_PLAIN))


class TestWrapText(unittest.TestCase):
    def test_wraps_long_line(self):
        lines = t2.wrap_text("aaaa bbbb cccc dddd", 11)
        self.assertTrue(all(len(ln) <= 11 for ln in lines))
        self.assertEqual(" ".join(lines).split(), ["aaaa", "bbbb", "cccc", "dddd"])

    def test_hard_wraps_long_word(self):
        lines = t2.wrap_text("x" * 50, 10)
        self.assertEqual(len(lines), 5)


class TestRenderFrame(unittest.TestCase):
    def test_pinned_shows_bottom(self):
        state = t2.UIState()
        fill(state, 30)
        rows = t2.render_frame(state, height=10, width=80)
        view_h = 8
        self.assertEqual(rows[view_h - 1][0][0], "line 29")
        self.assertEqual(rows[0][0][0], "")  # separator row
        self.assertEqual(rows[1][0][0], "line 26")

    def test_detached_view_stays_put_while_streaming(self):
        state = t2.UIState()
        fill(state, 30)
        state.view_top = 16
        rows_before = t2.render_frame(state, height=10, width=80)
        # more content arrives (streaming) while the user is reading
        fill(state, 40)
        rows_after = t2.render_frame(state, height=10, width=80)
        self.assertEqual(rows_before[0][0][0], rows_after[0][0][0])
        self.assertEqual(rows_after[0][0][0], "line 8")


class TestInputWrapping(unittest.TestCase):
    def _state(self, buf: str, cursor: int) -> t2.UIState:
        state = t2.UIState()
        state.input_buf = buf
        state.cursor = cursor
        return state

    def test_long_line_wraps_across_rows(self):
        state = self._state("a" * 50, 50)  # chunk_w = 20 - 3 = 17
        rows = t2.render_frame(state, height=10, width=20)
        self.assertEqual(len(rows), 10)  # 6 viewport + status + 3 input rows
        last3 = rows[-3:]
        self.assertEqual(last3[0][1][0], "a" * 17)
        self.assertEqual(last3[1][1][0], "a" * 17)
        self.assertEqual(last3[2][1][0], "a" * 16)
        self.assertEqual(last3[0][0][0], "> ")  # prompt only on first chunk
        self.assertEqual(last3[1][0][0], "  ")

    def test_cursor_backspace_stays_on_screen(self):
        state = self._state("a" * 50, 50)
        for _ in range(49):  # backspace down to one char
            t2.apply_key(state, 127)  # DEL
        row, col = t2._cursor_pos(state, height=10, width=20)
        # the block shrinks from 3 rows to 1 as text is deleted, so the
        # cursor sits on the bottom row - always on screen
        self.assertEqual((row, col), (9, 3))


class TestThinkingToggle(unittest.TestCase):
    def _state(self) -> t2.UIState:
        return t2.UIState()

    def _screen_text(self, state: t2.UIState) -> str:
        rows = t2.render_frame(state, height=10, width=80)
        return "".join(seg[0] for row in rows for seg in row)

    def test_collapsed_placeholder_by_default(self):
        state = self._state()
        t2.apply_token(state, "thinking", "let   me\nthink\n\n" + "b" * 100, True)
        t2.apply_token(state, "assistant", "hi", False)  # finalizes the thinking block
        text = self._screen_text(state)
        self.assertIn("116 chars:", text)  # len of raw incl. newlines
        self.assertIn("let me think", text)  # whitespace collapsed in preview
        self.assertNotIn("b" * 100, text)  # tail beyond the 100-char preview

    def test_toggle_round_trip(self):
        state = self._state()
        t2.apply_token(state, "thinking", "let me think", True)
        t2.apply_key(state, t2.K_CTRL_T)
        self.assertTrue(state.show_thinking)
        self.assertIn("let me think", self._screen_text(state))  # expanded on screen
        t2.apply_key(state, t2.K_CTRL_T)
        self.assertFalse(state.show_thinking)


class TestApplyToken(unittest.TestCase):
    def test_streaming_accumulates(self):
        state = t2.UIState()
        t2.apply_token(state, "assistant", "hel", True)
        t2.apply_token(state, "assistant", "lo", True)
        self.assertEqual(state.active.raw, "hello")

    def test_finalization_on_role_switch_and_end_marker(self):
        state = t2.UIState()
        t2.apply_token(state, "thinking", "hmm", True)
        t2.apply_token(state, "assistant", "hi", True)
        self.assertEqual(len(state.blocks), 1)
        self.assertEqual(state.blocks[0].raw, "hmm")
        self.assertEqual(state.active.raw, "hi")
        t2.apply_token(state, "assistant", "", False)
        self.assertIsNone(state.active)
        self.assertEqual(len(state.blocks), 2)  # thinking + assistant both finalized

    def test_oneshot_status_line(self):
        state = t2.UIState()
        t2.apply_token(state, "tool", "run_bash('ls')", False)
        self.assertEqual(len(state.blocks), 1)
        self.assertEqual(state.blocks[0].role, "tool")
        self.assertEqual(state.blocks[0].style, t2.STY_TOOL)

    def test_empty_streaming_block_not_kept(self):
        state = t2.UIState()
        t2.apply_token(state, "assistant", "", False)
        self.assertEqual(len(state.blocks), 0)


class TestApplyKey(unittest.TestCase):
    def _state(self, **kw) -> t2.UIState:
        s = t2.UIState()
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    def test_typing_and_cursor(self):
        state = self._state()
        for ch in "ab":
            t2.apply_key(state, ord(ch))
        t2.apply_key(state, t2.K_LEFT)
        t2.apply_key(state, ord("X"))
        self.assertEqual(state.input_buf, "aXb")
        self.assertEqual(state.cursor, 2)

    def test_enter(self):
        state = self._state(input_buf="  hello  ")
        action, payload = t2.apply_key(state, t2.K_ENTER)
        self.assertEqual((action, payload), ("submit", "hello"))
        self.assertEqual(state.input_buf, "")
        self.assertEqual(t2.apply_key(self._state(), t2.K_ENTER), ("", ""))  # empty is a no-op

    def test_ctrl_q_quits(self):
        self.assertEqual(t2.apply_key(self._state(), t2.K_CTRL_Q), ("quit", ""))

    def test_wheel_events(self):
        state = self._state()
        fill(state, 50)
        t2.apply_key(state, ("wheel", 3), total=50)
        self.assertEqual(state.view_top, 23)  # bottom (26) minus 3
        t2.apply_key(state, ("wheel", -10), total=50)
        self.assertIsNone(state.view_top)  # past the bottom: re-pinned


class TestBlocksFromHistory(unittest.TestCase):
    def test_full_mapping(self):
        msgs = [
            {"role": "user", "content": "do it"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "let me think",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run_bash", "arguments": '{"command": "ls"}'}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "output"},
            {"role": "assistant", "content": "done"},
        ]
        blocks = t2.blocks_from_history(msgs)
        self.assertEqual([b.role for b in blocks], ["user", "thinking", "tool", "assistant"])
        self.assertEqual(blocks[0].prefix, "You: ")
        self.assertEqual(blocks[1].style, t2.STY_DIM)
        self.assertIn("run_bash(", blocks[2].raw)
        self.assertEqual(blocks[3].prefix, "\U0001F916: ")

    def test_compaction_marker_and_summary(self):
        msgs = [
            {"role": "user", "content": "[compaction] The conversation before this point was summarized.", "compaction_marker": True},
            {"role": "assistant", "content": "## Goal\nstuff", "compaction_summary": True},
        ]
        blocks = t2.blocks_from_history(msgs)
        self.assertEqual(len(blocks), 2)
        self.assertIn("[compaction]", blocks[0].raw)
        self.assertEqual(blocks[1].prefix, "(summary) ")

class TestApplyEvent(unittest.TestCase):
    def test_dispatch(self):
        state = t2.UIState(context_size=100)
        t2.apply_event(state, Token("0", "assistant", "hi", True))
        self.assertEqual(state.active.raw, "hi")
        t2.apply_event(state, UsageReport("0", {"prompt_tokens": 10, "completion_tokens": 5}))
        self.assertEqual(state.tokens, 15)
        state.busy = True
        t2.apply_event(state, TurnDone("m1"))
        self.assertFalse(state.busy)
        self.assertEqual(state.status, "ready")
        self.assertIsNone(state.active)
        t2.apply_event(state, History([{"role": "user", "content": "old"}]))
        self.assertEqual(len(state.blocks), 2)
        self.assertEqual(state.blocks[-1].raw, "old")

    def test_error_renders_on_status_line_not_stream(self):
        state = t2.UIState(context_size=100)
        t2.apply_event(state, Error("u1", "Model server connection failed (attempt 1)"))
        self.assertEqual(state.status, "Model server connection failed (attempt 1)")
        self.assertEqual(state.status_style, t2.STY_ERROR)
        self.assertFalse(any(b.role == "error" for b in state.blocks))
        line = t2.render_frame(state, height=5, width=80)[-2][0][0]
        self.assertIn("Model server connection failed", line)

    def test_server_status_flips_status_line(self):
        state = t2.UIState(context_size=100)
        t2.apply_event(state, ServerStatus(False))
        self.assertEqual(state.status, "waiting for model server...")
        self.assertEqual(state.status_style, t2.STY_WARN)
        t2.apply_event(state, ServerStatus(True))
        self.assertEqual(state.status, "ready")
        self.assertEqual(state.status_style, t2.STY_STATUS)

    def test_turn_done_resets_error_status_style(self):
        state = t2.UIState(context_size=100)
        t2.apply_event(state, Error("u1", "boom"))
        t2.apply_event(state, TurnDone("m1"))
        self.assertEqual(state.status, "ready")
        self.assertEqual(state.status_style, t2.STY_STATUS)

    def test_compaction_done_updates_status_line_and_token_meter(self):
        state = t2.UIState(context_size=100)
        t2.apply_event(state, UsageReport("u1", {"prompt_tokens": 90, "completion_tokens": 5}))
        self.assertEqual(state.tokens, 95)
        # posted after TurnDone: the message survives the "ready" reset
        t2.apply_event(state, TurnDone("m1"))
        t2.apply_event(state, CompactionDone("u1", 30))
        self.assertEqual(state.tokens, 30)
        self.assertEqual(state.status, "compacted → 30 tokens")
        self.assertEqual(state.status_style, t2.STY_STATUS)
        line = t2.render_frame(state, height=5, width=80)[-2][0][0]
        self.assertIn("compacted", line)
        self.assertIn("tokens: 30/100 (30%)", line)
        # next turn's TurnDone clears it
        t2.apply_event(state, TurnDone("m2"))
        self.assertEqual(state.status, "ready")

    def test_status_bar_token_percentage(self):
        state = t2.UIState(context_size=1600)
        state.tokens = 400
        status_line = t2.render_frame(state, height=5, width=80)[-2][0][0]
        self.assertIn("tokens: 400/1600 (25%)", status_line)


class TestInputDecoder(unittest.TestCase):
    def test_plain_keys(self):
        d = t2._InputDecoder()
        self.assertEqual(d.feed(b"hi", 1.0), [("key", ord("h")), ("key", ord("i"))])

    def test_paste_ignores_enter_and_ctrl_c(self):
        d = t2._InputDecoder()
        self.assertEqual(d.feed(b"\x1b[200~a\r\x03b\x1b[201~", 1.0), [("paste", "a\nb")])

    def test_keys_around_paste(self):
        d = t2._InputDecoder()
        self.assertEqual(
            d.feed(b"x\x1b[200~p\rq\x1b[201~y", 1.0),
            [("key", ord("x")), ("paste", "p\nq"), ("key", ord("y"))],
        )

    def test_paste_split_across_feeds(self):
        d = t2._InputDecoder()
        self.assertEqual(d.feed(b"\x1b[200~li", 1.0), [])
        self.assertEqual(d.feed(b"ne1\nlin", 1.5), [])
        self.assertEqual(d.feed(b"e2\x1b[201~", 2.0), [("paste", "line1\nline2")])

    def test_bare_esc_grace_still_works(self):
        d = t2._InputDecoder()
        self.assertEqual(d.feed(b"\x1b", 1.0), [])
        self.assertEqual(d.feed(b"", 1.1), [("key", t2.K_ESC)])

    def test_paste_containing_esc_byte(self):
        d = t2._InputDecoder()
        self.assertEqual(d.feed(b"\x1b[200~a\x1bb\x1b[201~", 1.0), [("paste", "ab")])


class TestMultilineInput(unittest.TestCase):
    def test_multiline_input_renders_block(self):
        state = t2.UIState()
        state.input_buf = "line one\nline two"
        rows = t2.render_frame(state, height=10, width=80)
        self.assertEqual(len(rows), 10)
        self.assertIn("status:", "".join(s[0] for s in rows[7]))
        self.assertEqual(rows[8][0][0], "> ")
        self.assertEqual(rows[8][1][0], "line one")
        self.assertEqual(rows[9][0][0], "  ")
        self.assertEqual(rows[9][1][0], "line two")

    def test_multiline_submit_keeps_newlines(self):
        state = t2.UIState()
        t2._insert_text(state, "line1\nline2")
        action, payload = t2.apply_key(state, t2.K_ENTER)
        self.assertEqual(action, "submit")
        self.assertEqual(payload, "line1\nline2")


class TestSubmitMessage(unittest.TestCase):
    def test_prompt_recorded_and_turn_started(self):
        state = t2.UIState()
        state.view_top = 5  # detached (reading old content)
        t2.apply_token(state, "assistant", "old answer", True)
        t2.submit_message(state, "my prompt")
        self.assertTrue(state.busy)
        self.assertIsNone(state.view_top)  # re-pinned to follow the new turn
        block = state.blocks[-1]
        self.assertEqual((block.role, block.prefix, block.raw, block.style), ("user", "You: ", "my prompt", t2.STY_YOU))
        # the previous active block was finalized into scrollback first
        self.assertEqual(state.blocks[0].raw, "old answer")
        self.assertIsNone(state.active)


class TestDecodeInput(unittest.TestCase):
    def test_enter_backspace_ctrl(self):
        self.assertEqual(t2._decode_input(b"\r"), ([t2.K_ENTER], b""))
        self.assertEqual(t2._decode_input(b"\n"), ([t2.K_ENTER], b""))
        self.assertEqual(t2._decode_input(b"\x7f"), ([t2.K_BACKSPACE], b""))
        self.assertEqual(t2._decode_input(b"\x08"), ([t2.K_BACKSPACE], b""))
        self.assertEqual(t2._decode_input(b"\x03"), ([t2.K_CTRL_C], b""))
        self.assertEqual(t2._decode_input(b"\x11"), ([t2.K_CTRL_Q], b""))
        self.assertEqual(t2._decode_input(b"\x14"), ([t2.K_CTRL_T], b""))

    def test_ctrl_t_end_to_end(self):
        # regression: byte 20 was swallowed by the decoder's catch-all, so
        # ctrl+t never reached apply_key at all
        state = t2.UIState()
        keys, rem = t2._decode_input(b"\x14")
        self.assertEqual(rem, b"")
        for k in keys:
            t2.apply_key(state, k)
        self.assertTrue(state.show_thinking)

    def test_function_keys(self):
        self.assertEqual(t2._decode_input(b"\x1b[A"), ([t2.K_UP], b""))
        self.assertEqual(t2._decode_input(b"\x1b[B"), ([t2.K_DOWN], b""))
        self.assertEqual(t2._decode_input(b"\x1b[C"), ([t2.K_RIGHT], b""))
        self.assertEqual(t2._decode_input(b"\x1b[D"), ([t2.K_LEFT], b""))
        self.assertEqual(t2._decode_input(b"\x1b[5~"), ([t2.K_PPAGE], b""))
        self.assertEqual(t2._decode_input(b"\x1b[6~"), ([t2.K_NPAGE], b""))
        self.assertEqual(t2._decode_input(b"\x1b[H"), ([t2.K_HOME], b""))
        self.assertEqual(t2._decode_input(b"\x1b[F"), ([t2.K_END], b""))

    def test_partial_sequences_held_back(self):
        self.assertEqual(t2._decode_input(b"\x1b"), ([], b"\x1b"))
        self.assertEqual(t2._decode_input(b"\x1b["), ([], b"\x1b["))
        keys, rem = t2._decode_input(b"\x1b[A\x1b[5")  # complete + partial in one feed
        self.assertEqual(keys, [t2.K_UP])
        self.assertEqual(rem, b"\x1b[5")

    def test_unknown_sequences_swallowed(self):
        self.assertEqual(t2._decode_input(b"\x1b[<2;5;3M"), ([], b""))  # right click swallowed
        self.assertEqual(t2._decode_input(b"\x1bM\x01\x02\x03"), ([], b""))  # legacy mouse
        self.assertEqual(t2._decode_input(b"\x1b[99;99u"), ([], b""))

    def test_mixed_stream(self):
        keys, rem = t2._decode_input(b"a\x1b[Bb")
        self.assertEqual(keys, [ord("a"), t2.K_DOWN, ord("b")])
        self.assertEqual(rem, b"")


class TestEditorKeys(unittest.TestCase):
    def test_delete_key(self):
        self.assertEqual(t2._decode_input(b"\x1b[3~"), ([t2.K_DELETE], b""))
        state = t2.UIState()
        state.input_buf = "abcde"
        state.cursor = 2
        t2.apply_key(state, t2.K_DELETE)
        self.assertEqual(state.input_buf, "abde")
        self.assertEqual(state.cursor, 2)
        # at the end: no-op
        state.cursor = len(state.input_buf)
        t2.apply_key(state, t2.K_DELETE)
        self.assertEqual(state.input_buf, "abde")

    def test_ctrl_arrow_word_jump(self):
        self.assertEqual(t2._decode_input(b"\x1b[1;5D"), ([t2.K_CTRL_LEFT], b""))
        self.assertEqual(t2._decode_input(b"\x1b[1;5C"), ([t2.K_CTRL_RIGHT], b""))
        state = t2.UIState()
        state.input_buf = "hello world foo"
        state.cursor = 5  # on the space before 'world'
        t2.apply_key(state, t2.K_CTRL_RIGHT)
        self.assertEqual(state.cursor, 11)  # end of 'world'
        t2.apply_key(state, t2.K_CTRL_LEFT)
        self.assertEqual(state.cursor, 6)  # start of 'world'
        t2.apply_key(state, t2.K_CTRL_LEFT)
        self.assertEqual(state.cursor, 0)  # start of buffer
        state.cursor = len(state.input_buf)
        t2.apply_key(state, t2.K_CTRL_RIGHT)
        self.assertEqual(state.cursor, 15)  # already at the end

    def test_home_end_cursor(self):
        self.assertEqual(t2._decode_input(b"\x1b[1~"), ([t2.K_HOME], b""))
        self.assertEqual(t2._decode_input(b"\x1b[4~"), ([t2.K_END], b""))
        state = t2.UIState()
        state.input_buf = "hello\nworld"
        state.cursor = 7  # inside the second line
        t2.apply_key(state, t2.K_HOME)
        self.assertEqual(state.cursor, 6)
        t2.apply_key(state, t2.K_END)
        self.assertEqual(state.cursor, 11)
        # single line: full range
        state.input_buf = "abc"
        state.cursor = 1
        t2.apply_key(state, t2.K_HOME)
        self.assertEqual(state.cursor, 0)
        t2.apply_key(state, t2.K_END)
        self.assertEqual(state.cursor, 3)

    def test_wheel_sgr_events(self):
        self.assertEqual(t2._decode_input(b"\x1b[<64;5;3M"), ([("wheel", 1)], b""))
        self.assertEqual(t2._decode_input(b"\x1b[<65;5;3m"), ([("wheel", -1)], b""))
        d = t2._InputDecoder()
        self.assertEqual(d.feed(b"\x1b[<64;5;3M", 1.0), [("wheel", 1)])
        self.assertEqual(d.feed(b"\x1b[<65;5;3M", 1.1), [("wheel", -1)])
        # one notch = one line of scroll
        state = t2.UIState()
        fill(state, 50)
        t2.apply_key(state, ("wheel", 1), total=50)
        self.assertEqual(state.view_top, 25)  # bottom (26) minus 1

    def test_row_never_exceeds_width(self):
        cases = [
            ("user", "You: ", "x" * 300),
            ("assistant", "\U0001F916: ", "y" * 300),
            ("info", "", "z" * 300),
            ("error", "ERROR: ", "e" * 300),
        ]
        for role, prefix, text in cases:
            state = t2.UIState()
            state.blocks.append(t2.Block(role, prefix, text, t2.STY_PLAIN))
            for width in (40, 80, 120):
                rows = t2.render_frame(state, height=24, width=width)
                # scrollback rows only; the pinned status bar and hint line are
                # clipped at the screen edge by _render
                for row in rows[:-2]:
                    cells = sum(
                        sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in t)
                        for t, _ in row
                    )
                    self.assertLessEqual(cells, width - 1)

    def test_collapsed_placeholder_wraps(self):
        state = t2.UIState()
        t2.apply_token(state, "thinking", "b" * 500, True)
        t2.apply_token(state, "assistant", "hi", False)
        rows = t2.render_frame(state, height=24, width=40)
        for row in rows[:-2]:  # pinned status bar / hint are clipped by _render
            cells = sum(
                sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in t)
                for t, _ in row
            )
            self.assertLessEqual(cells, 39)


class TestHistoryRecall(unittest.TestCase):
    def test_up_down_history_recall(self):
        state = t2.UIState()
        t2.submit_message(state, "first")
        t2.submit_message(state, "second")
        state.input_buf = "draft"
        t2.apply_key(state, t2.K_UP)
        self.assertEqual(state.input_buf, "second")
        self.assertEqual(state.cursor, 6)
        t2.apply_key(state, t2.K_UP)
        self.assertEqual(state.input_buf, "first")
        t2.apply_key(state, t2.K_DOWN)
        self.assertEqual(state.input_buf, "second")
        t2.apply_key(state, t2.K_DOWN)
        self.assertEqual(state.input_buf, "draft")  # stashed line restored
        self.assertIsNone(state.hist_index)

    def test_up_without_history_still_scrolls(self):
        state = t2.UIState()
        fill(state, 50)
        t2.apply_key(state, t2.K_UP, total=99)
        self.assertEqual(state.view_top, 72)  # bottom (75) minus 3

    def test_edit_ends_history_nav(self):
        state = t2.UIState()
        t2.submit_message(state, "one")
        t2.submit_message(state, "two")
        t2.apply_key(state, t2.K_UP)  # 'two'
        t2.apply_key(state, t2.K_UP)  # 'one'
        t2.apply_key(state, ord("x"))  # editing ends navigation
        self.assertIsNone(state.hist_index)
        t2.apply_key(state, t2.K_UP)
        self.assertEqual(state.input_buf, "two")  # recalls the newest again

    def test_submit_ends_history_nav(self):
        state = t2.UIState()
        t2.submit_message(state, "one")
        t2.apply_key(state, t2.K_UP)  # recall 'one'
        action, _ = t2.apply_key(state, t2.K_ENTER)  # submit it
        self.assertEqual(action, "submit")
        self.assertIsNone(state.hist_index)
        t2.apply_key(state, t2.K_UP)
        self.assertEqual(state.input_buf, "one")  # the just-submitted message


class TestMouseSelection(unittest.TestCase):
    def test_sgr_mouse_events_decode(self):
        dec = t2._InputDecoder()
        ev = dec.feed(b"\x1b[<0;5;3M\x1b[<32;8;4M\x1b[<0;9;5m", 0.0)
        self.assertIn(("mouse", ("press", 5, 3)), ev)
        self.assertIn(("mouse", ("drag", 8, 4)), ev)
        self.assertIn(("mouse", ("release", 9, 5)), ev)

    def test_wheel_still_decodes(self):
        dec = t2._InputDecoder()
        ev = dec.feed(b"\x1b[<64;1;1M\x1b[<65;1;1M", 0.0)
        self.assertIn(("wheel", 1), ev)
        self.assertIn(("wheel", -1), ev)

    def test_other_mouse_swallowed(self):
        dec = t2._InputDecoder()
        # right press, motion without button, middle release
        self.assertEqual(dec.feed(b"\x1b[<2;5;3M\x1b[<3;5;3M\x1b[<1;5;3m", 0.0), [])

    def test_middle_click_decodes(self):
        dec = t2._InputDecoder()
        ev = dec.feed(b"\x1b[<1;5;3M", 0.0)
        self.assertIn(("mouse", ("middle", 5, 3)), ev)

    def test_middle_click_pastes_clipboard(self):
        state = t2.UIState()
        saved = t2._CLIP_READ_CMDS
        t2._CLIP_READ_CMDS = [["sh", "-c", "printf 'pasted text'"]]
        try:
            t2._apply_mouse(state, "middle", 5, 3, 10, 40)
        finally:
            t2._CLIP_READ_CMDS = saved
        self.assertEqual(state.input_buf, "pasted text")
        self.assertEqual(state.cursor, 11)

    def test_middle_click_without_tool_is_noop(self):
        state = t2.UIState()
        saved = t2._CLIP_READ_CMDS
        t2._CLIP_READ_CMDS = []
        try:
            t2._apply_mouse(state, "middle", 5, 3, 10, 40)
        finally:
            t2._CLIP_READ_CMDS = saved
        self.assertEqual(state.input_buf, "")

    def test_middle_click_ends_history_nav(self):
        state = t2.UIState()
        t2.submit_message(state, "one")
        t2.apply_key(state, t2.K_UP)  # recall 'one'
        self.assertIsNotNone(state.hist_index)
        saved = t2._CLIP_READ_CMDS
        t2._CLIP_READ_CMDS = [["sh", "-c", "printf x"]]
        try:
            t2._apply_mouse(state, "middle", 5, 3, 10, 40)
        finally:
            t2._CLIP_READ_CMDS = saved
        self.assertIsNone(state.hist_index)

    def test_osc52_format(self):
        import base64
        seq = t2._copy_osc52("hi")
        self.assertTrue(seq.startswith("\x1b]52;c;"))
        self.assertTrue(seq.endswith("\x1b\\"))
        self.assertEqual(base64.b64decode(seq[7:-2]), b"hi")

    def test_selection_range_normalizes(self):
        state = t2.UIState()
        state.sel_anchor = (3, 9)
        state.sel_extent = (1, 4)
        self.assertEqual(t2._selection_range(state), (1, 3, 4, 9))
        state.sel_extent = None
        self.assertIsNone(t2._selection_range(state))

    def test_click_without_drag_no_copy(self):
        state = t2.UIState()
        t2._apply_mouse(state, "press", 5, 3, 10, 40)
        t2._apply_mouse(state, "release", 5, 3, 10, 40)
        self.assertIsNone(state.sel_anchor)
        self.assertEqual(state.status, "ready")

    def _select_hello(self, state: t2.UIState) -> None:
        # screen row 0 is "You: hello world"; "hello world" spans cols 5..15
        t2._apply_mouse(state, "press", 6, 1, 10, 40)
        t2._apply_mouse(state, "drag", 17, 1, 10, 40)

    def test_drag_release_copies_osc52(self):
        state = t2.UIState()
        t2.submit_message(state, "hello world")
        self._select_hello(state)
        saved = t2._CLIP_CMDS
        t2._CLIP_CMDS = []  # force the OSC 52 path
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                text = t2._copy_selection(state, 10, 40)
        finally:
            t2._CLIP_CMDS = saved
        self.assertEqual(text, "hello world")
        self.assertEqual(out.getvalue(), t2._copy_osc52("hello world"))
        self.assertEqual(state.status, "copied 11 chars")

    def test_release_copies_and_clears(self):
        state = t2.UIState()
        t2.submit_message(state, "hello world")
        self._select_hello(state)
        saved = t2._CLIP_CMDS
        t2._CLIP_CMDS = []
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                t2._apply_mouse(state, "release", 17, 1, 10, 40)
        finally:
            t2._CLIP_CMDS = saved
        self.assertIsNone(state.sel_anchor)
        self.assertIsNone(state.sel_extent)
        self.assertEqual(state.status, "copied 11 chars")

    def test_clipboard_tool_preferred_over_osc52(self):
        state = t2.UIState()
        t2.submit_message(state, "hello world")
        self._select_hello(state)
        saved = t2._CLIP_CMDS
        t2._CLIP_CMDS = [["true"]]  # succeeds; no OSC 52 should be emitted
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                text = t2._copy_selection(state, 10, 40)
        finally:
            t2._CLIP_CMDS = saved
        self.assertEqual(text, "hello world")
        self.assertEqual(out.getvalue(), "")

    def test_osc52_fallback_when_tool_fails(self):
        state = t2.UIState()
        t2.submit_message(state, "hello world")
        self._select_hello(state)
        saved = t2._CLIP_CMDS
        t2._CLIP_CMDS = [["false"]]  # fails; OSC 52 must be emitted
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                t2._copy_selection(state, 10, 40)
        finally:
            t2._CLIP_CMDS = saved
        self.assertEqual(out.getvalue(), t2._copy_osc52("hello world"))

    def test_tries_next_tool_when_first_fails(self):
        # e.g. wl-copy installed but the session is X11: xclip must get a turn
        state = t2.UIState()
        t2.submit_message(state, "hello world")
        self._select_hello(state)
        saved = t2._CLIP_CMDS
        t2._CLIP_CMDS = [["false"], ["true"]]
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                text = t2._copy_selection(state, 10, 40)
        finally:
            t2._CLIP_CMDS = saved
        self.assertEqual(text, "hello world")
        self.assertEqual(out.getvalue(), "")

    def test_pick_clipboard_cmds(self):
        import shutil as sh
        cmds = t2._pick_clipboard_cmds()
        self.assertIsInstance(cmds, list)
        if sh.which("xclip"):
            self.assertIn(["xclip", "-selection", "clipboard"], cmds)
        if sh.which("wl-copy"):
            self.assertIn(["wl-copy"], cmds)

    def test_multiline_selection(self):
        state = t2.UIState()
        t2.submit_message(state, "line one")
        t2.apply_token(state, "assistant", "second line", chunk=False)
        # row 0 = "You: line one", row 1 = blank gap, row 2 = "second line";
        # top line from col 0 to end-of-line, bottom line to col 5
        t2._apply_mouse(state, "press", 1, 1, 10, 40)
        t2._apply_mouse(state, "drag", 6, 3, 10, 40)
        self.assertEqual(t2._extract_selection(state, 10, 40), "You: line one\n\nsecon")

    def test_drag_down_selects_remainder_of_first_line(self):
        state = t2.UIState()
        t2.submit_message(state, "first line here")
        t2.apply_token(state, "assistant", "second line here", chunk=False)
        # "You: first line here": start mid-line (col 11), drag down to col 4
        # of the second line: top line from col 11 to end, bottom line from
        # start to col 4, joined with newlines (row 1 is the blank gap)
        t2._apply_mouse(state, "press", 12, 1, 10, 40)
        t2._apply_mouse(state, "drag", 5, 3, 10, 40)
        self.assertEqual(t2._extract_selection(state, 10, 40), "line here\n\nseco")

    def test_row_bounds_line_aware(self):
        state = t2.UIState()
        state.sel_anchor = (0, 6)
        state.sel_extent = (2, 4)
        self.assertEqual(t2._sel_row_bounds(state, 0), (6, 9999))
        self.assertEqual(t2._sel_row_bounds(state, 1), (0, 9999))
        self.assertEqual(t2._sel_row_bounds(state, 2), (0, 4))
        self.assertIsNone(t2._sel_row_bounds(state, 3))

    def test_wide_char_boundary(self):
        state = t2.UIState()
        t2.submit_message(state, "x\U0001F600y")
        # "You: x😀y": x=5, 😀=6-7, y=8; a boundary on the wide char's right
        # half still selects it whole
        state.sel_anchor = (0, 7)
        state.sel_extent = (0, 9)
        self.assertEqual(t2._extract_selection(state, 10, 40), "\U0001F600y")

    def test_scroll_clears_selection(self):
        state = t2.UIState()
        fill(state, 50)
        t2._apply_mouse(state, "press", 5, 3, 24, 80)
        t2._apply_mouse(state, "drag", 5, 10, 24, 80)
        self.assertIsNotNone(state.sel_extent)
        t2.apply_key(state, t2.K_UP, total=99)  # no user history -> scroll
        self.assertIsNone(state.sel_anchor)


class TestCellLen(unittest.TestCase):
    def test_cell_widths(self):
        self.assertEqual(t2._cell_len("abc"), 3)
        self.assertEqual(t2._cell_len("\U0001F916"), 2)  # robot emoji
        self.assertEqual(t2._cell_len("\U0001F311"), 2)  # new moon spinner
        self.assertEqual(t2._cell_len("\U0001F916: "), 4)  # emoji(2) + ':' + ' '


if __name__ == "__main__":
    unittest.main()
