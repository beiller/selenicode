import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clients.terminal as t2
from bus import History, Token, TurnDone, UsageReport


def fill(state: t2.UIState, n: int) -> None:
    """Give the state n one-line blocks."""
    for i in range(n):
        state.blocks.append(t2.Block("assistant", "", f"line {i}", t2.STY_PLAIN))


class TestWrapText(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(t2.wrap_text("", 40), [""])

    def test_short_line(self):
        self.assertEqual(t2.wrap_text("hello", 40), ["hello"])

    def test_wraps_long_line(self):
        lines = t2.wrap_text("aaaa bbbb cccc dddd", 11)
        self.assertTrue(all(len(ln) <= 11 for ln in lines))
        self.assertEqual(" ".join(lines).split(), ["aaaa", "bbbb", "cccc", "dddd"])

    def test_keeps_empty_paragraphs(self):
        self.assertEqual(t2.wrap_text("a\n\nb", 40), ["a", "", "b"])

    def test_hard_wraps_long_word(self):
        lines = t2.wrap_text("x" * 50, 10)
        self.assertEqual(len(lines), 5)


class TestRenderFrame(unittest.TestCase):
    def test_frame_shape_and_pinned_bars(self):
        state = t2.UIState(context_size=1000)
        fill(state, 3)
        rows = t2.render_frame(state, height=10, width=80)
        self.assertEqual(len(rows), 10)
        status_text = "".join(seg[0] for seg in rows[8])
        self.assertIn("ready", status_text)
        self.assertEqual(rows[8][0][1], t2.STY_STATUS)
        self.assertTrue(rows[9][0][0].startswith("> "))

    def test_few_lines_top_aligned(self):
        state = t2.UIState()
        fill(state, 3)
        rows = t2.render_frame(state, height=10, width=80)
        self.assertEqual(rows[0][0][0], "line 0")
        self.assertEqual(rows[1][0][0], "line 1")

    def test_pinned_shows_bottom(self):
        state = t2.UIState()
        fill(state, 30)
        rows = t2.render_frame(state, height=10, width=80)
        view_h = 8
        self.assertEqual(rows[view_h - 1][0][0], "line 29")
        self.assertEqual(rows[0][0][0], "line 22")

    def test_detached_view_offsets_window(self):
        state = t2.UIState()
        fill(state, 30)
        state.view_top = 17  # bottom (22) minus 5
        rows = t2.render_frame(state, height=10, width=80)
        self.assertEqual(rows[7][0][0], "line 24")

    def test_detached_view_clamped_at_top(self):
        state = t2.UIState()
        fill(state, 30)
        state.view_top = 0
        rows = t2.render_frame(state, height=10, width=80)
        self.assertEqual(rows[0][0][0], "line 0")

    def test_detached_view_stays_put_while_streaming(self):
        state = t2.UIState()
        fill(state, 30)
        state.view_top = 17
        rows_before = t2.render_frame(state, height=10, width=80)
        # more content arrives (streaming) while the user is reading
        fill(state, 40)
        rows_after = t2.render_frame(state, height=10, width=80)
        self.assertEqual(rows_before[0][0][0], rows_after[0][0][0])
        self.assertEqual(rows_after[0][0][0], "line 17")

    def test_count_lines(self):
        state = t2.UIState()
        fill(state, 3)
        self.assertEqual(t2.count_lines(state, 80), 3)
        t2.apply_token(state, "assistant", "abc", True)
        self.assertEqual(t2.count_lines(state, 80), 4)

    def test_prompt_hint_only_when_empty(self):
        state = t2.UIState()
        rows = t2.render_frame(state, height=5, width=80)
        self.assertTrue(any("ctrl+q" in seg[0] for seg in rows[-1]))
        state.input_buf = "hi"
        rows = t2.render_frame(state, height=5, width=80)
        self.assertFalse(any("ctrl+q" in seg[0] for seg in rows[-1]))
        self.assertEqual(rows[-1][1][0], "hi")

    def test_token_usage_in_status(self):
        state = t2.UIState(context_size=16384)
        state.tokens = 500
        rows = t2.render_frame(state, height=5, width=80)
        self.assertIn("tokens: 500/16384", "".join(seg[0] for seg in rows[-2]))


class TestApplyToken(unittest.TestCase):
    def test_streaming_accumulates(self):
        state = t2.UIState()
        t2.apply_token(state, "assistant", "hel", True)
        t2.apply_token(state, "assistant", "lo", True)
        self.assertEqual(state.active.raw, "hello")

    def test_role_switch_finalizes(self):
        state = t2.UIState()
        t2.apply_token(state, "thinking", "hmm", True)
        t2.apply_token(state, "assistant", "hi", True)
        self.assertEqual(len(state.blocks), 1)
        self.assertEqual(state.blocks[0].raw, "hmm")
        self.assertEqual(state.active.raw, "hi")

    def test_end_marker_finalizes(self):
        state = t2.UIState()
        t2.apply_token(state, "assistant", "hi", True)
        t2.apply_token(state, "assistant", "", False)
        self.assertIsNone(state.active)
        self.assertEqual(len(state.blocks), 1)

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

    def test_backspace(self):
        state = self._state(input_buf="ab", cursor=2)
        t2.apply_key(state, t2.K_BACKSPACE)
        self.assertEqual(state.input_buf, "a")
        self.assertEqual(state.cursor, 1)

    def test_enter_submits_and_clears(self):
        state = self._state(input_buf="  hello  ")
        action, payload = t2.apply_key(state, t2.K_ENTER)
        self.assertEqual((action, payload), ("submit", "hello"))
        self.assertEqual(state.input_buf, "")

    def test_enter_empty_is_noop(self):
        state = self._state()
        self.assertEqual(t2.apply_key(state, t2.K_ENTER), ("", ""))

    def test_ctrl_c_interrupts_when_busy(self):
        state = self._state(busy=True)
        self.assertEqual(t2.apply_key(state, t2.K_CTRL_C), ("interrupt", ""))

    def test_ctrl_c_clears_input_when_idle(self):
        state = self._state(input_buf="abc")
        self.assertEqual(t2.apply_key(state, t2.K_CTRL_C), ("", ""))
        self.assertEqual(state.input_buf, "")

    def test_ctrl_q_quits(self):
        self.assertEqual(t2.apply_key(self._state(), t2.K_CTRL_Q), ("quit", ""))

    def test_esc_clears_input(self):
        state = self._state(input_buf="abc")
        t2.apply_key(state, t2.K_ESC)
        self.assertEqual(state.input_buf, "")

    def test_page_up_down_and_end(self):
        fill(self_state := self._state(), 50)
        action, _ = t2.apply_key(self_state, t2.K_PPAGE, view_h=10, total=50)
        self.assertEqual(action, "")
        self.assertEqual(self_state.view_top, 31)  # bottom (40) minus a page (9)
        t2.apply_key(self_state, t2.K_NPAGE, view_h=10, total=50)
        self.assertIsNone(self_state.view_top)  # back at the bottom: re-pinned
        t2.apply_key(self_state, t2.K_HOME)
        self.assertEqual(self_state.view_top, 0)
        t2.apply_key(self_state, t2.K_END)
        self.assertIsNone(self_state.view_top)

    def test_scroll_up_anchors_at_bottom(self):
        state = self._state()
        fill(state, 30)
        t2.apply_key(state, t2.K_UP, view_h=8, total=30)
        self.assertEqual(state.view_top, 19)  # bottom (22) minus 3

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

    def test_compaction_marker_and_summary(self):
        msgs = [
            {"role": "user", "content": "[compaction] The conversation before this point was summarized.", "compaction_marker": True},
            {"role": "assistant", "content": "## Goal\nstuff", "compaction_summary": True},
        ]
        blocks = t2.blocks_from_history(msgs)
        self.assertEqual(len(blocks), 2)
        self.assertIn("[compaction]", blocks[0].raw)
        self.assertEqual(blocks[1].prefix, "(summary) ")

    def test_empty_content_skipped(self):
        blocks = t2.blocks_from_history([{"role": "user", "content": ""}, {"role": "assistant", "content": None}])
        self.assertEqual(blocks, [])


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

    def test_usage_updates_token_count(self):
        state = t2.UIState()
        t2.apply_event(state, UsageReport("0", {"prompt_tokens": 7, "completion_tokens": 3}))
        self.assertEqual(state.tokens, 10)

    def test_status_bar_spinner_and_ready(self):
        state = t2.UIState(context_size=100)
        status_line = t2.render_frame(state, height=5, width=80)[-2][0][0]
        self.assertEqual(status_line, "status: ready tokens: 0/100 (0%)")
        state.busy = True
        state.frame = 3
        status_line = t2.render_frame(state, height=5, width=80)[-2][0][0]
        self.assertEqual(status_line, f"status: {t2.SPINNER[3]} thinking tokens: 0/100 (0%)")

    def test_status_bar_token_percentage(self):
        state = t2.UIState(context_size=1600)
        state.tokens = 400
        status_line = t2.render_frame(state, height=5, width=80)[-2][0][0]
        self.assertIn("tokens: 400/1600 (25%)", status_line)


class TestInputDecoder(unittest.TestCase):
    def test_plain_keys(self):
        d = t2._InputDecoder()
        self.assertEqual(d.feed(b"hi", 1.0), [("key", ord("h")), ("key", ord("i"))])

    def test_paste_is_literal(self):
        d = t2._InputDecoder()
        self.assertEqual(d.feed(b"\x1b[200~line1\r\nline2\x1b[201~", 1.0), [("paste", "line1\nline2")])

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

    def test_multiline_cursor_pos(self):
        state = t2.UIState()
        state.input_buf = "ab\ncd"
        state.cursor = 0
        self.assertEqual(t2._cursor_pos(state, 10), (8, 2))
        state.cursor = 3  # start of line 2
        self.assertEqual(t2._cursor_pos(state, 10), (9, 2))
        state.cursor = 5  # end of "cd"
        self.assertEqual(t2._cursor_pos(state, 10), (9, 4))

    def test_insert_text(self):
        state = t2.UIState()
        state.input_buf = "abc"
        state.cursor = 1
        t2._insert_text(state, "XY")
        self.assertEqual(state.input_buf, "aXYbc")
        self.assertEqual(state.cursor, 3)

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

    def test_matches_history_rendering(self):
        live = t2.UIState()
        t2.submit_message(live, "hello")
        from_hist = t2.blocks_from_history([{"role": "user", "content": "hello"}])
        a, b = live.blocks[-1], from_hist[0]
        self.assertEqual((a.role, a.prefix, a.raw, a.style), (b.role, b.prefix, b.raw, b.style))


class TestDecodeInput(unittest.TestCase):
    def test_printable(self):
        self.assertEqual(t2._decode_input(b"hello"), ([ord(c) for c in "hello"], b""))

    def test_enter_backspace_ctrl(self):
        self.assertEqual(t2._decode_input(b"\r"), ([t2.K_ENTER], b""))
        self.assertEqual(t2._decode_input(b"\n"), ([t2.K_ENTER], b""))
        self.assertEqual(t2._decode_input(b"\x7f"), ([t2.K_BACKSPACE], b""))
        self.assertEqual(t2._decode_input(b"\x08"), ([t2.K_BACKSPACE], b""))
        self.assertEqual(t2._decode_input(b"\x03"), ([t2.K_CTRL_C], b""))
        self.assertEqual(t2._decode_input(b"\x11"), ([t2.K_CTRL_Q], b""))

    def test_function_keys(self):
        self.assertEqual(t2._decode_input(b"\x1b[A"), ([t2.K_UP], b""))
        self.assertEqual(t2._decode_input(b"\x1b[B"), ([t2.K_DOWN], b""))
        self.assertEqual(t2._decode_input(b"\x1b[C"), ([t2.K_RIGHT], b""))
        self.assertEqual(t2._decode_input(b"\x1b[D"), ([t2.K_LEFT], b""))
        self.assertEqual(t2._decode_input(b"\x1b[5~"), ([t2.K_PPAGE], b""))
        self.assertEqual(t2._decode_input(b"\x1b[6~"), ([t2.K_NPAGE], b""))
        self.assertEqual(t2._decode_input(b"\x1b[H"), ([t2.K_HOME], b""))
        self.assertEqual(t2._decode_input(b"\x1b[F"), ([t2.K_END], b""))

    def test_split_escape_held_back(self):
        self.assertEqual(t2._decode_input(b"\x1b"), ([], b"\x1b"))
        self.assertEqual(t2._decode_input(b"\x1b["), ([], b"\x1b["))
        keys, rem = t2._decode_input(b"\x1b")
        self.assertEqual(rem, b"\x1b")

    def test_mouse_swallowed(self):
        self.assertEqual(t2._decode_input(b"\x1b[<64;5;3M"), ([], b""))
        self.assertEqual(t2._decode_input(b"\x1bM\x01\x02\x03"), ([], b""))

    def test_unknown_csi_swallowed(self):
        self.assertEqual(t2._decode_input(b"\x1b[99;99u"), ([], b""))

    def test_mixed_stream(self):
        keys, rem = t2._decode_input(b"a\x1b[Bb")
        self.assertEqual(keys, [ord("a"), t2.K_DOWN, ord("b")])
        self.assertEqual(rem, b"")

    def test_trailing_partial_held_with_complete_before(self):
        keys, rem = t2._decode_input(b"\x1b[A\x1b[5")
        self.assertEqual(keys, [t2.K_UP])
        self.assertEqual(rem, b"\x1b[5")


if __name__ == "__main__":
    unittest.main()
