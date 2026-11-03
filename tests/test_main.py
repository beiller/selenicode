import asyncio
import json
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bus as bus_mod
import main


def make_assistant_msg(name: str, args: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": name, "arguments": args}}],
    }


class TestExecuteToolCalls(unittest.TestCase):
    def test_sync_tool_handler(self):
        def mytool(x: str = "") -> str:
            return "sync-" + x

        results = asyncio.run(
            main.execute_tool_calls(make_assistant_msg("mytool", '{"x": "hi"}'), {"mytool": mytool}, "user1")
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1]["role"], "tool")
        self.assertEqual(results[1]["content"], "sync-hi")

    def test_async_tool_handler_is_awaited(self):
        async def mytool(x: str = "") -> str:
            await asyncio.sleep(0)
            return "async-" + x

        results = asyncio.run(
            main.execute_tool_calls(make_assistant_msg("mytool", '{"x": "hi"}'), {"mytool": mytool}, "user1")
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1]["role"], "tool")
        self.assertEqual(results[1]["content"], "async-hi")

    def test_compact_session_is_registered(self):
        tools, registry = main.build_tools()
        names = [t["function"]["name"] for t in tools]
        self.assertIn("compact_session", names)
        self.assertIs(registry["compact_session"], main.compact_session)


class TestInterrupt(unittest.TestCase):
    def _run_loop(self, fake_stream, set_flag=None):
        b = bus_mod.Bus()
        messages = [{"role": "user", "content": "hi"}]

        async def scenario():
            with mock.patch.object(main, "bus", b), mock.patch.object(
                main, "stream_chat_completion", fake_stream
            ), mock.patch.object(main, "get_tools", lambda: ([], {})), mock.patch.object(
                main, "compaction_trigger", lambda s: 10**9
            ), mock.patch.object(main, "context_token_count", lambda m: 1):
                if set_flag is not None:
                    flag_task = asyncio.create_task(set_flag(b))
                    result = await main.run_tool_loop("http://x", "u1", messages, "sess")
                    flag_task.cancel()
                else:
                    result = await main.run_tool_loop("http://x", "u1", messages, "sess")
            return result, messages, b

        return asyncio.run(scenario())

    def test_interrupt_before_stream_ends_turn_without_calling_model(self):
        def boom(*a):
            raise AssertionError("stream must not be started")

        b = bus_mod.Bus()
        b.interrupted.add("u1")
        with mock.patch.object(main, "bus", b), mock.patch.object(
            main, "stream_chat_completion", boom
        ), mock.patch.object(main, "get_tools", lambda: ([], {})), mock.patch.object(
            main, "compaction_trigger", lambda s: 10**9
        ), mock.patch.object(main, "context_token_count", lambda m: 1):
            result = asyncio.run(main.run_tool_loop("http://x", "u1", [{"role": "user", "content": "hi"}], "sess"))
        self.assertEqual(result, "")
        self.assertFalse(b.interrupted)

    def test_interrupt_stalled_stream_cancels_and_skips_tools(self):
        async def fake_stream(base_url, messages, tools, session_id, user_id):
            yield {"choices": [{"delta": {"content": "par"}}]}
            await asyncio.sleep(30)  # stalled: no more chunks

        async def set_flag(b):
            await asyncio.sleep(0.2)
            b.interrupted.add("u1")

        result, messages, b = self._run_loop(fake_stream, set_flag)
        self.assertEqual(result, "")
        self.assertEqual(len(messages), 1)  # no assistant or tool messages appended
        self.assertFalse(b.interrupted)

    def test_interrupt_flowing_stream_keeps_partial_and_skips_tools(self):
        b = bus_mod.Bus()
        messages = [{"role": "user", "content": "hi"}]

        async def fake_stream(base_url, msgs, tools, session_id, user_id):
            for i in range(100):
                yield {"choices": [{"delta": {"content": f"c{i}"}}]}
                if i == 3:
                    b.interrupted.add("u1")  # interrupt while chunks are flowing

        async def scenario():
            with mock.patch.object(main, "bus", b), mock.patch.object(
                main, "stream_chat_completion", fake_stream
            ), mock.patch.object(main, "get_tools", lambda: ([], {})), mock.patch.object(
                main, "compaction_trigger", lambda s: 10**9
            ), mock.patch.object(main, "context_token_count", lambda m: 1):
                return await main.run_tool_loop("http://x", "u1", messages, "sess")

        result = asyncio.run(scenario())
        self.assertEqual(result, "")
        self.assertEqual(len(messages), 2)
        partial = messages[1]
        self.assertEqual(partial["role"], "assistant")
        self.assertTrue(partial["content"].startswith("c0"))
        self.assertNotIn("tool_calls", partial)
        self.assertFalse(b.interrupted)


class TestAsyncMainHistory(unittest.TestCase):
    def _run_async_main(self, client_type, session_id):
        async def fake_run(bus, context_size=0):
            return None

        async def fake_init():
            return None

        async def fake_main(resume=False, session_id=None):
            return None

        fake_client = types.SimpleNamespace(run=fake_run)
        with mock.patch.object(main, "_load_client", lambda ct: fake_client), mock.patch.object(
            main, "main_init", fake_init
        ), mock.patch.object(main, "main", fake_main), mock.patch.object(
            main, "read_conversation", lambda sid: [{"role": "user", "content": "hi"}]
        ):
            asyncio.run(main.async_main(client_type, session_id))

        events = []
        while not main.bus.outbound.empty():
            events.append(main.bus.outbound.get_nowait())
        return events

    def test_history_posted_for_terminal_resume(self):
        events = self._run_async_main("terminal", "sess1234")
        histories = [e for e in events if isinstance(e, bus_mod.History)]
        self.assertEqual(len(histories), 1)
        self.assertEqual(histories[0].messages, [{"role": "user", "content": "hi"}])

    def test_no_history_for_discord_even_on_resume(self):
        events = self._run_async_main("discord_client", "sess1234")
        self.assertFalse(any(isinstance(e, bus_mod.History) for e in events))

    def test_no_history_for_fresh_terminal_session(self):
        events = self._run_async_main("terminal", None)
        self.assertFalse(any(isinstance(e, bus_mod.History) for e in events))


class TestSessionWipeGuard(unittest.TestCase):
    """A wrong user_id yields an empty in-memory list; compact/reset must not
    then overwrite a non-empty session file (data-loss regression)."""

    DISK_MSGS = [{"role": "user", "content": "saved"}, {"role": "assistant", "content": "hi"}]

    def _call_compact(self, user_id):
        written = []

        async def fake_compact(base_url, messages, session_id, context_size, model):
            return None

        with mock.patch.object(main, "read_conversation", lambda sid: list(self.DISK_MSGS)), \
             mock.patch.object(main, "compact_messages", fake_compact), \
             mock.patch.object(main, "write_conversation", lambda sid, msgs: written.append((sid, msgs))):
            return json.loads(asyncio.run(main.compact_session(user_id, "sess1"))), written

    def _call_reset(self, user_id):
        written = []
        with mock.patch.object(main, "read_conversation", lambda sid: list(self.DISK_MSGS)), \
             mock.patch.object(main, "archive_conversation", lambda sid, msgs: "/tmp/archive.txt"), \
             mock.patch.object(main, "set_user_messages", lambda uid, msgs: None), \
             mock.patch.object(main, "write_conversation", lambda sid, msgs: written.append((sid, msgs))):
            return json.loads(main.reset_session(user_id, "sess1")), written

    def test_compact_refuses_to_wipe_nonempty_session(self):
        status, written = self._call_compact("wrong_user")
        self.assertTrue(status["status"].startswith("error"))
        self.assertEqual(written, [])

    def test_reset_refuses_to_wipe_nonempty_session(self):
        status, written = self._call_reset("wrong_user")
        self.assertTrue(status["status"].startswith("error"))
        self.assertEqual(written, [])

    def test_compact_proceeds_when_memory_has_messages(self):
        main.user_registry["good_user"] = list(self.DISK_MSGS)
        try:
            status, written = self._call_compact("good_user")
        finally:
            main.user_registry.pop("good_user", None)
        self.assertFalse(status["status"].startswith("error"))
        self.assertEqual(len(written), 1)

    def test_reset_proceeds_when_memory_has_messages(self):
        main.user_registry["good_user"] = list(self.DISK_MSGS)
        try:
            status, written = self._call_reset("good_user")
        finally:
            main.user_registry.pop("good_user", None)
        self.assertFalse(status["status"].startswith("error"))
        self.assertEqual(len(written), 1)

    def test_guard_allows_empty_write_when_disk_missing(self):
        written = []

        async def fake_compact(base_url, messages, session_id, context_size, model):
            return None

        def missing(sid):
            raise FileNotFoundError(sid)

        with mock.patch.object(main, "read_conversation", missing), \
             mock.patch.object(main, "compact_messages", fake_compact), \
             mock.patch.object(main, "write_conversation", lambda sid, msgs: written.append((sid, msgs))):
            status = json.loads(asyncio.run(main.compact_session("any_user", "sess1")))
        self.assertFalse(status["status"].startswith("error"))
        self.assertEqual(len(written), 1)


if __name__ == "__main__":
    unittest.main()
