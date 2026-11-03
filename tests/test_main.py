import asyncio
import os
import sys
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


if __name__ == "__main__":
    unittest.main()
