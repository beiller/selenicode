import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main
from core import bus as bus_mod
from core import config


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

    def test_tool_call_emit_carries_args_for_diff(self):
        def edit_file(file_path: str = "", old_string: str = "", new_string: str = "") -> str:
            return "ok"

        b = bus_mod.Bus()
        with mock.patch.object(main, "bus", b):
            asyncio.run(
                main.execute_tool_calls(
                    make_assistant_msg("edit_file", '{"file_path": "a.py", "old_string": "x\\ny", "new_string": "x\\nZ"}'),
                    {"edit_file": edit_file},
                    "user1",
                )
            )
        ev = b.outbound.get_nowait()
        self.assertIsInstance(ev, bus_mod.Token)
        self.assertEqual(ev.role, "tool")
        self.assertEqual(ev.extra["name"], "edit_file")
        self.assertEqual(ev.extra["args"]["new_string"], "x\nZ")

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

    def test_mid_loop_compaction_posts_done(self):
        b = bus_mod.Bus()
        messages = [{"role": "user", "content": "hi"}]

        async def fake_stream(base_url, msgs, tools, session_id, user_id):
            yield {"choices": [{"delta": {"content": "ok"}}]}

        async def ok_compact(base_url, messages, session_id, context_size, model):
            return True

        async def scenario():
            b.bind()
            with mock.patch.object(main, "bus", b), \
                 mock.patch.object(main, "stream_chat_completion", fake_stream), \
                 mock.patch.object(main, "get_tools", lambda: ([], {})), \
                 mock.patch.object(main, "compact_messages", ok_compact), \
                 mock.patch.object(main, "CONTEXT_SIZE", 16384), \
                 mock.patch.object(main, "compaction_trigger", lambda s: 100), \
                 mock.patch.object(main, "context_token_count", side_effect=[5000, 800]):
                return await main.run_tool_loop("http://x", "u1", messages, "sess")

        asyncio.run(scenario())
        events = []
        while not b.outbound.empty():
            events.append(b.outbound.get_nowait())
        done = [e for e in events if isinstance(e, bus_mod.CompactionDone)]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0].tokens_after, 800)


class TestAsyncMainHistory(unittest.TestCase):
    def _run_async_main(self, client_type, session_id):
        async def fake_run(bus, context_size=0):
            return None

        async def fake_load():
            return None

        async def fake_main(resume=False, session_id=None):
            return None

        async def fake_ctx():
            return None

        fake_client = types.SimpleNamespace(run=fake_run)
        with mock.patch.object(main, "_load_client", lambda ct: fake_client), mock.patch.object(
            main, "startup_load_model", fake_load
        ), mock.patch.object(main, "main", fake_main), mock.patch.object(
            main, "wait_for_context_size", fake_ctx
        ), mock.patch.object(
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

    def test_no_history_for_discord_or_fresh_session(self):
        events = self._run_async_main("discord_client", "sess1234")
        self.assertFalse(any(isinstance(e, bus_mod.History) for e in events))
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
             mock.patch.object(main, "write_conversation", lambda sid, msgs: written.append((sid, msgs))), \
             mock.patch.object(main, "CONTEXT_SIZE", 16384):
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
        self.assertIn("refusing", status.get("error", ""))
        self.assertEqual(written, [])

    def test_reset_refuses_to_wipe_nonempty_session(self):
        status, written = self._call_reset("wrong_user")
        self.assertIn("refusing", status.get("error", ""))
        self.assertEqual(written, [])

    def test_compact_proceeds_when_memory_has_messages(self):
        main.user_registry["good_user"] = list(self.DISK_MSGS)
        try:
            status, written = self._call_compact("good_user")
        finally:
            main.user_registry.pop("good_user", None)
        self.assertNotIn("error", status)
        self.assertEqual(len(written), 1)

    def test_reset_proceeds_when_memory_has_messages(self):
        main.user_registry["good_user"] = list(self.DISK_MSGS)
        try:
            status, written = self._call_reset("good_user")
        finally:
            main.user_registry.pop("good_user", None)
        self.assertNotIn("error", status)
        self.assertEqual(len(written), 1)

    def test_guard_allows_empty_write_when_disk_missing(self):
        written = []

        async def fake_compact(base_url, messages, session_id, context_size, model):
            return None

        def missing(sid):
            raise FileNotFoundError(sid)

        with mock.patch.object(main, "read_conversation", missing), \
             mock.patch.object(main, "compact_messages", fake_compact), \
             mock.patch.object(main, "write_conversation", lambda sid, msgs: written.append((sid, msgs))), \
             mock.patch.object(main, "CONTEXT_SIZE", 16384):
            status = json.loads(asyncio.run(main.compact_session("any_user", "sess1")))
        self.assertNotIn("error", status)
        self.assertEqual(len(written), 1)


class TestErrorReporting(unittest.TestCase):
    def _drain(self, b):
        events = []
        while not b.outbound.empty():
            events.append(b.outbound.get_nowait())
        return events

    def test_report_error_posts_error_event_and_logs(self):
        b = bus_mod.Bus()
        with mock.patch.object(main, "bus", b), self.assertLogs(level="ERROR") as logs:
            main.report_error("u1", "something failed", ValueError("boom"))
        ev = b.outbound.get_nowait()
        self.assertIsInstance(ev, bus_mod.Error)
        self.assertEqual(ev.user_id, "u1")
        self.assertEqual(ev.text, "something failed: boom")
        self.assertTrue(any("boom" in line for line in logs.output))

    def test_turn_exception_is_reported_and_loop_continues(self):
        b = bus_mod.Bus()
        calls = {"n": 0}

        async def flaky_handle(text, base_url, user_id, session_id, msg_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("turn blew up")

        async def scenario():
            b.bind()
            with mock.patch.object(main, "bus", b), mock.patch.object(
                main, "handle_message", flaky_handle
            ), mock.patch.object(main, "CONTEXT_SIZE", 16384):
                b.post(bus_mod.UserMessage("u1", "hello", "m1"))
                b.post(bus_mod.UserMessage("u1", "again", "m2"))
                b.post(bus_mod.Shutdown())
                await main.main()

        asyncio.run(scenario())
        self.assertEqual(calls["n"], 2)
        errors = [e for e in self._drain(b) if isinstance(e, bus_mod.Error)]
        self.assertEqual(len(errors), 1)
        self.assertIn("turn blew up", errors[0].text)

    def test_compact_command_error_is_reported(self):
        b = bus_mod.Bus()

        async def failing_compact(user_id="", session_id=""):
            return json.dumps({"error": "compaction failed; context still 999 tokens"})

        async def scenario():
            b.bind()
            with mock.patch.object(main, "bus", b), mock.patch.object(
                main, "compact_session", failing_compact
            ), mock.patch.object(main, "CONTEXT_SIZE", 16384):
                b.post(bus_mod.UserMessage("u1", "/compact", "m1"))
                b.post(bus_mod.Shutdown())
                await main.main()

        asyncio.run(scenario())
        events = self._drain(b)
        errors = [e for e in events if isinstance(e, bus_mod.Error)]
        self.assertEqual(len(errors), 1)
        self.assertIn("compaction failed", errors[0].text)
        self.assertTrue(any(isinstance(e, bus_mod.TurnDone) for e in events))

    def test_compact_command_posts_done_after_turn_done(self):
        b = bus_mod.Bus()

        async def ok_compact(user_id="", session_id=""):
            return json.dumps({"status": "Compaction complete. Context size is 800 of 16384"})

        async def scenario():
            b.bind()
            with mock.patch.object(main, "bus", b), mock.patch.object(
                main, "compact_session", ok_compact
            ), mock.patch.object(main, "CONTEXT_SIZE", 16384), mock.patch.object(
                main, "context_token_count", lambda m: 800
            ), mock.patch.object(main, "get_user_messages", lambda uid: [{"role": "user"}]):
                b.post(bus_mod.UserMessage("u1", "/compact", "m1"))
                b.post(bus_mod.Shutdown())
                await main.main()

        asyncio.run(scenario())
        events = self._drain(b)
        idx_turn = next(i for i, e in enumerate(events) if isinstance(e, bus_mod.TurnDone))
        idx_compact = next(i for i, e in enumerate(events) if isinstance(e, bus_mod.CompactionDone))
        self.assertLess(idx_turn, idx_compact)
        self.assertEqual(events[idx_compact].tokens_after, 800)
        # the completion message moved to the status line; no chat-stream info block
        self.assertFalse(any(isinstance(e, bus_mod.Token) and "Compaction complete" in e.text for e in events))

    def test_handle_message_posts_compaction_done_after_turn_done(self):
        b = bus_mod.Bus()

        async def no_op_loop(base_url, user_id, messages, session_id):
            return "ok"

        async def ok_compact(base_url, messages, session_id, context_size, model):
            return True

        async def scenario():
            b.bind()
            with mock.patch.object(main, "bus", b), \
                 mock.patch.object(main, "run_tool_loop", no_op_loop), \
                 mock.patch.object(main, "compact_messages", ok_compact), \
                 mock.patch.object(main, "CONTEXT_SIZE", 16384), \
                 mock.patch.object(main, "compaction_trigger", lambda s: 100), \
                 mock.patch.object(main, "context_token_count", lambda m: 5000), \
                 mock.patch.object(main, "approximate_token_count", lambda s: 800), \
                 mock.patch.object(main, "get_user_messages", lambda uid: [{"role": "user"}]), \
                 mock.patch.object(main, "write_conversation", lambda *a: None):
                await main.handle_message("hi", "http://x", "u1", "sess", "m1")

        asyncio.run(scenario())
        events = self._drain(b)
        idx_turn = next(i for i, e in enumerate(events) if isinstance(e, bus_mod.TurnDone))
        idx_compact = next(i for i, e in enumerate(events) if isinstance(e, bus_mod.CompactionDone))
        self.assertLess(idx_turn, idx_compact)
        self.assertEqual(events[idx_compact].tokens_after, 800)

    def test_handle_message_no_compaction_posts_no_done(self):
        b = bus_mod.Bus()

        async def no_op_loop(base_url, user_id, messages, session_id):
            return "ok"

        async def scenario():
            b.bind()
            with mock.patch.object(main, "bus", b), \
                 mock.patch.object(main, "run_tool_loop", no_op_loop), \
                 mock.patch.object(main, "CONTEXT_SIZE", 16384), \
                 mock.patch.object(main, "compaction_trigger", lambda s: 10**9), \
                 mock.patch.object(main, "context_token_count", lambda m: 1), \
                 mock.patch.object(main, "get_user_messages", lambda uid: [{"role": "user"}]), \
                 mock.patch.object(main, "write_conversation", lambda *a: None):
                await main.handle_message("hi", "http://x", "u1", "sess", "m1")

        asyncio.run(scenario())
        self.assertFalse(any(isinstance(e, bus_mod.CompactionDone) for e in self._drain(b)))


class TestRunDaemon(unittest.TestCase):
    def test_run_daemon_returns_result(self):
        async def scenario():
            return await main._run_daemon(lambda x: x * 2, 21)

        self.assertEqual(asyncio.run(scenario()), 42)

    def test_run_daemon_cancellation_does_not_wait_for_work(self):
        import time

        def slow(x):
            time.sleep(30)
            return x

        async def scenario():
            task = asyncio.create_task(main._run_daemon(slow, 1))
            await asyncio.sleep(0.05)
            t0 = time.monotonic()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            return time.monotonic() - t0

        elapsed = asyncio.run(scenario())
        self.assertLess(elapsed, 2)


class TestContextSizeGate(unittest.TestCase):
    def setUp(self):
        main.CONTEXT_SIZE = None
        main.ctx_ready.clear()
        main.shutdown.clear()

    def tearDown(self):
        main.CONTEXT_SIZE = None
        main.ctx_ready.clear()
        main.shutdown.clear()

    def test_wait_for_context_size_success_sets_global_and_event(self):
        async def scenario():
            with mock.patch.object(
                main, "_fetch_models_once", return_value=[{"id": "other", "max_model_len": 248320}]
            ):
                task = asyncio.create_task(main.wait_for_context_size())
                for _ in range(100):
                    if main.ctx_ready.is_set():
                        break
                    await asyncio.sleep(0.01)
                task.cancel()

        asyncio.run(scenario())
        self.assertEqual(main.CONTEXT_SIZE, 248320)
        self.assertTrue(main.ctx_ready.is_set())
        events = []
        try:
            while True:
                events.append(main.bus.outbound.get_nowait())
        except Exception:
            pass
        self.assertTrue(any(isinstance(e, bus_mod.ServerStatus) and e.connected for e in events))

    def test_wait_for_context_size_retries_and_surfaces_each_failed_attempt(self):
        calls = {"n": 0}
        errors = []

        def flaky_fetch(base_url):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("server down")
            return [{"id": "other", "max_model_len": 4096}]

        async def scenario():
            with mock.patch.object(main, "_fetch_models_once", flaky_fetch), mock.patch.object(
                main, "report_error", lambda uid, msg, exc=None: errors.append(msg)
            ):
                task = asyncio.create_task(main.wait_for_context_size())
                for _ in range(300):
                    if main.ctx_ready.is_set():
                        break
                    await asyncio.sleep(0.01)
                task.cancel()

        asyncio.run(scenario())
        self.assertEqual(calls["n"], 3)
        self.assertEqual(main.CONTEXT_SIZE, 4096)
        self.assertGreaterEqual(len(errors), 2)
        self.assertIn("attempt 1", errors[0])
        self.assertIn("attempt 2", errors[1])

    def test_client_starts_without_waiting_for_server_work(self):
        flag = {"run_called": False}

        async def hanging_load():
            await asyncio.sleep(30)

        async def hanging_ctx():
            await asyncio.sleep(30)

        async def fake_run(bus, context_size=0):
            flag["run_called"] = True
            return None

        async def scenario():
            with mock.patch.object(main, "_load_client", lambda ct: types.SimpleNamespace(run=fake_run)), \
                 mock.patch.object(main, "startup_load_model", hanging_load), \
                 mock.patch.object(main, "wait_for_context_size", hanging_ctx):
                await asyncio.wait_for(main.async_main("terminal", None), timeout=2)

        asyncio.run(scenario())
        self.assertTrue(flag["run_called"])

    def test_exit_is_not_blocked_by_inflight_worker_threads(self):
        import time

        def blocking_work():
            time.sleep(30)

        async def hanging_load():
            await main._run_daemon(blocking_work)

        async def hanging_ctx():
            await main._run_daemon(blocking_work)

        async def fake_run(bus, context_size=0):
            return None  # simulates ctrl+q: the UI exits immediately

        async def scenario():
            with mock.patch.object(main, "_load_client", lambda ct: types.SimpleNamespace(run=fake_run)), \
                 mock.patch.object(main, "startup_load_model", hanging_load), \
                 mock.patch.object(main, "wait_for_context_size", hanging_ctx):
                await asyncio.wait_for(main.async_main("terminal", None), timeout=2)

        t0 = time.monotonic()
        asyncio.run(scenario())
        self.assertLess(time.monotonic() - t0, 10)

    def test_turn_gated_until_context_size_resolved(self):
        b = bus_mod.Bus()
        ran = {"n": 0}

        async def ok_handle(text, base_url, user_id, session_id, msg_id):
            ran["n"] += 1

        async def scenario():
            b.bind()
            with mock.patch.object(main, "bus", b), mock.patch.object(main, "handle_message", ok_handle):
                b.post(bus_mod.UserMessage("u1", "hello", "m1"))
                b.post(bus_mod.Shutdown())
                task = asyncio.create_task(main.main())
                await asyncio.sleep(0.05)
                self.assertEqual(ran["n"], 0)
                main.CONTEXT_SIZE = 8192
                main.ctx_ready.set()
                await task

        asyncio.run(scenario())
        self.assertEqual(ran["n"], 1)


    def test_fetch_models_sends_api_key_header_only_when_set(self):
        old = config.API_KEY
        try:
            for key, expect in (("sekret", "Bearer sekret"), ("", None)):
                config.API_KEY = key
                with mock.patch.object(main.urllib.request, "Request") as fake_req, mock.patch.object(
                    main.urllib.request, "urlopen"
                ) as fake_open:
                    fake_open.return_value.__enter__.return_value.read.return_value = b'{"data": []}'
                    main._fetch_models_once("http://x")
                headers = fake_req.call_args.kwargs.get("headers", {})
                self.assertEqual(headers.get("Authorization"), expect)
        finally:
            config.API_KEY = old


class TestServerCommand(unittest.TestCase):
    def setUp(self):
        self._old_url = config.BASE_URL
        self._old_key = config.API_KEY
        config.BASE_URL = "http://old.example:8080"
        config.API_KEY = "oldkey"
        main.CONTEXT_SIZE = 16384
        # fresh events: asyncio.Event binds to the first loop that uses it and never rebinds
        main.ctx_ready = asyncio.Event()
        main.ctx_ready.set()
        main.ctx_task = None

    def tearDown(self):
        config.BASE_URL = self._old_url
        config.API_KEY = self._old_key
        main.CONTEXT_SIZE = None
        main.ctx_task = None
        main.ctx_ready = asyncio.Event()

    def test_server_switches_url_and_resets_gate(self):
        spawned = {"n": 0}

        async def fake_wait():
            spawned["n"] += 1
            await asyncio.sleep(30)

        async def old_fetcher():
            await asyncio.sleep(30)  # simulates the retry loop, possibly mid-backoff

        async def scenario():
            b = bus_mod.Bus()
            with mock.patch.object(main, "bus", b), mock.patch.object(
                main, "wait_for_context_size", fake_wait
            ):
                b.bind()
                old = asyncio.create_task(old_fetcher())
                main.ctx_task = old
                b.post(bus_mod.UserMessage("u1", "/server http://new.example:9000 secret123", "m1"))
                b.post(bus_mod.Shutdown())
                await asyncio.wait_for(asyncio.create_task(main.main()), timeout=2)
                await asyncio.sleep(0.01)
                self.assertTrue(old.cancelled())  # stale retry loop must not delay the switch

        asyncio.run(scenario())
        self.assertEqual(config.BASE_URL, "http://new.example:9000")  # used verbatim
        self.assertEqual(config.API_KEY, "secret123")
        self.assertIsNone(main.CONTEXT_SIZE)
        self.assertFalse(main.ctx_ready.is_set())
        self.assertEqual(spawned["n"], 1)  # fresh fetcher starts immediately against the new URL

    def test_server_bad_args_report_error_and_leave_state_untouched(self):
        errors = []

        async def scenario():
            b = bus_mod.Bus()
            with mock.patch.object(main, "bus", b), mock.patch.object(
                main, "report_error", lambda uid, msg, exc=None: errors.append(msg)
            ):
                b.bind()
                b.post(bus_mod.UserMessage("u1", "/server", "m1"))
                b.post(bus_mod.UserMessage("u1", "/server a b", "m2"))
                b.post(bus_mod.UserMessage("u1", "/server newhost.example", "m3"))  # no scheme
                b.post(bus_mod.Shutdown())
                await asyncio.wait_for(asyncio.create_task(main.main()), timeout=2)

        asyncio.run(scenario())
        self.assertEqual(config.BASE_URL, "http://old.example:8080")
        self.assertEqual(config.API_KEY, "oldkey")  # bad args must not touch the key either
        self.assertEqual(main.CONTEXT_SIZE, 16384)
        self.assertTrue(main.ctx_ready.is_set())
        self.assertGreaterEqual(len(errors), 3)

    def test_turn_gated_again_after_server_switch(self):
        ran = {"n": 0}

        async def ok_handle(text, base_url, user_id, session_id, msg_id):
            ran["n"] += 1

        async def scenario():
            b = bus_mod.Bus()
            with mock.patch.object(main, "bus", b), mock.patch.object(
                main, "handle_message", ok_handle
            ), mock.patch.object(
                main, "wait_for_context_size", lambda: asyncio.sleep(30)
            ):
                b.bind()
                b.post(bus_mod.UserMessage("u1", "/server http://new.example:9000", "m1"))  # bare URL clears the key
                b.post(bus_mod.UserMessage("u1", "hello", "m2"))
                task = asyncio.create_task(main.main())
                await asyncio.sleep(0.1)
                self.assertEqual(ran["n"], 0)  # the turn must wait for the new server's context size
                main.CONTEXT_SIZE = 8192
                main.ctx_ready.set()
                b.post(bus_mod.Shutdown())
                await asyncio.wait_for(task, timeout=2)

        asyncio.run(scenario())
        self.assertEqual(ran["n"], 1)
        self.assertEqual(config.API_KEY, "")  # bare URL cleared the previously set key

    def test_server_interrupts_a_prompt_blocked_at_the_gate(self):
        """Deadlock regression: a /server typed while a prompt blocks at the gate must be served immediately."""
        handled = []

        async def ok_handle(text, base_url, user_id, session_id, msg_id):
            handled.append(text)

        async def scenario():
            b = bus_mod.Bus()
            with mock.patch.object(main, "bus", b), mock.patch.object(
                main, "handle_message", ok_handle
            ), mock.patch.object(main, "wait_for_context_size", lambda: asyncio.sleep(30)):
                b.bind()
                main.CONTEXT_SIZE = None
                main.ctx_ready = asyncio.Event()  # fresh event: binds to the first loop that uses it
                b.post(bus_mod.UserMessage("u1", "hello", "m1"))  # blocks at the gate
                task = asyncio.create_task(main.main())
                await asyncio.sleep(0.05)
                self.assertEqual(handled, [])  # still gated
                b.post(bus_mod.UserMessage("u1", "/server http://new.example:9000", "m2"))
                for _ in range(300):
                    if config.BASE_URL == "http://new.example:9000":
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(config.BASE_URL, "http://new.example:9000")  # served while gated (no deadlock)
                main.CONTEXT_SIZE = 8192
                main.ctx_ready.set()
                b.post(bus_mod.Shutdown())
                await asyncio.wait_for(task, timeout=2)

        old_url = config.BASE_URL
        try:
            config.BASE_URL = "http://old.example:8080"
            asyncio.run(scenario())
        finally:
            config.BASE_URL = old_url
        self.assertEqual(handled, ["hello"])  # the gated prompt runs after the gate opens, in order


class TestHelpCommand(unittest.TestCase):
    def setUp(self):
        self._old_ctx = main.CONTEXT_SIZE
        main.CONTEXT_SIZE = None  # gate unresolved: /help must not block on the server

    def tearDown(self):
        main.CONTEXT_SIZE = self._old_ctx

    def test_help_lists_commands_pre_gate(self):
        events = []

        async def scenario():
            b = bus_mod.Bus()
            with mock.patch.object(main, "bus", b):
                b.bind()
                b.post(bus_mod.UserMessage("u1", "/help", "m1"))
                b.post(bus_mod.Shutdown())
                await asyncio.wait_for(asyncio.create_task(main.main()), timeout=2)
                while True:
                    try:
                        events.append(b.outbound.get_nowait())
                    except Exception:
                        break

        asyncio.run(scenario())
        help_texts = [e.text for e in events if isinstance(e, bus_mod.Token)]
        self.assertTrue(any("/compact" in t and "/server" in t and "/help" in t for t in help_texts))
        self.assertTrue(any(isinstance(e, bus_mod.TurnDone) for e in events))


class TestVisionHelpers(unittest.TestCase):
    def test_tool_output_to_content_marker_becomes_parts(self):
        out = main._tool_output_to_content(json.dumps({"type": "image", "path": "/a/b.png", "mime": "image/png"}))
        self.assertIsInstance(out, list)
        self.assertEqual(out[0]["type"], "text")
        self.assertIn("/a/b.png", out[0]["text"])
        self.assertEqual(out[1]["type"], "image_ref")
        self.assertEqual(out[1]["path"], "/a/b.png")
        self.assertEqual(out[1]["mime"], "image/png")

    def test_tool_output_to_content_plain_passes_through(self):
        self.assertEqual(main._tool_output_to_content("hello"), "hello")
        self.assertEqual(main._tool_output_to_content(json.dumps({"error": "nope"})), json.dumps({"error": "nope"}))

    def test_attach_images_encodes_and_keeps_history_form(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.png")
            with open(path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
            content = [
                {"type": "text", "text": "Image loaded from x.png (image/png)"},
                {"type": "image_ref", "path": path, "mime": "image/png"},
            ]
            encoded = main._attach_images(content)
        self.assertEqual(encoded[1]["type"], "image_url")
        self.assertTrue(encoded[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        # history (the input list) is untouched
        self.assertEqual(content[1]["type"], "image_ref")
        # text part is copied, not shared
        self.assertIsNot(encoded[0], content[0])

    def test_attach_images_missing_file_falls_back_to_text(self):
        encoded = main._attach_images([{"type": "image_ref", "path": "/no/such/file.png", "mime": "image/png"}])
        self.assertEqual(encoded[0]["type"], "text")
        self.assertIn("image unavailable", encoded[0]["text"])

    def test_scrub_media_marker(self):
        self.assertEqual(main._scrub_media_marker('a message contains "<__media__>"'), 'a message contains "[media]"')
        # no marker -> same object, not a copy
        plain = "plain text"
        self.assertIs(main._scrub_media_marker(plain), plain)

    def test_attach_images_scrubs_reserved_marker_from_text_parts(self):
        content = [
            {"type": "text", "text": 'return "<__media__>";'},
            {"type": "image_ref", "path": "/no/such/file.png", "mime": "image/png"},
        ]
        encoded = main._attach_images(content)
        self.assertNotIn(main.MEDIA_MARKER, encoded[0]["text"])
        self.assertIn("[media]", encoded[0]["text"])
        # history keeps the literal marker (only the send copy is scrubbed)
        self.assertEqual(content[0]["text"], 'return "<__media__>";')

    def test_prepare_message_scrubs_string_content_and_reasoning(self):
        msg = {
            "role": "user",
            "content": 'error: a message contains "<__media__>"',
            "reasoning_content": "the marker is <__media__>",
            "timestamp": "2026-11-03T14:35:00",
        }
        out = main._prepare_message(msg)
        self.assertNotIn(main.MEDIA_MARKER, out["content"])
        self.assertIn("[media]", out["content"])
        self.assertNotIn(main.MEDIA_MARKER, out["reasoning_content"])
        self.assertNotIn("timestamp", out)
        # original message untouched
        self.assertIn(main.MEDIA_MARKER, msg["content"])
        self.assertIn(main.MEDIA_MARKER, msg["reasoning_content"])

    def test_prepare_message_plain_strings_pass_through(self):
        msg = {"role": "user", "content": "hello"}
        out = main._prepare_message(msg)
        self.assertIs(out["content"], "hello")

    def test_prepare_message_scrubs_marker_in_tool_call_arguments(self):
        # regression: a resumed session whose tool-call arguments quote the marker
        # (e.g. the edit that adds the scrub itself) 400'd once the request carried media
        msg = {
            "role": "assistant",
            "content": None,
            "reasoning_content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "edit_file", "arguments": json.dumps({"new_string": 'MEDIA_MARKER = "<__media__>"'})},
                }
            ],
            "timestamp": "2026-11-03T14:35:00",
        }
        out = main._prepare_message(msg)
        self.assertNotIn(main.MEDIA_MARKER, json.dumps(out))
        self.assertIn("[media]", out["tool_calls"][0]["function"]["arguments"])
        # original message untouched
        self.assertIn(main.MEDIA_MARKER, msg["tool_calls"][0]["function"]["arguments"])

    def test_parse_vision_supported(self):
        with mock.patch.object(main, "CURRENT_MODEL", "m1"):
            self.assertTrue(main._parse_vision_supported([{"id": "m1", "capabilities": ["completion", "multimodal"]}]))
            self.assertFalse(main._parse_vision_supported([{"id": "m1", "capabilities": ["completion"]}]))
            # no model with vision at all -> False (the id-mismatch case falls back to any model)
            self.assertFalse(main._parse_vision_supported([{"id": "other", "capabilities": ["completion"]}]))
            self.assertTrue(main._parse_vision_supported([{"id": "m1", "capabilities": {"chat": True, "image": True}}]))
            self.assertFalse(main._parse_vision_supported([{"id": "m1", "capabilities": {"chat": True, "image": False}}]))
            # server may report a different id than the configured name: fall back to any model
            self.assertTrue(main._parse_vision_supported([{"id": "other", "capabilities": {"image": True}}]))


class TestStartupLoadModel(unittest.TestCase):
    def _run_startup(self, load_exc, loaded_model, get_loaded_exc=None):
        calls = []

        async def fake_run_daemon(fn, *args):
            calls.append(fn.__name__)
            if fn is main.load_model_and_set:
                if load_exc is not None:
                    raise load_exc
                return None
            if fn is main.get_loaded_model:
                if get_loaded_exc is not None:
                    raise get_loaded_exc
                return loaded_model
            raise AssertionError(f"unexpected function {fn}")

        with mock.patch.object(main, "CURRENT_MODEL", "m1"), mock.patch.object(main, "_run_daemon", fake_run_daemon):
            asyncio.run(main.startup_load_model())
        return calls

    def test_unsupported_endpoint_with_preloaded_model_warns(self):
        import urllib.error

        exc = urllib.error.HTTPError("http://x/models/load", 404, "Not Found", {}, None)
        with self.assertLogs("root", level="WARNING") as logs:
            calls = self._run_startup(exc, "qwen35")
        self.assertIn("get_loaded_model", calls)
        self.assertTrue(any("does not support dynamic model loading" in m and "'qwen35'" in m for m in logs.output))

    def test_unsupported_endpoint_without_preloaded_model_errors(self):
        import urllib.error

        exc = urllib.error.HTTPError("http://x/models/load", 404, "Not Found", {}, None)
        with self.assertLogs("root", level="ERROR") as logs:
            calls = self._run_startup(exc, None)
        self.assertIn("get_loaded_model", calls)
        self.assertTrue(any("cannot load one" in m for m in logs.output))

    def test_unsupported_endpoint_lookup_failure_errors(self):
        import urllib.error

        exc = urllib.error.HTTPError("http://x/models/load", 405, "Method Not Allowed", {}, None)
        with self.assertLogs("root", level="ERROR") as logs:
            self._run_startup(exc, None, get_loaded_exc=RuntimeError("server down"))
        self.assertTrue(any("cannot load one" in m for m in logs.output))

    def test_other_failure_errors_without_lookup(self):
        with self.assertLogs("root", level="ERROR") as logs:
            calls = self._run_startup(ConnectionError("refused"), "qwen35")
        self.assertNotIn("get_loaded_model", calls)
        self.assertTrue(any("Model load at startup failed" in m for m in logs.output))

    def test_success_no_lookup(self):
        calls = self._run_startup(None, None)
        self.assertEqual(calls, ["load_model_and_set"])


if __name__ == "__main__":
    unittest.main()
