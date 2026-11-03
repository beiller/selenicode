import asyncio
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bus import Bus, Error, History, Interrupt, Token, TurnDone, UserMessage
from clients.common import consume_tokens


class TestBus(unittest.TestCase):
    def test_outbound_round_trip(self):
        async def scenario():
            b = Bus()
            b.outbound.put_nowait(Token("u1", "assistant", "hi"))
            return await b.outbound.get()

        ev = asyncio.run(scenario())
        self.assertEqual(ev.text, "hi")
        self.assertTrue(ev.chunk)

    def test_post_inbound_same_thread(self):
        async def scenario():
            b = Bus()
            b.bind()
            b.post(UserMessage("u1", "hello", "m1"))
            return await b.inbound.get()

        ev = asyncio.run(scenario())
        self.assertEqual(ev.msg_id, "m1")

    def test_history_outbound_round_trip(self):
        async def scenario():
            b = Bus()
            b.outbound.put_nowait(History([{"role": "user", "content": "hi"}]))
            return await b.outbound.get()

        ev = asyncio.run(scenario())
        self.assertIsInstance(ev, History)
        self.assertEqual(ev.messages[0]["content"], "hi")

    def test_interrupt_flag_set_on_post_without_draining(self):
        async def scenario():
            b = Bus()
            b.bind()
            b.post(Interrupt("u9"))
            return "u9" in b.interrupted

        self.assertTrue(asyncio.run(scenario()))

    def test_post_from_other_thread(self):
        async def scenario():
            b = Bus()
            b.bind()
            posted = []

            def worker():
                b.post(Interrupt("u2"))
                b.post(TurnDone("m9"))
                posted.append(True)

            t = threading.Thread(target=worker)
            t.start()
            first = await asyncio.wait_for(b.inbound.get(), timeout=5)
            second = await asyncio.wait_for(b.inbound.get(), timeout=5)
            t.join(timeout=5)
            return first, second, posted

        first, second, posted = asyncio.run(scenario())
        self.assertIsInstance(first, Interrupt)
        self.assertIsInstance(second, TurnDone)
        self.assertEqual(posted, [True])

    def test_post_before_bind_raises(self):
        b = Bus()
        with self.assertRaises(RuntimeError):
            b.post(Interrupt("u1"))


class TestConsumeTokens(unittest.TestCase):
    def test_error_event_routed_to_on_error(self):
        async def scenario():
            b = Bus()
            errors, completes = [], []

            async def on_complete(user_id, role, text):
                completes.append((user_id, role, text))

            async def on_error(user_id, text):
                errors.append((user_id, text))

            task = asyncio.create_task(consume_tokens(b, on_complete, on_error))
            b.outbound.put_nowait(Error("u1", "boom"))
            b.outbound.put_nowait(Token("u1", "assistant", "hi", False))
            for _ in range(100):
                if errors and completes:
                    break
                await asyncio.sleep(0.01)
            task.cancel()
            return errors, completes

        errors, completes = asyncio.run(scenario())
        self.assertEqual(errors, [("u1", "boom")])
        self.assertEqual(completes, [("u1", "assistant", "hi")])


if __name__ == "__main__":
    unittest.main()
