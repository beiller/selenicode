import asyncio
import os
import sys
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bus import Bus, Interrupt, Token, TurnDone, UserMessage


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


if __name__ == "__main__":
    unittest.main()
