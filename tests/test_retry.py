import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retry import HttpError, retry_fibonacci


class TestRetryFibonacci(unittest.TestCase):
    def test_async_succeeds_after_failures(self):
        calls = {"n": 0}

        @retry_fibonacci(max_retries=3, base_delay=0.01)
        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("busy")
            return "ok"

        self.assertEqual(asyncio.run(flaky()), "ok")
        self.assertEqual(calls["n"], 3)

    def test_sync_succeeds_after_failures(self):
        calls = {"n": 0}

        @retry_fibonacci(max_retries=2, base_delay=0.01)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise TimeoutError("slow")
            return "ok"

        self.assertEqual(flaky(), "ok")
        self.assertEqual(calls["n"], 2)

    def test_exhaustion_raises_original_exception_as_is(self):
        original = ConnectionError("down")

        @retry_fibonacci(max_retries=2, base_delay=0.01)
        async def always_fail():
            raise original

        with self.assertRaises(ConnectionError) as ctx:
            asyncio.run(always_fail())
        self.assertIs(ctx.exception, original)

    def test_args_and_kwargs_pass_through(self):
        @retry_fibonacci(max_retries=1, base_delay=0.01)
        async def add(a, b, c=0):
            return a + b + c

        self.assertEqual(asyncio.run(add(1, 2, c=5)), 8)

    def test_5xx_is_retried(self):
        calls = {"n": 0}

        @retry_fibonacci(max_retries=2, base_delay=0.01)
        async def request():
            calls["n"] += 1
            if calls["n"] < 2:
                raise HttpError(503, "busy")
            return "ok"

        self.assertEqual(asyncio.run(request()), "ok")
        self.assertEqual(calls["n"], 2)

    def test_4xx_http_error_class_raises_immediately(self):
        calls = {"n": 0}

        @retry_fibonacci(max_retries=3, base_delay=0.01)
        async def request():
            calls["n"] += 1
            raise HttpError(404, "no such endpoint")

        with self.assertRaises(HttpError):
            asyncio.run(request())
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
