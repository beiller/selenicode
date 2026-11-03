import asyncio
import json
import os
import pathlib
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import compaction

REAL_CONVERSATION = pathlib.Path(__file__).parent.parent / "conversations" / "bakupte6.json"


class CompactionBase(unittest.TestCase):
    def setUp(self):
        self.summarize_calls = []
        self.env_patcher = mock.patch.dict(
            os.environ,
            {"COMPACTION_TRIGGER_PCT": "80", "COMPACTION_TARGET_PCT": "20", "COMPACTION_TAIL_TURNS": "2"},
        )
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def run_compact(self, messages, context_size=16384, summary="SUMMARY"):
        async def stub(base_url, model, system_prompt, msgs):
            self.summarize_calls.append(msgs)
            return summary

        with mock.patch.object(compaction, "summarize_messages", stub):
            asyncio.run(compaction.compact_messages("http://localhost:1", messages, "sess", context_size, "test-model"))


class TestHeadTail(CompactionBase):
    def test_summary_inserted_as_assistant_message_after_marker(self):
        messages = [
            {"role": "user", "content": "U" * 3000},
            {"role": "assistant", "content": "A" * 3000},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        self.run_compact(messages, context_size=2000)
        self.assertEqual(messages[0]["role"], "user")
        self.assertTrue(messages[0].get("compaction_marker"))
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertTrue(messages[1].get("compaction_summary"))
        self.assertEqual(messages[1]["content"], "SUMMARY")
        self.assertEqual(messages[2]["content"], "hi")
        self.assertEqual(messages[3]["content"], "hello")

    def test_tail_never_starts_with_tool_message(self):
        messages = [
            {"role": "user", "content": "U" * 3000},
            {"role": "assistant", "content": "A" * 3000},
            {"role": "user", "content": "u2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_bash", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "T" * 8000},
            {"role": "assistant", "content": "done"},
        ]
        self.run_compact(messages, context_size=2000)
        tail_start = next(i for i, m in enumerate(messages) if m.get("compaction_marker")) + 1
        first_tail = messages[tail_start + 1]
        self.assertNotEqual(first_tail["role"], "tool")

    def test_replay_last_user_when_in_head(self):
        user = {"role": "user", "content": "U" * 3000}
        messages = [user, {"role": "assistant", "content": "A" * 3000}]
        self.run_compact(messages, context_size=1000)
        self.assertEqual(len(messages), 3)
        self.assertTrue(messages[0].get("compaction_marker"))
        self.assertEqual(messages[2]["content"], "U" * 3000)

    def test_anchored_previous_summary_on_second_compaction(self):
        messages = [
            {"role": "user", "content": "U" * 3000},
            {"role": "assistant", "content": "A" * 3000},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        self.run_compact(messages, context_size=2000, summary="SUMMARY1")
        # New conversation growth pushes the context back over the target
        messages.append({"role": "user", "content": "G" * 3000})
        messages.append({"role": "assistant", "content": "g" * 3000})
        self.run_compact(messages, context_size=2000, summary="SUMMARY2")
        prompt = self.summarize_calls[1][-1]["content"]
        self.assertIn("<previous-summary>\nSUMMARY1\n</previous-summary>", prompt)
        self.assertEqual(len([m for m in messages if m.get("compaction_marker")]), 1)
        self.assertEqual(messages[1]["content"], "SUMMARY2")

    def test_summarize_failure_raises_and_keeps_messages(self):
        async def failing(base_url, model, system_prompt, msgs):
            raise ConnectionError("server busy")

        with mock.patch.object(compaction, "summarize_messages", failing):
            messages = [
                {"role": "user", "content": "U" * 3000},
                {"role": "assistant", "content": "A" * 3000},
            ]
            with self.assertRaises(ConnectionError):
                asyncio.run(compaction.compact_messages("http://localhost:1", messages, "sess", 1000, "test-model"))
        self.assertEqual(messages[0]["content"], "U" * 3000)
        self.assertFalse(any(m.get("compaction_marker") for m in messages))

    def test_no_compaction_when_under_target(self):
        messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        self.run_compact(messages, context_size=16384)
        self.assertEqual(self.summarize_calls, [])
        self.assertEqual(messages[0]["content"], "hello")

class TestPrune(CompactionBase):
    def _prune_fixture(self, old_size):
        messages = [
            {"role": "user", "content": "u1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "run_bash", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "T" * old_size},
            {"role": "user", "content": "u2"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "run_bash", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "T" * 8000},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
        return messages, messages[2], messages[5]

    def test_prune_erases_old_tool_outputs(self):
        # Old output (3000 tokens) exceeds the 25% protect budget (2500 of 10000) and frees more than the 10% minimum
        messages, old_tool, new_tool = self._prune_fixture(12000)
        self.run_compact(messages, context_size=10000)
        self.assertEqual(old_tool["content"], compaction.PRUNED_TOOL_NOTE)
        self.assertEqual(new_tool["content"], "T" * 8000)

    def test_prune_skips_when_freed_below_minimum(self):
        messages, old_tool, _ = self._prune_fixture(400)
        self.run_compact(messages, context_size=10000)
        self.assertEqual(old_tool["content"], "T" * 400)


class TestBudget(CompactionBase):
    def test_preserve_recent_budget_clamps(self):
        with mock.patch.dict(os.environ, {"COMPACTION_TARGET_PCT": "50"}):
            self.assertEqual(compaction.preserve_recent_budget(16384), 4096)
        with mock.patch.dict(os.environ, {"COMPACTION_TARGET_PCT": "20"}):
            self.assertEqual(compaction.preserve_recent_budget(16384), 3276)
        self.assertEqual(compaction.preserve_recent_budget(1_000_000), 8000)


class TestRealConversation(CompactionBase):
    def test_compact_bakupte6_conversation(self):
        with open(REAL_CONVERSATION) as f:
            messages = json.load(f)
        # ~144k tokens, so a 500k context (20% target = 100k) is needed to trigger
        context_size = 500_000
        self.assertGreater(compaction.context_token_count(messages), compaction.compaction_target(context_size))
        original_last = messages[-1]
        self.run_compact(messages, context_size=context_size)

        marker_idx = next(i for i, m in enumerate(messages) if m.get("compaction_marker"))
        self.assertEqual(marker_idx, 0)
        self.assertTrue(messages[1].get("compaction_summary"))
        tail = messages[2:]
        self.assertGreater(len(tail), 0)
        self.assertLessEqual(compaction.context_token_count(tail), compaction.preserve_recent_budget(context_size))
        self.assertIs(tail[-1], original_last)
        self.assertNotEqual(tail[0]["role"], "tool")
        self.assertLessEqual(
            compaction.context_token_count(messages), compaction.compaction_target(context_size) + 4096
        )

    def test_recompact_bakupte6_anchors_on_previous_summary(self):
        with open(REAL_CONVERSATION) as f:
            messages = json.load(f)
        context_size = 500_000
        self.run_compact(messages, context_size=context_size, summary="SUMMARY1")
        # New conversation growth pushes the context back over the target
        messages.append({"role": "user", "content": "G" * 400_000})
        messages.append({"role": "assistant", "content": "g" * 400_000})
        self.run_compact(messages, context_size=context_size, summary="SUMMARY2")
        prompt = self.summarize_calls[1][-1]["content"]
        self.assertIn("<previous-summary>\nSUMMARY1\n</previous-summary>", prompt)
        self.assertEqual(len([m for m in messages if m.get("compaction_marker")]), 1)


if __name__ == "__main__":
    unittest.main()
