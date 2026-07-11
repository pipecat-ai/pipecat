#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

import pytest

# Mem0 is an optional dependency; skip the whole module if it isn't installed.
pytest.importorskip("mem0")

from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.services.mem0.memory import Mem0MemoryService  # noqa: E402


class _FakeClient:
    """Stand-in for a Mem0 client that returns the 2.x {"results": [...]} shape."""

    def __init__(self, results):
        self._results = results

    def search(self, *args, **kwargs):
        return {"results": self._results}


def _make_service(results):
    """Build a Mem0MemoryService without touching the real Mem0 client/__init__."""
    svc = Mem0MemoryService.__new__(Mem0MemoryService)
    svc.memory_client = _FakeClient(results)
    svc.user_id = "u1"
    svc.agent_id = None
    svc.run_id = None
    svc.search_limit = 10
    svc.search_threshold = 0.1
    svc.last_query = None
    svc.system_prompt = "Based on previous conversations, I recall: \n\n"
    svc.add_as_system_message = True
    svc.position = 1
    return svc


class TestMem0EnhanceContext(unittest.IsolatedAsyncioTestCase):
    async def test_no_memories_injects_nothing(self):
        """With no relevant memories, no recall message is injected (see mem0 bug)."""
        svc = _make_service([])
        ctx = LLMContext(messages=[{"role": "user", "content": "hello"}])

        await svc._enhance_context_with_memories(ctx, "hello")

        self.assertEqual(ctx.get_messages(), [{"role": "user", "content": "hello"}])

    async def test_memories_are_injected(self):
        """Relevant memories are formatted and inserted into the context."""
        svc = _make_service([{"memory": "User likes coffee"}, {"memory": "User is in Lisbon"}])
        ctx = LLMContext(messages=[{"role": "user", "content": "hello"}])

        await svc._enhance_context_with_memories(ctx, "hello")

        injected = [m for m in ctx.get_messages() if "I recall" in str(m.get("content", ""))]
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0]["role"], "system")
        self.assertIn("User likes coffee", injected[0]["content"])
        self.assertIn("User is in Lisbon", injected[0]["content"])


if __name__ == "__main__":
    unittest.main()
