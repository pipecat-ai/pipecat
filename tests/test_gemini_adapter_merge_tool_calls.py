#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for GeminiLLMAdapter._merge_parallel_tool_calls_for_thinking().

Tests the merging of parallel tool call and response messages when thinking is enabled:

1. Parallel interleaved calls are merged into single model and user turns
2. Sequential calls (each with own thought_signature) are not merged
3. Non-tool message boundaries prevent cross-boundary merging
4. Three or more parallel calls merge correctly
5. Single tool call passes through unchanged
6. No thought signatures means no merging (fast-exit)
7. Empty message list returns empty
8. Batch-first ordering (calls together, then responses together) merges correctly
9. Mixed parallel and sequential groups in one history are handled independently
"""

import unittest

from google.genai.types import Content, FunctionCall, FunctionResponse, Part

from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter


class TestMergeParallelToolCallsForThinking(unittest.TestCase):
    def setUp(self) -> None:
        """Sets up a common adapter instance for all tests."""
        self.adapter = GeminiLLMAdapter()

    def _make_tool_call(self, name, call_id, sig=None):
        """Create a model Content with a function_call part."""
        part = Part(function_call=FunctionCall(id=call_id, name=name, args={}))
        if sig:
            part.thought_signature = sig
        return Content(role="model", parts=[part])

    def _make_tool_response(self, name, call_id):
        """Create a user Content with a function_response part."""
        part = Part(
            function_response=FunctionResponse(id=call_id, name=name, response={"ok": True})
        )
        return Content(role="user", parts=[part])

    def _make_sig_dict(self, call_id):
        """Create a thought_signature dict for a function call."""
        return {"signature": "sig_" + call_id, "bookmark": {"function_call": call_id}}

    def test_parallel_interleaved_calls_merged(self):
        """Parallel interleaved calls and responses are merged into single turns."""
        messages = [
            self._make_tool_call("fn1", "id1", sig="sigA"),
            self._make_tool_response("fn1", "id1"),
            self._make_tool_call("fn2", "id2"),
            self._make_tool_response("fn2", "id2"),
        ]
        sigs = [self._make_sig_dict("id1")]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(sigs, messages)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].role, "model")
        self.assertEqual(len(result[0].parts), 2)
        self.assertEqual(result[0].parts[0].function_call.id, "id1")
        self.assertEqual(result[0].parts[1].function_call.id, "id2")
        self.assertEqual(result[1].role, "user")
        self.assertEqual(len(result[1].parts), 2)
        self.assertEqual(result[1].parts[0].function_response.id, "id1")
        self.assertEqual(result[1].parts[1].function_response.id, "id2")

    def test_sequential_calls_not_merged(self):
        """Sequential calls with separate thought_signatures are not merged."""
        messages = [
            self._make_tool_call("fn1", "id1", sig="sigA"),
            self._make_tool_response("fn1", "id1"),
            self._make_tool_call("fn2", "id2", sig="sigB"),
            self._make_tool_response("fn2", "id2"),
        ]
        sigs = [self._make_sig_dict("id1"), self._make_sig_dict("id2")]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(sigs, messages)

        self.assertEqual(len(result), 4)
        self.assertEqual(len(result[0].parts), 1)
        self.assertEqual(len(result[1].parts), 1)
        self.assertEqual(len(result[2].parts), 1)
        self.assertEqual(len(result[3].parts), 1)

    def test_text_boundary_prevents_merge(self):
        """A non-tool message boundary prevents merging across it."""
        messages = [
            self._make_tool_call("fn1", "id1", sig="sigA"),
            self._make_tool_response("fn1", "id1"),
            Content(role="model", parts=[Part(text="some response")]),
            self._make_tool_call("fn2", "id2"),
            self._make_tool_response("fn2", "id2"),
        ]
        sigs = [self._make_sig_dict("id1")]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(sigs, messages)

        self.assertEqual(len(result), 5)
        self.assertEqual(len(result[0].parts), 1)
        self.assertEqual(len(result[1].parts), 1)

    def test_three_parallel_calls_merged(self):
        """Three parallel calls are merged into a single model and user turn."""
        messages = [
            self._make_tool_call("fn1", "id1", sig="sigA"),
            self._make_tool_response("fn1", "id1"),
            self._make_tool_call("fn2", "id2"),
            self._make_tool_response("fn2", "id2"),
            self._make_tool_call("fn3", "id3"),
            self._make_tool_response("fn3", "id3"),
        ]
        sigs = [self._make_sig_dict("id1")]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(sigs, messages)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].parts), 3)
        self.assertEqual(len(result[1].parts), 3)

    def test_single_call_unchanged(self):
        """A single tool call with thought_signature passes through unchanged."""
        messages = [
            self._make_tool_call("fn1", "id1", sig="sigA"),
            self._make_tool_response("fn1", "id1"),
        ]
        sigs = [self._make_sig_dict("id1")]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(sigs, messages)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].parts), 1)
        self.assertEqual(len(result[1].parts), 1)

    def test_no_thinking_unchanged(self):
        """Without thought signatures, messages pass through unchanged."""
        messages = [
            self._make_tool_call("fn1", "id1"),
            self._make_tool_response("fn1", "id1"),
        ]
        result = self.adapter._merge_parallel_tool_calls_for_thinking([], messages)

        self.assertEqual(len(result), 2)

    def test_empty_messages(self):
        """Empty message list returns empty."""
        result = self.adapter._merge_parallel_tool_calls_for_thinking([], [])
        self.assertEqual(result, [])

    def test_batch_first_ordering_merged(self):
        """Batch-first ordering (calls together, then responses) is merged correctly."""
        messages = [
            self._make_tool_call("fn1", "id1", sig="sigA"),
            self._make_tool_call("fn2", "id2"),
            self._make_tool_response("fn1", "id1"),
            self._make_tool_response("fn2", "id2"),
        ]
        sigs = [self._make_sig_dict("id1")]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(sigs, messages)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0].parts), 2)
        self.assertEqual(len(result[1].parts), 2)

    def test_mixed_parallel_and_sequential_groups(self):
        """Mixed parallel and sequential groups are handled independently."""
        messages = [
            self._make_tool_call("fn1", "id1", sig="sigA"),
            self._make_tool_response("fn1", "id1"),
            self._make_tool_call("fn2", "id2"),
            self._make_tool_response("fn2", "id2"),
            self._make_tool_call("fn3", "id3", sig="sigB"),
            self._make_tool_response("fn3", "id3"),
        ]
        sigs = [self._make_sig_dict("id1"), self._make_sig_dict("id3")]
        result = self.adapter._merge_parallel_tool_calls_for_thinking(sigs, messages)

        self.assertEqual(len(result), 4)
        self.assertEqual(len(result[0].parts), 2)
        self.assertEqual(len(result[1].parts), 2)
        self.assertEqual(len(result[2].parts), 1)
        self.assertEqual(len(result[3].parts), 1)


if __name__ == "__main__":
    unittest.main()
