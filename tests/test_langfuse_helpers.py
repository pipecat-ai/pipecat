#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import os
import unittest
from unittest.mock import patch

from pipecat.utils.tracing.langfuse_helpers import (
    build_llm_output_payload,
    mark_trace_public,
    set_trace_public_resolver,
    standardize_tools_to_chatml,
)


class _RecordingSpan:
    """Span stand-in that records the attributes set on it."""

    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


class TestMarkTracePublic(unittest.TestCase):
    """Tests for the LANGFUSE_TRACES_PUBLIC gate on mark_trace_public()."""

    def _mark(self, env):
        span = _RecordingSpan()
        # clear=True so an inherited flag can't leak into the test.
        with patch.dict(os.environ, env, clear=True):
            mark_trace_public(span)
        return span

    def test_private_by_default(self):
        """Traces stay private when the flag is unset."""
        span = self._mark({})
        self.assertNotIn("langfuse.trace.public", span.attributes)

    def test_private_when_flag_disabled(self):
        """An explicit falsy value keeps the trace private."""
        span = self._mark({"LANGFUSE_TRACES_PUBLIC": "false"})
        self.assertNotIn("langfuse.trace.public", span.attributes)

    def test_public_when_flag_enabled(self):
        """The trace is marked public only when the flag opts in."""
        for value in ("1", "true", "TRUE", "yes"):
            with self.subTest(value=value):
                span = self._mark({"LANGFUSE_TRACES_PUBLIC": value})
                self.assertIs(span.attributes.get("langfuse.trace.public"), True)


class TestTracePublicResolver(unittest.TestCase):
    """Tests for the app-installed resolver that overrides the env flag."""

    def setUp(self):
        self.addCleanup(set_trace_public_resolver, None)

    def _mark(self, env):
        span = _RecordingSpan()
        with patch.dict(os.environ, env, clear=True):
            mark_trace_public(span)
        return span

    def test_resolver_overrides_enabled_env_flag(self):
        """A resolver saying no keeps the trace private despite the env flag."""
        set_trace_public_resolver(lambda: False)
        span = self._mark({"LANGFUSE_TRACES_PUBLIC": "true"})
        self.assertNotIn("langfuse.trace.public", span.attributes)

    def test_resolver_overrides_absent_env_flag(self):
        """A resolver saying yes marks the trace public without the env flag."""
        set_trace_public_resolver(lambda: True)
        span = self._mark({})
        self.assertIs(span.attributes.get("langfuse.trace.public"), True)

    def test_failing_resolver_keeps_trace_private(self):
        """Visibility fails closed when the resolver raises."""

        def boom():
            raise RuntimeError("no org context")

        set_trace_public_resolver(boom)
        span = self._mark({"LANGFUSE_TRACES_PUBLIC": "true"})
        self.assertNotIn("langfuse.trace.public", span.attributes)

    def test_clearing_resolver_restores_env_flag(self):
        """Passing None hands the decision back to the env var."""
        set_trace_public_resolver(lambda: False)
        set_trace_public_resolver(None)
        span = self._mark({"LANGFUSE_TRACES_PUBLIC": "true"})
        self.assertIs(span.attributes.get("langfuse.trace.public"), True)


class TestToolPayloads(unittest.TestCase):
    def test_native_and_json_schema_parameters_survive_normalization(self):
        parameters = {
            "type": "object",
            "properties": {"mode": {"const": "report"}},
            "required": ["mode"],
            "additionalProperties": False,
        }
        for field in ("parameters", "parameters_json_schema"):
            with self.subTest(field=field):
                tools = [
                    {
                        "function_declarations": [
                            {"name": "fetch_report", "description": "Fetch", field: parameters}
                        ]
                    }
                ]
                formatted = standardize_tools_to_chatml(tools)
                self.assertEqual(formatted[0]["function"]["parameters"], parameters)
                self.assertIn(field, tools[0]["function_declarations"][0])

    def test_parallel_calls_keep_text_and_json_arguments(self):
        output = build_llm_output_payload(
            "Fetching both reports.",
            [
                {
                    "tool_call_id": "call-1",
                    "function_name": "fetch_report",
                    "arguments": {"name": "Ada"},
                },
                {
                    "tool_call_id": "call-2",
                    "function_name": "fetch_report",
                    "arguments": '{"name": "Grace"}',
                },
            ],
        )
        (message,) = json.loads(output)
        self.assertEqual(message["role"], "assistant")
        self.assertEqual(message["content"], "Fetching both reports.")
        self.assertEqual([call["id"] for call in message["tool_calls"]], ["call-1", "call-2"])
        self.assertEqual(
            [json.loads(call["function"]["arguments"]) for call in message["tool_calls"]],
            [{"name": "Ada"}, {"name": "Grace"}],
        )


if __name__ == "__main__":
    unittest.main()
