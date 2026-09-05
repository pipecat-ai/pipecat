#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import unittest
from unittest.mock import patch

from pipecat.utils.tracing.langfuse_helpers import (
    mark_trace_public,
    set_trace_public_resolver,
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


if __name__ == "__main__":
    unittest.main()
