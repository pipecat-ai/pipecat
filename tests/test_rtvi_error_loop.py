#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression test for the RTVI / output transport ErrorFrame feedback loop.

When an output transport fails to send a message it calls ``push_error``, which
pushes an ``ErrorFrame`` upstream. If ``RTVIProcessor`` is upstream it turns that
ErrorFrame into an ``RTVI.Error`` message and pushes it back down to the same
transport, which fails again. That is a self-amplifying loop: one failed send
becomes an unbounded stream of ErrorFrames at event loop speed.

See support ticket T-2660: four production storms, roughly 1.4 million ErrorFrame
allocations each, about 2,400 log lines per second for 10 minutes.

The fake transport here caps its failures so a regression fails fast instead of
hanging or exhausting memory.
"""

import asyncio
import unittest

from pipecat.frames.frames import ErrorFrame, OutputTransportMessageUrgentFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

# The loop is unbounded on a broken build. Stop failing after this many sends so
# the test fails fast instead of hanging.
MAX_FAILURES = 10

# Guard against a build where the loop somehow keeps running past MAX_FAILURES.
TEST_TIMEOUT_SECS = 20


class _FailingOutputTransport(BaseOutputTransport):
    """An output transport whose sends fail the way Daily's do when signaling dies.

    Mirrors ``DailyOutputTransport.send_message``: the client returns an error and
    the transport reports it with ``push_error``, which travels upstream.
    """

    def __init__(self, max_failures: int = MAX_FAILURES):
        super().__init__(TransportParams())
        self.send_message_count = 0
        self._max_failures = max_failures

    async def send_message(self, frame):
        self.send_message_count += 1
        if self.send_message_count <= self._max_failures:
            await self.push_error("Unable to send message: fake signaling failure")


class TestRTVIErrorFrameSendLoop(unittest.IsolatedAsyncioTestCase):
    async def test_failed_transport_send_does_not_loop(self):
        """One failed send produces one ErrorFrame, not a storm of them."""
        transport = _FailingOutputTransport()
        rtvi = RTVIProcessor()
        pipeline = Pipeline([rtvi, transport])

        # A single outbound message that the transport will fail to send. On a
        # broken build the resulting ErrorFrame comes back down as an RTVI error
        # message and the transport is asked to send again, and again.
        kickoff = OutputTransportMessageUrgentFrame(message={"label": "test", "type": "kickoff"})

        # The SleepFrame holds the EndFrame back so a looping build reaches its
        # failure cap instead of being cut short by the pipeline shutting down.
        _, up_frames = await asyncio.wait_for(
            run_test(
                pipeline,
                enable_rtvi=True,
                frames_to_send=[kickoff, SleepFrame(sleep=0.5)],
            ),
            timeout=TEST_TIMEOUT_SECS,
        )

        error_frames = [frame for frame in up_frames if isinstance(frame, ErrorFrame)]

        self.assertEqual(
            transport.send_message_count,
            1,
            f"send_message was called {transport.send_message_count} times for one "
            "outbound message: the failed send is being re-broadcast over the same "
            "dead connection",
        )
        self.assertEqual(
            len(error_frames),
            1,
            f"{len(error_frames)} ErrorFrames reached the app for one failed send",
        )
