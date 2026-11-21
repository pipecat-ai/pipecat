#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the client version RTVIObserver serves before and after client-ready."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, PropertyMock, patch

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import LLMTextFrame, OutputTransportMessageUrgentFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.pipeline.capabilities import BotCapabilities
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.observer import RTVIObserverParams
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor


class TestRTVIObserverClientVersion(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        worker = SimpleNamespace(
            capabilities=BotCapabilities(audio_in=True, audio_out=True),
            flush_pipeline=AsyncMock(),
        )
        worker_patch = patch.object(
            RTVIProcessor, "pipeline_worker", new_callable=PropertyMock, return_value=worker
        )
        worker_patch.start()
        self.addCleanup(worker_patch.stop)

        self.processor = RTVIProcessor()
        # llm-raw-text is sent only to clients of 2.2.0 and later.
        self.observer = self.processor.create_rtvi_observer(
            params=RTVIObserverParams(llm_raw_text_enabled=True)
        )
        self.sent: list[str] = []

        async def push_frame(frame, direction=FrameDirection.DOWNSTREAM):
            if isinstance(frame, OutputTransportMessageUrgentFrame):
                self.sent.append(frame.message["type"])

        self.processor.push_frame = push_frame
        self.source = FrameProcessor()

    async def asyncTearDown(self):
        await self.observer.cleanup()
        await self.processor.cleanup()

    async def _push_llm_text(self):
        await self.observer.on_push_frame(
            FramePushed(
                source=self.source,
                destination=self.source,
                frame=LLMTextFrame(text="Hello"),
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
                first_push=True,
            )
        )

    async def test_before_client_ready_the_current_protocol_is_served(self):
        await self._push_llm_text()

        self.assertIn("llm-raw-text", self.sent)

    async def test_after_client_ready_the_client_version_is_served(self):
        await self.processor._handle_message(
            RTVI.Message(type="client-ready", id="ready", data={"version": "2.1.0"})
        )
        await self._push_llm_text()

        self.assertNotIn("llm-raw-text", self.sent)


if __name__ == "__main__":
    unittest.main()
