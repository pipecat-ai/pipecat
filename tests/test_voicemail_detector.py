#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.extensions.voicemail.voicemail_detector import VoicemailDetector
from pipecat.frames.frames import (
    EndWorkerFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test


class _PassthroughLLM(FrameProcessor):
    """Stands in for the classifier LLM.

    The verdict is injected as the frame sequence a streaming LLM emits, so
    ClassificationProcessor runs its real path with no network.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _MainPipelineProcessor(FrameProcessor):
    """Stands in for a processor downstream of the detector."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def _verdict_frames(verdict: str) -> list[Frame]:
    return [
        LLMFullResponseStartFrame(),
        LLMTextFrame(text=verdict),
        LLMFullResponseEndFrame(),
        SleepFrame(0.4),
    ]


def _detector() -> VoicemailDetector:
    return VoicemailDetector(
        llm=_PassthroughLLM(),  # type: ignore[arg-type]
        voicemail_response_delay=0.1,
    )


class TestVoicemailDetectorEndWorkerFrame(unittest.IsolatedAsyncioTestCase):
    async def test_docs_snippet_pushing_upstream_from_processor(self):
        detector = _detector()

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor: FrameProcessor):
            await processor.push_frame(
                EndWorkerFrame(reason="Voicemail detected."), FrameDirection.UPSTREAM
            )

        _down, up = await run_test(
            detector, frames_to_send=_verdict_frames("VOICEMAIL"), start_timeout=5.0
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in up),
            f"EndWorkerFrame did not escape upstream: {[type(f).__name__ for f in up]}",
        )

    async def test_docs_snippet_pushing_downstream_from_processor(self):
        detector = _detector()

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor: FrameProcessor):
            await processor.push_frame(
                EndWorkerFrame(reason="Voicemail detected."), FrameDirection.DOWNSTREAM
            )

        down, _up = await run_test(
            detector, frames_to_send=_verdict_frames("VOICEMAIL"), start_timeout=5.0
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in down),
            f"EndWorkerFrame did not escape downstream: {[type(f).__name__ for f in down]}",
        )

    async def test_voicemail_verdict_lets_upstream_end_from_main_pipeline(self):
        detector = _detector()
        ender = _MainPipelineProcessor()

        @detector.event_handler("on_voicemail_detected")
        async def _on_verdict(_processor: FrameProcessor):
            await ender.push_frame(
                EndWorkerFrame(reason="VOICEMAIL detected."), FrameDirection.UPSTREAM
            )

        _down, up = await run_test(
            Pipeline([detector, ender]),
            frames_to_send=_verdict_frames("VOICEMAIL"),
            start_timeout=5.0,
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in up),
            f"EndWorkerFrame did not escape upstream after VOICEMAIL: "
            f"{[type(f).__name__ for f in up]}",
        )

    async def test_conversation_verdict_lets_upstream_end_from_main_pipeline(self):
        detector = _detector()
        ender = _MainPipelineProcessor()

        @detector.event_handler("on_conversation_detected")
        async def _on_verdict(_processor: FrameProcessor):
            await ender.push_frame(
                EndWorkerFrame(reason="CONVERSATION detected."), FrameDirection.UPSTREAM
            )

        _down, up = await run_test(
            Pipeline([detector, ender]),
            frames_to_send=_verdict_frames("CONVERSATION"),
            start_timeout=5.0,
        )
        self.assertTrue(
            any(isinstance(f, EndWorkerFrame) for f in up),
            f"EndWorkerFrame did not escape upstream after CONVERSATION: "
            f"{[type(f).__name__ for f in up]}",
        )
