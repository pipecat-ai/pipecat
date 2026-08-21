#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Regression tests for VoicemailDetector frame gating.

Once a VOICEMAIL verdict lands, the detector's gates close. An EndWorkerFrame
pushed from the documented on_voicemail_detected handler (or from anywhere
downstream of the detector) has to make it past those gates, or the handler
cannot end the call.
"""

import unittest
from typing import Any

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

    The verdict is injected as the exact frame sequence a streaming LLM emits
    (LLMFullResponseStartFrame / LLMTextFrame / LLMFullResponseEndFrame), so
    ClassificationProcessor runs its real code path and no network is needed.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class _MainPipelineProcessor(FrameProcessor):
    """Stands in for whatever sits downstream of the detector in a real app.

    The only property that matters is its position: downstream of the detector,
    so an upstream push has to cross the whole detector.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def _verdict_frames(verdict: str) -> list[Any]:
    return [
        LLMFullResponseStartFrame(),
        LLMTextFrame(verdict),
        LLMFullResponseEndFrame(),
        SleepFrame(1.0),  # outlasts voicemail_response_delay
    ]


def _detector() -> VoicemailDetector:
    return VoicemailDetector(
        llm=_PassthroughLLM(),  # type: ignore[arg-type]
        voicemail_response_delay=0.3,
    )


class TestVoicemailDetector(unittest.IsolatedAsyncioTestCase):
    async def test_docs_handler_end_frame_escapes_the_detector(self):
        """The documented on_voicemail_detected handler can end the call.

        The handler pushes EndWorkerFrame upstream from the classification
        processor, so it has to pass the closed ClassifierGate.
        """
        detector = _detector()

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(processor: FrameProcessor):
            await processor.push_frame(
                EndWorkerFrame(reason="Voicemail detected."),
                FrameDirection.UPSTREAM,
            )

        _down, up = await run_test(detector, frames_to_send=_verdict_frames("VOICEMAIL"))

        assert any(isinstance(frame, EndWorkerFrame) for frame in up), (
            "the EndWorkerFrame pushed by the documented handler never escaped the detector"
        )

    async def test_end_frame_from_main_pipeline_after_voicemail(self):
        """An upstream EndWorkerFrame from after the detector still gets out.

        Both gates close on a VOICEMAIL verdict, so this exercises the
        conversation branch gate as well as the classifier gate.
        """
        detector = _detector()
        ender = _MainPipelineProcessor()

        @detector.event_handler("on_voicemail_detected")
        async def _on_voicemail(_processor: FrameProcessor):
            await ender.push_frame(
                EndWorkerFrame(reason="Voicemail detected."),
                FrameDirection.UPSTREAM,
            )

        _down, up = await run_test(
            Pipeline([detector, ender]),
            frames_to_send=_verdict_frames("VOICEMAIL"),
        )

        assert any(isinstance(frame, EndWorkerFrame) for frame in up), (
            "the EndWorkerFrame never made it past the closed gates"
        )

    async def test_end_frame_from_main_pipeline_after_conversation(self):
        """A CONVERSATION verdict leaves the upstream path open, as before."""
        detector = _detector()
        ender = _MainPipelineProcessor()

        @detector.event_handler("on_conversation_detected")
        async def _on_conversation(_processor: FrameProcessor):
            await ender.push_frame(
                EndWorkerFrame(reason="Conversation detected."),
                FrameDirection.UPSTREAM,
            )

        _down, up = await run_test(
            Pipeline([detector, ender]),
            frames_to_send=_verdict_frames("CONVERSATION"),
        )

        assert any(isinstance(frame, EndWorkerFrame) for frame in up), (
            "the conversation branch did not let the frame out"
        )
