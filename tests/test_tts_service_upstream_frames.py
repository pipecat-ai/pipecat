#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time
import unittest
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    DataFrame,
    Frame,
    LLMFullResponseEndFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import SleepFrame, run_test


@dataclass
class ReturningFrame(DataFrame):
    """Stands in for the context frame the assistant aggregator pushes upstream.

    Deliberately not a TextFrame: the service would synthesize that rather than
    pass it through, which is not what this test is about.
    """

    label: str = ""


class StubTTSService(TTSService):
    """A TTS service that emits one audio frame per utterance."""

    def __init__(self, **kwargs):
        super().__init__(
            pause_frame_processing=True,
            sample_rate=16000,
            settings=TTSSettings(model=None, voice=None, language=None),
            **kwargs,
        )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield TTSStartedFrame()
        yield TTSAudioRawFrame(audio=b"\x00" * 640, sample_rate=16000, num_channels=1)
        yield TTSStoppedFrame()


class ArrivalRecorder(FrameProcessor):
    """Sits before the TTS and timestamps the returning frame's arrival."""

    def __init__(self, marks: dict, **kwargs):
        super().__init__(**kwargs)
        self._marks = marks

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, ReturningFrame):
            self._marks.setdefault("returned", time.monotonic())
        await self.push_frame(frame, direction)


class ToolResultSimulator(FrameProcessor):
    """Sits after the TTS, standing in for the assistant context aggregator.

    On `LLMFullResponseEndFrame` — the point at which the TTS has just taken its
    speaking pause — it pushes a frame upstream, the way the aggregator pushes
    the updated context upstream after a function call result. It also marks the
    moment the bot's audio ends, so the test can tell whether the frame got
    through before or after playback finished.
    """

    def __init__(self, marks: dict, **kwargs):
        super().__init__(**kwargs)
        self._marks = marks

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, LLMFullResponseEndFrame):
            await self.push_frame(ReturningFrame(label="tool result"), FrameDirection.UPSTREAM)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._marks.setdefault("bot_stopped", time.monotonic())


class TestTTSServiceUpstreamFrames(unittest.IsolatedAsyncioTestCase):
    async def test_upstream_frame_is_not_held_by_the_speaking_pause(self):
        """A frame travelling upstream must not wait for playback to finish.

        `TTSService` pauses its non-system frame queue while it is speaking, to
        keep audio from overlapping, and resumes on `BotStoppedSpeakingFrame`.
        The pause is direction-agnostic, so it also holds frames travelling
        upstream — including the context frame the assistant aggregator pushes
        upstream after a function call result, which is what asks the LLM to
        produce the next response. Holding it defers the whole next inference
        until the current utterance has played out.
        """
        marks = {}

        pipeline = Pipeline(
            [
                ArrivalRecorder(marks),
                StubTTSService(),
                ToolResultSimulator(marks),
            ]
        )

        frames_to_send = [
            # The LLM's response: some text to speak...
            TextFrame(text="Here are the available doctors."),
            # ...which the transport starts playing.
            BotStartedSpeakingFrame(),
            # End of the LLM response. The TTS pauses here, because it is
            # speaking, and will not resume until BotStoppedSpeakingFrame.
            # ToolResultSimulator pushes ReturningFrame upstream at this point.
            LLMFullResponseEndFrame(),
            # Playback continues for a while.
            SleepFrame(sleep=0.5),
            # Playback ends; the pause lifts.
            BotStoppedSpeakingFrame(),
            SleepFrame(sleep=0.3),
        ]

        await run_test(pipeline, frames_to_send=frames_to_send)

        self.assertIn("returned", marks, "the upstream frame never arrived at all")
        self.assertIn("bot_stopped", marks, "the bot never stopped speaking")

        held_for_ms = (marks["returned"] - marks["bot_stopped"]) * 1000
        self.assertLess(
            marks["returned"],
            marks["bot_stopped"],
            f"the upstream frame arrived {held_for_ms:.1f}ms AFTER playback ended: "
            "it was held by the speaking pause instead of crossing the TTS "
            "service while the bot was still speaking. On a real call the hold "
            "lasts as long as the utterance, delaying the next LLM inference by "
            "exactly that much.",
        )


if __name__ == "__main__":
    unittest.main()
