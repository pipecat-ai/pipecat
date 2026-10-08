#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for which pushes RTVIObserver handles."""

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    AggregatedTextFrame,
    AggregatedTextProgressFrame,
    InputAudioRawFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIConfigureObserverFrame
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.text.base_text_aggregator import AggregationType


class TestRTVIObserverPushes(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.observer = RTVIObserver(params=RTVIObserverParams())
        self.observer.send_rtvi_message = AsyncMock()
        self.source = FrameProcessor()

    async def _push(self, frame, *, first_push=True, source=None):
        source = source or self.source
        await self.observer.on_push_frame(
            FramePushed(
                source=source,
                destination=source,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
                first_push=first_push,
            )
        )

    async def test_audio_is_skipped_unless_audio_levels_are_reported(self):
        for frame_type in (InputAudioRawFrame, TTSAudioRawFrame):
            await self._push(frame_type(audio=b"\0" * 320, sample_rate=16000, num_channels=1))
        self.observer.send_rtvi_message.assert_not_awaited()

        observer = RTVIObserver(
            params=RTVIObserverParams(user_audio_level_enabled=True, audio_level_period_secs=0)
        )
        observer.send_rtvi_message = AsyncMock()
        self.observer = observer
        await self._push(InputAudioRawFrame(audio=b"\0" * 320, sample_rate=16000, num_channels=1))
        observer.send_rtvi_message.assert_awaited_once()

    async def test_a_frame_is_handled_on_its_first_push_only(self):
        frame = UserStartedSpeakingFrame()

        await self._push(frame)
        await self._push(frame, first_push=False)

        self.observer.send_rtvi_message.assert_awaited_once()

    async def test_a_frame_disabled_on_its_first_push_is_never_handled(self):
        frame = VADUserStartedSpeakingFrame()

        await self._push(frame)
        self.observer._apply_config(RTVIConfigureObserverFrame(vad_user_speaking_enabled=True))
        await self._push(frame, first_push=False)

        self.observer.send_rtvi_message.assert_not_awaited()

    async def test_aggregated_text_is_handled_once_it_has_gone_through_the_transport(self):
        frame = AggregatedTextFrame(text="hello", text_type="sentence")
        transport = BaseOutputTransport(TransportParams())

        await self._push(frame)
        self.assertEqual(self.observer._queued_aggregated_text_frames, [])

        await self._push(frame, first_push=False, source=transport)
        self.assertEqual(self.observer._queued_aggregated_text_frames, [frame])


class TestRTVIObserverSkippedTypes(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.observer = RTVIObserver(params=RTVIObserverParams(skip_text_types=["status"]))
        self.observer.send_rtvi_message = AsyncMock()
        # The bot is speaking, so text is sent right away instead of queued.
        self.observer._bot_is_speaking = True
        self.transport = BaseOutputTransport(TransportParams())

    async def _push(self, frame, context_id):
        frame.context_id = context_id
        await self.observer.on_push_frame(
            FramePushed(
                source=self.transport,
                destination=self.transport,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
                first_push=True,
            )
        )

    def _sent_texts(self):
        return [call.args[0].data.text for call in self.observer.send_rtvi_message.await_args_list]

    async def test_a_skipped_segment_keeps_its_progress_and_words_from_the_client(self):
        segment = AggregatedTextFrame("One moment, please.", "status")
        await self._push(segment, "status")
        await self._push(TTSTextFrame("One", text_type=AggregationType.WORD), "status")
        progress = AggregatedTextProgressFrame(
            segment_id=segment.id,
            context_id="status",
            text="One moment, please.",
            text_type="status",
            accumulated_text="One",
            remaining_text=" moment, please.",
        )
        await self._push(progress, "status")

        self.observer.send_rtvi_message.assert_not_awaited()

    async def test_the_next_segment_is_sent_as_usual(self):
        await self._push(AggregatedTextFrame("One moment, please.", "status"), "status")
        await self._push(AggregatedTextFrame("That sounds fun.", AggregationType.SENTENCE), "llm")
        await self._push(TTSTextFrame("That", text_type=AggregationType.WORD), "llm")

        self.assertIn("That sounds fun.", self._sent_texts())
        self.assertIn("That", self._sent_texts())

    async def test_a_segment_after_a_skipped_one_in_the_same_context_is_sent(self):
        # All the segments of one LLM response share a TTS context.
        await self._push(AggregatedTextFrame("One moment, please.", "status"), "llm")
        await self._push(TTSTextFrame("One", text_type=AggregationType.WORD), "llm")
        await self._push(AggregatedTextFrame("That sounds fun.", AggregationType.SENTENCE), "llm")
        await self._push(TTSTextFrame("That", text_type=AggregationType.WORD), "llm")

        self.assertEqual(self._sent_texts(), ["That sounds fun.", "That"])


if __name__ == "__main__":
    unittest.main()
