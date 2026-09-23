#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for RTVI observer frame-id retention."""

import unittest
from unittest.mock import AsyncMock

from pipecat.frames.frames import (
    AggregatedTextFrame,
    FunctionCallFromLLM,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    LLMMarkerResponseFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIConfigureObserverFrame
from pipecat.processors.frameworks.rtvi.observer import (
    RTVIFunctionCallReportLevel,
    RTVIObserver,
    RTVIObserverParams,
)
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams


class TestRTVIObserverFrameRetention(unittest.IsolatedAsyncioTestCase):
    async def _push(self, observer: RTVIObserver, frame, source: FrameProcessor):
        await observer.on_push_frame(
            FramePushed(
                source=source,
                destination=source,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )

    async def test_unhandled_audio_frames_do_not_accumulate_ids(self):
        observer = RTVIObserver(params=RTVIObserverParams())
        observer.send_rtvi_message = AsyncMock()
        source = FrameProcessor()

        for frame_type in (InputAudioRawFrame, TTSAudioRawFrame):
            for _ in range(128):
                await self._push(
                    observer,
                    frame_type(audio=b"\0" * 320, sample_rate=16000, num_channels=1),
                    source,
                )

        self.assertEqual(observer._frames_seen, set())
        observer.send_rtvi_message.assert_not_awaited()

    async def test_handled_frames_still_deduplicate(self):
        observer = RTVIObserver(params=RTVIObserverParams())
        observer.send_rtvi_message = AsyncMock()
        source = FrameProcessor()
        frame = UserStartedSpeakingFrame()

        await self._push(observer, frame, source)
        await self._push(observer, frame, source)

        observer.send_rtvi_message.assert_awaited_once()
        self.assertEqual(observer._frames_seen, {frame.id})

    async def test_failed_handling_can_retry_the_same_frame(self):
        observer = RTVIObserver(params=RTVIObserverParams())
        observer.send_rtvi_message = AsyncMock(side_effect=[RuntimeError("send failed"), None])
        source = FrameProcessor()
        frame = UserStartedSpeakingFrame()

        with self.assertRaisesRegex(RuntimeError, "send failed"):
            await self._push(observer, frame, source)
        self.assertNotIn(frame.id, observer._frames_seen)

        await self._push(observer, frame, source)

        self.assertEqual(observer._frames_seen, {frame.id})
        self.assertEqual(observer.send_rtvi_message.await_count, 2)

    async def test_enabled_audio_levels_still_deduplicate_audio_frames(self):
        observer = RTVIObserver(
            params=RTVIObserverParams(user_audio_level_enabled=True, audio_level_period_secs=60)
        )
        observer.send_rtvi_message = AsyncMock()
        source = FrameProcessor()
        frame = InputAudioRawFrame(audio=b"\0" * 320, sample_rate=16000, num_channels=1)

        await self._push(observer, frame, source)
        await self._push(observer, frame, source)

        self.assertEqual(observer._frames_seen, {frame.id})

    async def test_aggregated_frames_still_retry_after_non_output_sighting(self):
        observer = RTVIObserver(params=RTVIObserverParams())
        observer.send_rtvi_message = AsyncMock()
        source = FrameProcessor()
        output_transport = BaseOutputTransport(TransportParams())
        frame = AggregatedTextFrame(text="hello", aggregated_by="sentence")

        await self._push(observer, frame, source)
        self.assertNotIn(frame.id, observer._frames_seen)

        await self._push(observer, frame, output_transport)
        await self._push(observer, frame, output_transport)

        self.assertEqual(observer._frames_seen, {frame.id})
        self.assertEqual(observer._queued_aggregated_text_frames, [frame])
        observer.send_rtvi_message.assert_not_awaited()

    async def test_disabled_marker_frames_still_deduplicate_after_runtime_enable(self):
        observer = RTVIObserver(params=RTVIObserverParams())
        observer.send_rtvi_message = AsyncMock()
        source = FrameProcessor()
        frame = LLMMarkerResponseFrame(raw="hello")

        await self._push(observer, frame, source)
        observer._apply_config(RTVIConfigureObserverFrame(bot_llm_marker_enabled=True))
        await self._push(observer, frame, source)

        observer.send_rtvi_message.assert_not_awaited()
        self.assertNotIn(frame.id, observer._frames_seen)

    async def test_disabled_vad_frames_still_deduplicate_after_runtime_enable(self):
        observer = RTVIObserver(params=RTVIObserverParams())
        observer.send_rtvi_message = AsyncMock()
        source = FrameProcessor()
        frame = VADUserStartedSpeakingFrame()

        await self._push(observer, frame, source)
        observer._apply_config(RTVIConfigureObserverFrame(vad_user_speaking_enabled=True))
        await self._push(observer, frame, source)

        observer.send_rtvi_message.assert_not_awaited()
        self.assertNotIn(frame.id, observer._frames_seen)

    async def test_disabled_function_calls_still_deduplicate_after_runtime_enable(self):
        observer = RTVIObserver(
            params=RTVIObserverParams(
                function_call_report_level={"*": RTVIFunctionCallReportLevel.DISABLED}
            )
        )
        observer.send_rtvi_message = AsyncMock()
        source = FrameProcessor()
        frame = FunctionCallsStartedFrame(
            function_calls=[
                FunctionCallFromLLM(
                    function_name="get_weather",
                    tool_call_id="call-1",
                    arguments={},
                    context=None,
                )
            ]
        )

        await self._push(observer, frame, source)
        observer._apply_config(
            RTVIConfigureObserverFrame(
                function_call_report_level={"*": RTVIFunctionCallReportLevel.FULL}
            )
        )
        await self._push(observer, frame, source)

        observer.send_rtvi_message.assert_not_awaited()
        self.assertNotIn(frame.id, observer._frames_seen)


if __name__ == "__main__":
    unittest.main()
