#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the reading of how the user spoke in RTVIObserver's user-input messages."""

import unittest
from unittest.mock import AsyncMock

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.expressive.perception import PerceptionSignal
from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

READING = [
    PerceptionSignal("tone", labels=[("frustrated", 0.86), ("neutral", 0.10)]),
    PerceptionSignal("speech_rate", value=4.1),
]


class TestRTVIObserverPerception(unittest.IsolatedAsyncioTestCase):
    async def _user_input_for(self, frame: Frame) -> RTVI.UserInputMessageData:
        observer = RTVIObserver()
        observer.send_rtvi_message = AsyncMock()
        transport = BaseOutputTransport(TransportParams())
        await observer.on_push_frame(
            FramePushed(
                source=transport,
                destination=transport,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
        sent = [
            call.args[0]
            for call in observer.send_rtvi_message.await_args_list
            if isinstance(call.args[0], RTVI.UserInputMessage)
        ]
        self.assertEqual(len(sent), 1)
        return sent[0].data

    async def test_a_reading_travels_with_its_transcription(self):
        data = await self._user_input_for(
            InterimTranscriptionFrame(
                text="Can you tell me", user_id="", timestamp="now", perception=READING
            )
        )

        self.assertFalse(data.final)
        self.assertEqual(
            data.model_dump(include={"perception"}),
            {
                "perception": [
                    {
                        "name": "tone",
                        "labels": [
                            {"label": "frustrated", "confidence": 0.86},
                            {"label": "neutral", "confidence": 0.10},
                        ],
                        "value": None,
                    },
                    {"name": "speech_rate", "labels": [], "value": 4.1},
                ]
            },
        )

    async def test_a_transcription_without_a_reading_has_no_perception(self):
        data = await self._user_input_for(
            TranscriptionFrame(text="Where's my order?", user_id="", timestamp="now")
        )

        self.assertIsNone(data.perception)
