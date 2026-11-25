#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the reading of how the user spoke in RTVIObserver's user-input messages."""

import unittest
from unittest.mock import AsyncMock

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.expressive.perception import PerceptionCapabilities, PerceptionSignal
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    PerceptionMetadataFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
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


class TestRTVIObserverPerceptionAfterTheSTT(unittest.IsolatedAsyncioTestCase):
    """A transcription waits for a perceiver after the STT, such as PipecatToneReader."""

    async def asyncSetUp(self):
        self.observer = RTVIObserver()
        self.observer.send_rtvi_message = AsyncMock()
        self.stt = FrameProcessor(name="stt")
        self.reader = FrameProcessor(name="reader")
        self.aggregator = FrameProcessor(name="aggregator")

    async def _push(self, frame: Frame, source, destination, *, first_push: bool):
        await self.observer.on_push_frame(
            FramePushed(
                source=source,
                destination=destination,
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
                first_push=first_push,
            )
        )

    def _user_inputs(self) -> list[RTVI.UserInputMessageData]:
        return [
            call.args[0].data
            for call in self.observer.send_rtvi_message.await_args_list
            if isinstance(call.args[0], RTVI.UserInputMessage)
        ]

    async def _announce_reader(self):
        metadata = PerceptionMetadataFrame(
            service_name="reader", perception=PerceptionCapabilities()
        )
        await self._push(metadata, self.reader, self.aggregator, first_push=True)
        # Passed along, it doesn't make the next processor a perceiver.
        await self._push(metadata, self.aggregator, self.stt, first_push=False)

    async def test_a_transcription_is_reported_once_read(self):
        await self._announce_reader()
        frame = TranscriptionFrame(text="Oh great, it's late.", user_id="", timestamp="now")

        await self._push(frame, self.stt, self.reader, first_push=True)
        self.assertEqual(self._user_inputs(), [])

        frame.perception = READING
        await self._push(frame, self.reader, self.aggregator, first_push=False)
        await self._push(frame, self.aggregator, self.stt, first_push=False)

        (data,) = self._user_inputs()
        self.assertEqual(data.text, "Oh great, it's late.")
        self.assertEqual([signal.name for signal in data.perception or []], ["tone", "speech_rate"])

    async def test_without_a_perceiver_a_transcription_is_reported_when_first_pushed(self):
        frame = TranscriptionFrame(text="Where's my order?", user_id="", timestamp="now")

        await self._push(frame, self.stt, self.reader, first_push=True)
        await self._push(frame, self.reader, self.aggregator, first_push=False)

        self.assertEqual(len(self._user_inputs()), 1)
