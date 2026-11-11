#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the diarization speaker in RTVIObserver's transcription messages."""

import unittest
from unittest.mock import AsyncMock

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TranscriptionSpeaker,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams

SPEAKER = TranscriptionSpeaker(label="B", number=2)


class TestRTVIObserverSpeaker(unittest.IsolatedAsyncioTestCase):
    async def _messages_for(self, frame: Frame) -> list:
        observer = RTVIObserver(params=RTVIObserverParams(stt_raw_text_enabled=True))
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
        return [
            call.args[0]
            for call in observer.send_rtvi_message.await_args_list
            if isinstance(call.args[0], (RTVI.UserInputMessage, RTVI.STTRawTextMessage))
        ]

    async def test_a_diarized_transcription_carries_its_speaker(self):
        messages = await self._messages_for(
            TranscriptionFrame(
                text="Mine is the banana.", user_id="", timestamp="now", speaker=SPEAKER
            )
        )

        self.assertEqual(
            [m.data.model_dump(include={"speaker"}) for m in messages],
            [{"speaker": {"label": "B", "number": 2}}] * 2,
        )

    async def test_an_interim_transcription_carries_its_speaker(self):
        messages = await self._messages_for(
            InterimTranscriptionFrame(text="Mine is", user_id="", timestamp="now", speaker=SPEAKER)
        )

        self.assertEqual(
            [(m.data.final, m.data.speaker) for m in messages],
            [(False, SPEAKER)] * 2,
        )

    async def test_a_transcription_without_a_speaker_sends_none(self):
        messages = await self._messages_for(
            TranscriptionFrame(text="Hello.", user_id="", timestamp="now")
        )

        self.assertEqual(len(messages), 2)
        for message in messages:
            self.assertIsNone(message.data.speaker)
            self.assertNotIn("speaker", message.model_dump(exclude_none=True)["data"])
