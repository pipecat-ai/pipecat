#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the bot output RTVIObserver sends for a speech-to-speech transcript."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.text.base_text_aggregator import TextType

DELTAS = ["Sure", ",", " let", "'s", " talk", " about", " Kal", "di", ".", " It", " began", " in"]


def _delta(text):
    # What a speech-to-speech service pushes for each piece of its transcript.
    frame = TTSTextFrame(text, text_type=TextType.SENTENCE)
    frame.includes_inter_frame_spaces = True
    return frame


class TestRTVIObserverTranscript(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.observer = RTVIObserver(
            params=RTVIObserverParams(bot_tts_enabled=True, tts_raw_text_enabled=True)
        )
        self.observer.send_rtvi_message = AsyncMock()
        self.transport = BaseOutputTransport(TransportParams())

    async def _push(self, frame):
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

    async def _push_deltas(self, deltas=DELTAS):
        await self._push(BotStartedSpeakingFrame())
        for text in deltas:
            await self._push(_delta(text))

    def _outputs(self):
        return [
            call.args[0].data
            for call in self.observer.send_rtvi_message.await_args_list
            if isinstance(call.args[0], RTVI.BotOutputMessage)
        ]

    def _output_texts(self):
        return [data.text for data in self._outputs()]

    async def test_deltas_are_sent_as_whole_sentences(self):
        await self._push_deltas()

        self.assertEqual(self._output_texts(), ["Sure, let's talk about Kaldi."])

    async def test_the_sentence_that_never_ends_is_sent_when_the_bot_stops(self):
        await self._push_deltas()
        await self._push(BotStoppedSpeakingFrame())

        self.assertEqual(self._output_texts(), ["Sure, let's talk about Kaldi.", "It began in"])

    async def test_the_sentence_cut_off_is_sent_on_an_interruption(self):
        await self._push_deltas()
        await self._push(InterruptionFrame())

        self.assertEqual(self._output_texts(), ["Sure, let's talk about Kaldi.", "It began in"])

    async def test_each_sentence_is_a_segment_of_its_own(self):
        await self._push_deltas()
        await self._push(BotStoppedSpeakingFrame())

        first, second = self._outputs()
        self.assertNotEqual(first.segment_id, second.segment_id)
        self.assertEqual(first.output_type, TextType.SENTENCE)

    async def test_text_waits_for_the_bot_to_speak(self):
        for text in DELTAS:
            await self._push(_delta(text))
        self.assertEqual(self._outputs(), [])

        await self._push(BotStartedSpeakingFrame())

        self.assertEqual(self._output_texts(), ["Sure, let's talk about Kaldi."])

    async def test_the_tts_text_messages_still_carry_each_delta(self):
        await self._push_deltas(["Sure", ",", " ok."])

        tts_texts = [
            call.args[0].data.text
            for call in self.observer.send_rtvi_message.await_args_list
            if isinstance(call.args[0], RTVI.BotTTSTextMessage)
        ]
        raw_texts = [
            call.args[0].data.text
            for call in self.observer.send_rtvi_message.await_args_list
            if isinstance(call.args[0], RTVI.TTSRawTextMessage)
        ]
        self.assertEqual(tts_texts, ["Sure", ",", " ok."])
        self.assertEqual(raw_texts, ["Sure", ",", " ok."])

    async def test_a_transcript_of_a_skipped_text_type_is_not_sent(self):
        self.observer._params.skip_text_types = [TextType.SENTENCE]
        await self._push_deltas()
        await self._push(BotStoppedSpeakingFrame())

        self.assertEqual(self._outputs(), [])

    async def test_a_sentence_spoken_by_a_tts_is_sent_as_it_is(self):
        await self._push(BotStartedSpeakingFrame())
        sentence = TTSTextFrame("Hi there.", TextType.SENTENCE, segment_id=7)
        sentence.includes_inter_frame_spaces = True
        await self._push(sentence)

        self.assertEqual(self._output_texts(), ["Hi there."])
        self.assertEqual(self._outputs()[0].segment_id, 7)

    async def test_a_legacy_client_gets_each_delta(self):
        self.observer._rtvi = SimpleNamespace(client_version=[1, 4, 0])
        await self._push_deltas(["Sure", ",", " ok."])

        self.assertEqual(self._output_texts(), ["Sure", ",", " ok."])


if __name__ == "__main__":
    unittest.main()
