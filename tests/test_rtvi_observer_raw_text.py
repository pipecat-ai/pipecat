#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the raw text messages RTVIObserver sends for the STT, LLM and TTS."""

import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    AggregatedTextFrame,
    InterimTranscriptionFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver, RTVIObserverParams
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.text.base_text_aggregator import TextType

RAW_TEXT_MESSAGES = (RTVI.STTRawTextMessage, RTVI.LLMRawTextMessage, RTVI.TTSRawTextMessage)


class TestRTVIObserverRawText(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.observer = RTVIObserver(
            params=RTVIObserverParams(
                stt_raw_text_enabled=True,
                llm_raw_text_enabled=True,
                tts_raw_text_enabled=True,
            )
        )
        self.observer.send_rtvi_message = AsyncMock()
        # The bot is speaking, so text is sent right away instead of queued.
        self.observer._bot_is_speaking = True
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

    async def _push_one_of_each(self):
        await self._push(TranscriptionFrame(text="Hello.", user_id="user", timestamp="now"))
        await self._push(LLMTextFrame(text="Hi"))
        segment = AggregatedTextFrame("Hi there.", TextType.SENTENCE, context_id="turn")
        segment.will_be_spoken = True
        await self._push(segment)
        await self._push(
            TTSTextFrame("Hi", TextType.WORD, context_id="turn", segment_id=segment.id)
        )

    def _sent_messages(self):
        return [call.args[0] for call in self.observer.send_rtvi_message.await_args_list]

    def _sent_raw_text(self):
        return [m for m in self._sent_messages() if isinstance(m, RAW_TEXT_MESSAGES)]

    async def test_raw_text_is_kept_from_the_client_by_default(self):
        self.observer._params = RTVIObserverParams()
        await self._push_one_of_each()

        self.assertEqual(self._sent_raw_text(), [])

    async def test_a_transcription_is_sent_as_stt_raw_text_before_user_transcription(self):
        await self._push(InterimTranscriptionFrame(text="Hel", user_id="user", timestamp="t1"))
        await self._push(TranscriptionFrame(text="Hello.", user_id="user", timestamp="t2"))

        messages = self._sent_messages()
        self.assertEqual(
            [type(m) for m in messages],
            [
                RTVI.UserInputMessage,
                RTVI.STTRawTextMessage,
                RTVI.UserTranscriptionMessage,
            ]
            * 2,
        )
        self.assertEqual(
            [messages[1].data, messages[4].data],
            [
                RTVI.STTRawTextMessageData(text="Hel", user_id="user", timestamp="t1", final=False),
                RTVI.STTRawTextMessageData(
                    text="Hello.", user_id="user", timestamp="t2", final=True
                ),
            ],
        )

    async def test_llm_text_is_sent_as_llm_raw_text_before_bot_llm_text(self):
        await self._push(LLMTextFrame(text="Hi"))

        messages = self._sent_messages()
        self.assertEqual(
            [type(m) for m in messages], [RTVI.LLMRawTextMessage, RTVI.BotLLMTextMessage]
        )
        self.assertEqual(messages[0].data.text, "Hi")

    async def test_llm_text_is_only_sent_as_llm_raw_text_without_bot_llm_messages(self):
        self.observer._params.bot_llm_enabled = False
        await self._push(LLMTextFrame(text="Hi there."))

        self.assertEqual([type(m) for m in self._sent_messages()], [RTVI.LLMRawTextMessage])

    async def test_spoken_text_is_sent_as_tts_raw_text_before_bot_tts_text(self):
        await self._push_one_of_each()

        tts_messages = [
            m
            for m in self._sent_messages()
            if isinstance(m, (RTVI.TTSRawTextMessage, RTVI.BotTTSTextMessage))
        ]
        self.assertEqual(
            [type(m) for m in tts_messages], [RTVI.TTSRawTextMessage, RTVI.BotTTSTextMessage]
        )
        self.assertEqual(tts_messages[0].data.text, "Hi")

    async def test_spoken_text_is_only_sent_as_tts_raw_text_without_bot_output_or_tts(self):
        self.observer._params.bot_output_enabled = False
        self.observer._params.bot_tts_enabled = False
        segment = AggregatedTextFrame("Hi there.", TextType.SENTENCE, context_id="turn")
        segment.will_be_spoken = True
        await self._push(segment)
        await self._push(
            TTSTextFrame("Hi", TextType.WORD, context_id="turn", segment_id=segment.id)
        )

        self.assertEqual([type(m) for m in self._sent_messages()], [RTVI.TTSRawTextMessage])

    async def test_tts_raw_text_leaves_out_skipped_text_types(self):
        self.observer._params.skip_text_types = ["status"]
        segment = AggregatedTextFrame("Looking that up.", "status", context_id="turn")
        segment.will_be_spoken = True
        await self._push(segment)
        await self._push(
            TTSTextFrame("Looking", TextType.WORD, context_id="turn", segment_id=segment.id)
        )

        self.assertEqual(self._sent_raw_text(), [])

    async def test_raw_text_is_kept_from_older_clients(self):
        for version in ([2, 1, 0], [1, 4, 0]):
            with self.subTest(version=version):
                self.observer._rtvi = SimpleNamespace(client_version=version)
                await self._push_one_of_each()

                self.assertEqual(self._sent_raw_text(), [])


class TestRTVIObserverParamsUserTranscription(unittest.TestCase):
    def test_turning_user_transcription_off_is_deprecated(self):
        with self.assertWarnsRegex(
            DeprecationWarning, "`RTVIObserverParams.user_transcription_enabled` is deprecated"
        ) as caught:
            RTVIObserverParams(user_transcription_enabled=False)

        self.assertEqual(caught.filename, __file__)

    def test_leaving_user_transcription_on_is_not_deprecated(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            RTVIObserverParams()
            RTVIObserverParams(user_transcription_enabled=True)


if __name__ == "__main__":
    unittest.main()
