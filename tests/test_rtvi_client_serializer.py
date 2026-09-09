#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for :class:`pipecat.serializers.rtvi_client.RTVIClientSerializer`."""

import base64
import json
import unittest

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.serializers.rtvi_client import RTVIClientSerializer


def _server(msg_type: str, data: dict | None = None) -> str:
    return json.dumps({"label": RTVI.MESSAGE_LABEL, "type": msg_type, "data": data})


class TestRTVIClientDeserialize(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.s = RTVIClientSerializer()

    async def test_llm_lifecycle(self):
        self.assertIsInstance(
            await self.s.deserialize(_server("bot-llm-started")), LLMFullResponseStartFrame
        )
        frame = await self.s.deserialize(_server("bot-llm-text", {"text": "Paris"}))
        self.assertIsInstance(frame, LLMTextFrame)
        self.assertEqual(frame.text, "Paris")
        self.assertIsInstance(
            await self.s.deserialize(_server("bot-llm-stopped")), LLMFullResponseEndFrame
        )

    async def test_tts_text(self):
        frame = await self.s.deserialize(_server("bot-tts-text", {"text": "Hello there!"}))
        self.assertIsInstance(frame, TTSTextFrame)
        self.assertEqual(frame.text, "Hello there!")

    async def test_speaking_and_interruption(self):
        self.assertIsInstance(
            await self.s.deserialize(_server("bot-started-speaking")), BotStartedSpeakingFrame
        )
        self.assertIsInstance(
            await self.s.deserialize(_server("bot-stopped-speaking")), BotStoppedSpeakingFrame
        )
        self.assertIsInstance(
            await self.s.deserialize(_server("bot-interrupted")), InterruptionFrame
        )

    async def test_user_transcription_final_vs_interim(self):
        final = await self.s.deserialize(
            _server(
                "user-transcription",
                {"text": "hello", "user_id": "u", "timestamp": "t", "final": True},
            )
        )
        self.assertIsInstance(final, TranscriptionFrame)
        self.assertEqual(final.text, "hello")
        self.assertEqual(final.user_id, "u")

        interim = await self.s.deserialize(
            _server("user-transcription", {"text": "hel", "final": False})
        )
        self.assertIsInstance(interim, InterimTranscriptionFrame)
        self.assertEqual(interim.text, "hel")

    async def test_function_call(self):
        frame = await self.s.deserialize(
            _server(
                "llm-function-call-in-progress",
                {
                    "function_name": "get_weather",
                    "tool_call_id": "c1",
                    "arguments": {"city": "Paris"},
                },
            )
        )
        self.assertIsInstance(frame, FunctionCallInProgressFrame)
        self.assertEqual(frame.function_name, "get_weather")
        self.assertEqual(frame.arguments, {"city": "Paris"})

    async def test_unknown_and_non_rtvi_dropped(self):
        self.assertIsNone(await self.s.deserialize(_server("bot-ready", {"version": "2.0.0"})))
        self.assertIsNone(await self.s.deserialize(_server("metrics", {})))
        self.assertIsNone(
            await self.s.deserialize(json.dumps({"type": "bot-llm-text"}))
        )  # no label
        self.assertIsNone(await self.s.deserialize("not json"))


class TestRTVIClientSerialize(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.s = RTVIClientSerializer()

    async def test_output_audio_becomes_raw_audio(self):
        frame = OutputAudioRawFrame(audio=b"\x01\x02\x03\x04", sample_rate=16000, num_channels=1)
        out = json.loads(await self.s.serialize(frame))
        self.assertEqual(out["label"], RTVI.MESSAGE_LABEL)
        self.assertEqual(out["type"], "raw-audio")
        self.assertEqual(out["data"]["sampleRate"], 16000)
        self.assertEqual(out["data"]["numChannels"], 1)
        self.assertEqual(base64.b64decode(out["data"]["base64Audio"]), b"\x01\x02\x03\x04")

    async def test_client_message_passthrough(self):
        msg = {
            "label": RTVI.MESSAGE_LABEL,
            "type": "send-text",
            "id": "1",
            "data": {"content": "hi", "options": {"run_immediately": True}},
        }
        out = json.loads(await self.s.serialize(OutputTransportMessageFrame(message=msg)))
        self.assertEqual(out, msg)

    async def test_non_rtvi_transport_message_dropped(self):
        out = await self.s.serialize(OutputTransportMessageFrame(message={"foo": "bar"}))
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
