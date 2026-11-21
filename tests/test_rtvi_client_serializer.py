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
    InputTransportMessageFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
)
from pipecat.serializers.rtvi_client import RTVIClientSerializer


def _server(msg_type: str, data: dict | None = None) -> str:
    return json.dumps({"label": RTVI.MESSAGE_LABEL, "type": msg_type, "data": data})


class TestRTVIClientDeserialize(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.s = RTVIClientSerializer()

    async def test_llm_reports_stay_raw_messages(self):
        # Reports about the bot's own LLM: this pipeline's LLM frames would be
        # about its own.
        for msg_type, data in (
            ("bot-llm-started", None),
            ("bot-llm-text", {"text": "Paris"}),
            ("bot-llm-marker", {"text": "◐", "kind": "short", "raw": "◐ wait"}),
            ("bot-llm-stopped", None),
        ):
            frame = await self.s.deserialize(_server(msg_type, data))
            self.assertIsInstance(frame, InputTransportMessageFrame)
            self.assertEqual(frame.message["type"], msg_type)
            self.assertEqual(frame.message["data"], data)

    async def test_tts_text_stays_a_raw_message(self):
        frame = await self.s.deserialize(_server("bot-tts-text", {"text": "Hello there!"}))
        self.assertIsInstance(frame, InputTransportMessageFrame)
        self.assertEqual(frame.message["data"]["text"], "Hello there!")

    async def test_speaking_and_interruption_stay_raw_messages(self):
        # Reports about the bot's own output: this pipeline's speaking frames and
        # interruptions would be about its own.
        for msg_type in ("bot-started-speaking", "bot-stopped-speaking", "bot-interrupted"):
            frame = await self.s.deserialize(_server(msg_type))
            self.assertIsInstance(frame, InputTransportMessageFrame)
            self.assertEqual(frame.message["type"], msg_type)

    async def test_user_transcription_stays_a_raw_message(self):
        # The bot's transcription of this pipeline's speech: a TranscriptionFrame
        # would pass it off as this pipeline's own.
        for data in (
            {"text": "hello", "user_id": "u", "timestamp": "t", "final": True},
            {"text": "hel", "final": False},
        ):
            frame = await self.s.deserialize(_server("user-transcription", data))
            self.assertIsInstance(frame, InputTransportMessageFrame)
            self.assertEqual(frame.message["data"], data)

    async def test_function_calls_stay_raw_messages(self):
        # Reports about the bot's own function calls: this pipeline's function
        # call frames would be about its own.
        for msg_type, data in (
            (
                "llm-function-call-in-progress",
                {
                    "function_name": "get_weather",
                    "tool_call_id": "c1",
                    "arguments": {"city": "Paris"},
                },
            ),
            (
                "llm-function-call-stopped",
                {"function_name": "get_weather", "tool_call_id": "c1", "cancelled": True},
            ),
        ):
            frame = await self.s.deserialize(_server(msg_type, data))
            self.assertIsInstance(frame, InputTransportMessageFrame)
            self.assertEqual(frame.message["type"], msg_type)
            self.assertEqual(frame.message["data"], data)

    async def test_every_rtvi_message_stays_raw(self):
        for msg_type in ("bot-ready", "metrics", "server-message"):
            frame = await self.s.deserialize(_server(msg_type, {}))
            self.assertIsInstance(frame, InputTransportMessageFrame)
            self.assertEqual(frame.message["type"], msg_type)

    async def test_non_rtvi_dropped(self):
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
