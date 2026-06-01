#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for :class:`pipecat.evals.serializer.RTVIEvalSerializer`."""

import json
import unittest

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.evals.serializer import EVAL_RESET_MESSAGE_TYPE, RTVIEvalSerializer
from pipecat.frames.frames import (
    InputTransportMessageFrame,
    LLMMessagesUpdateFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
)


class TestRTVIEvalSerializerDeserialize(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.serializer = RTVIEvalSerializer()

    async def test_send_text_wraps_as_transport_message(self):
        msg = {
            "label": RTVI.MESSAGE_LABEL,
            "type": "send-text",
            "id": "1",
            "data": {"content": "hi"},
        }
        frame = await self.serializer.deserialize(json.dumps(msg))
        self.assertIsInstance(frame, InputTransportMessageFrame)
        self.assertEqual(frame.message, msg)

    async def test_raw_audio_wraps_as_transport_message(self):
        # raw-audio is forwarded to the RTVIProcessor, which decodes it and
        # pushes an InputAudioRawFrame downstream into the input transport.
        msg = {
            "label": RTVI.MESSAGE_LABEL,
            "type": "raw-audio",
            "id": "2",
            "data": {"base64Audio": "AAAA", "sampleRate": 16000, "numChannels": 1},
        }
        frame = await self.serializer.deserialize(json.dumps(msg))
        self.assertIsInstance(frame, InputTransportMessageFrame)
        self.assertEqual(frame.message["type"], "raw-audio")

    async def test_other_rtvi_message_wraps_as_transport_message(self):
        msg = {"label": RTVI.MESSAGE_LABEL, "type": "client-ready", "id": "3", "data": {}}
        frame = await self.serializer.deserialize(json.dumps(msg))
        self.assertIsInstance(frame, InputTransportMessageFrame)

    async def test_eval_reset_short_circuits_to_messages_update(self):
        msg = {
            "label": RTVI.MESSAGE_LABEL,
            "type": "client-message",
            "id": "4",
            "data": {
                "t": EVAL_RESET_MESSAGE_TYPE,
                "d": {"messages": [{"role": "system", "content": "be terse"}]},
            },
        }
        frame = await self.serializer.deserialize(json.dumps(msg))
        self.assertIsInstance(frame, LLMMessagesUpdateFrame)
        self.assertEqual(frame.messages, [{"role": "system", "content": "be terse"}])
        self.assertFalse(frame.run_llm)

    async def test_non_reset_client_message_is_forwarded(self):
        msg = {
            "label": RTVI.MESSAGE_LABEL,
            "type": "client-message",
            "id": "5",
            "data": {"t": "something-else", "d": {}},
        }
        frame = await self.serializer.deserialize(json.dumps(msg))
        self.assertIsInstance(frame, InputTransportMessageFrame)

    async def test_non_rtvi_message_dropped(self):
        frame = await self.serializer.deserialize(json.dumps({"type": "send-text"}))
        self.assertIsNone(frame)

    async def test_non_json_dropped(self):
        self.assertIsNone(await self.serializer.deserialize("not json"))


class TestRTVIEvalSerializerSerialize(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.serializer = RTVIEvalSerializer()

    async def test_rtvi_server_message_serialized_to_json(self):
        message = RTVI.BotLLMStartedMessage().model_dump()
        frame = OutputTransportMessageUrgentFrame(message=message)
        payload = await self.serializer.serialize(frame)
        self.assertEqual(json.loads(payload)["type"], "bot-llm-started")

    async def test_non_rtvi_transport_message_dropped(self):
        frame = OutputTransportMessageUrgentFrame(message={"label": "other", "type": "x"})
        self.assertIsNone(await self.serializer.serialize(frame))

    async def test_audio_frame_dropped(self):
        frame = OutputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)
        self.assertIsNone(await self.serializer.serialize(frame))


if __name__ == "__main__":
    unittest.main()
