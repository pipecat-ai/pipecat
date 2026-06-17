#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for :class:`pipecat.evals.serializer.RTVIEvalSerializer`."""

import base64
import json
import unittest

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.evals.serializer import (
    EVAL_CONFIGURE_MESSAGE_TYPE,
    EVAL_CONTEXT_MESSAGE_TYPE,
    EVAL_IMAGE_MESSAGE_TYPE,
    RTVIEvalSerializer,
)
from pipecat.frames.frames import (
    InputTransportMessageFrame,
    LLMMessagesUpdateFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frameworks.rtvi.frames import RTVIConfigureObserverFrame
from pipecat.processors.frameworks.rtvi.observer import RTVIFunctionCallReportLevel


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

    async def test_eval_context_short_circuits_to_messages_update(self):
        msg = {
            "label": RTVI.MESSAGE_LABEL,
            "type": "client-message",
            "id": "4",
            "data": {
                "t": EVAL_CONTEXT_MESSAGE_TYPE,
                "d": {"messages": [{"role": "system", "content": "be terse"}]},
            },
        }
        frame = await self.serializer.deserialize(json.dumps(msg))
        self.assertIsInstance(frame, LLMMessagesUpdateFrame)
        self.assertEqual(frame.messages, [{"role": "system", "content": "be terse"}])
        self.assertFalse(frame.run_llm)

    async def test_eval_configure_short_circuits_to_observer_config(self):
        msg = {
            "label": RTVI.MESSAGE_LABEL,
            "type": "client-message",
            "id": "6",
            "data": {
                "t": EVAL_CONFIGURE_MESSAGE_TYPE,
                "d": {"function_call_report_level": {"*": "full"}},
            },
        }
        frame = await self.serializer.deserialize(json.dumps(msg))
        self.assertIsInstance(frame, RTVIConfigureObserverFrame)
        self.assertEqual(frame.function_call_report_level, {"*": RTVIFunctionCallReportLevel.FULL})

    async def test_eval_configure_enables_vad_user_speaking(self):
        msg = {
            "label": RTVI.MESSAGE_LABEL,
            "type": "client-message",
            "id": "8",
            "data": {
                "t": EVAL_CONFIGURE_MESSAGE_TYPE,
                "d": {"vad_user_speaking": True},
            },
        }
        frame = await self.serializer.deserialize(json.dumps(msg))
        self.assertIsInstance(frame, RTVIConfigureObserverFrame)
        self.assertTrue(frame.vad_user_speaking_enabled)
        # Unset report level stays None, so it isn't disturbed.
        self.assertIsNone(frame.function_call_report_level)

    async def test_eval_image_stored_and_not_forwarded(self):
        img = b"\x89PNG-fake-bytes"
        msg = {
            "label": RTVI.MESSAGE_LABEL,
            "type": "client-message",
            "id": "7",
            "data": {
                "t": EVAL_IMAGE_MESSAGE_TYPE,
                "d": {"image": base64.b64encode(img).decode("ascii"), "format": "image/png"},
            },
        }
        # eval-image is consumed (not forwarded) and kept for a later image request.
        self.assertIsNone(await self.serializer.deserialize(json.dumps(msg)))
        self.assertEqual(self.serializer.get_user_image(), (img, "image/png"))

    async def test_non_context_client_message_is_forwarded(self):
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
