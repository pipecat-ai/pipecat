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
    EVAL_BOT_AUDIO_TYPE,
    EVAL_CONFIGURE_MESSAGE_TYPE,
    EVAL_CONTEXT_MESSAGE_TYPE,
    EVAL_IMAGE_MESSAGE_TYPE,
    RTVIEvalSerializer,
    RTVIHarnessSerializer,
)
from pipecat.frames.frames import (
    InputAudioRawFrame,
    InputTransportMessageFrame,
    LLMFullResponseStartFrame,
    LLMMessagesUpdateFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
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

    async def test_dtmf_message_forwarded_to_processor(self):
        # DTMF is now a first-class RTVI message handled by the RTVIProcessor, so
        # the serializer just forwards it like any other RTVI message.
        msg = {
            "label": RTVI.MESSAGE_LABEL,
            "type": "dtmf",
            "id": "9",
            "data": {"button": "#"},
        }
        frame = await self.serializer.deserialize(json.dumps(msg))
        self.assertIsInstance(frame, InputTransportMessageFrame)
        self.assertEqual(frame.message["type"], "dtmf")

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


class TestRTVIHarnessSerializer(unittest.IsolatedAsyncioTestCase):
    """The client-side serializer the harness pipeline uses to talk to the bot."""

    def setUp(self):
        self.serializer = RTVIHarnessSerializer()

    def _server(self, msg_type: str, data: dict | None = None) -> str:
        return json.dumps({"label": RTVI.MESSAGE_LABEL, "type": msg_type, "data": data})

    async def test_eval_bot_audio_becomes_input_audio(self):
        pcm = b"\x01\x02\x03\x04"
        frame = await self.serializer.deserialize(
            self._server(
                EVAL_BOT_AUDIO_TYPE,
                {"audio": base64.b64encode(pcm).decode("ascii"), "sampleRate": 24000},
            )
        )
        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.audio, pcm)
        self.assertEqual(frame.sample_rate, 24000)
        self.assertEqual(frame.num_channels, 1)

    async def test_user_transcription_stays_a_raw_message(self):
        # Kept as the raw message (not a TranscriptionFrame) so the sink can tell it
        # apart from the STT's transcription of the bot's audio.
        data = {"text": "hello", "user_id": "u", "final": True}
        frame = await self.serializer.deserialize(self._server("user-transcription", data))
        self.assertIsInstance(frame, InputTransportMessageFrame)
        self.assertEqual(frame.message["type"], "user-transcription")
        self.assertEqual(frame.message["data"], data)

    async def test_generic_messages_delegate_to_base(self):
        # Non-eval RTVI messages keep the base RTVIClientSerializer mapping.
        self.assertIsInstance(
            await self.serializer.deserialize(self._server("bot-llm-started")),
            LLMFullResponseStartFrame,
        )
        self.assertIsNone(await self.serializer.deserialize(self._server("bot-ready")))
        self.assertIsNone(await self.serializer.deserialize("not json"))

    async def test_base_no_longer_maps_user_transcription_to_transcription(self):
        # Sanity: only the eval subclass diverts user-transcription; the generic
        # base still produces a TranscriptionFrame (unchanged for other clients).
        from pipecat.serializers.rtvi_client import RTVIClientSerializer

        frame = await RTVIClientSerializer().deserialize(
            self._server("user-transcription", {"text": "hi", "final": True})
        )
        self.assertIsInstance(frame, TranscriptionFrame)


if __name__ == "__main__":
    unittest.main()
