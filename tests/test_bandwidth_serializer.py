#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Bandwidth Programmable Voice WebSocket serializer."""

import base64
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    StartFrame,
)
from pipecat.serializers.bandwidth import BandwidthFrameSerializer


def _make_serializer(auto_hang_up: bool = False, **kwargs) -> BandwidthFrameSerializer:
    """Build a serializer with sensible defaults for tests."""
    params = BandwidthFrameSerializer.InputParams(auto_hang_up=auto_hang_up, **kwargs)
    return BandwidthFrameSerializer(
        stream_id="test-stream-id",
        call_id="test-call-id",
        account_id="test-account-id",
        username="test-user",
        password="test-pass",
        params=params,
    )


async def _setup(serializer: BandwidthFrameSerializer, sample_rate: int = 16000) -> None:
    """Drive setup() with a StartFrame so the serializer knows the pipeline rate."""
    start = StartFrame(
        audio_in_sample_rate=sample_rate,
        audio_out_sample_rate=sample_rate,
    )
    await serializer.setup(start)


class TestBandwidthFrameSerializerInit(unittest.IsolatedAsyncioTestCase):
    """Initialization and parameter validation."""

    def test_default_params(self):
        serializer = _make_serializer()
        self.assertEqual(serializer.stream_id, "test-stream-id")
        self.assertEqual(serializer.call_id, "test-call-id")
        self.assertEqual(serializer._params.outbound_encoding, "PCMU")
        self.assertEqual(serializer._params.bandwidth_sample_rate, 8000)

    def test_auto_hang_up_requires_credentials(self):
        with self.assertRaises(ValueError) as ctx:
            BandwidthFrameSerializer(
                stream_id="s",
                params=BandwidthFrameSerializer.InputParams(auto_hang_up=True),
            )
        msg = str(ctx.exception)
        for required in ("call_id", "account_id", "username", "password"):
            self.assertIn(required, msg)

    def test_auto_hang_up_disabled_skips_credential_check(self):
        # Should not raise.
        BandwidthFrameSerializer(
            stream_id="s",
            params=BandwidthFrameSerializer.InputParams(auto_hang_up=False),
        )

    def test_invalid_outbound_encoding(self):
        with self.assertRaises(ValueError):
            _make_serializer(outbound_encoding="OPUS")

    def test_invalid_pcm_rate(self):
        with self.assertRaises(ValueError):
            _make_serializer(outbound_encoding="PCM", outbound_pcm_sample_rate=11025)


class TestBandwidthFrameSerializerDeserialize(unittest.IsolatedAsyncioTestCase):
    """Inbound message handling."""

    async def test_media_event_inbound_track_produces_audio_frame(self):
        serializer = _make_serializer()
        # Match pipeline rate to Bandwidth's wire rate so the resampler is a
        # pass-through; we're testing the deserializer's framing logic, not
        # the resampler (which has its own coverage and a warm-up buffer that
        # complicates small-chunk unit tests).
        await _setup(serializer, sample_rate=8000)

        ulaw_silence = b"\xff" * 160
        message = {
            "eventType": "media",
            "track": "inbound",
            "payload": base64.b64encode(ulaw_silence).decode(),
            "sequenceNumber": "1",
        }
        frame = await serializer.deserialize(json.dumps(message))

        self.assertIsInstance(frame, InputAudioRawFrame)
        self.assertEqual(frame.sample_rate, 8000)
        self.assertEqual(frame.num_channels, 1)
        # ulaw → 16-bit PCM doubles size: 160 ulaw bytes → 320 PCM bytes.
        self.assertEqual(len(frame.audio), 320)

    async def test_media_event_outbound_track_is_ignored(self):
        """When tracks='both' Bandwidth echoes the bot's own audio back; drop it."""
        serializer = _make_serializer()
        await _setup(serializer)

        message = {
            "eventType": "media",
            "track": "outbound",
            "payload": base64.b64encode(b"\xff" * 80).decode(),
            "sequenceNumber": "1",
        }
        self.assertIsNone(await serializer.deserialize(json.dumps(message)))

    async def test_start_event_returns_none(self):
        serializer = _make_serializer()
        await _setup(serializer)
        message = {
            "eventType": "start",
            "metadata": {
                "accountId": "acc",
                "callId": "call",
                "streamId": "stream",
                "streamName": "name",
                "tracks": ["inbound"],
            },
        }
        self.assertIsNone(await serializer.deserialize(json.dumps(message)))

    async def test_stop_event_returns_none(self):
        serializer = _make_serializer()
        await _setup(serializer)
        self.assertIsNone(
            await serializer.deserialize(json.dumps({"eventType": "stop", "metadata": {}}))
        )

    async def test_invalid_json_returns_none(self):
        serializer = _make_serializer()
        await _setup(serializer)
        self.assertIsNone(await serializer.deserialize("not json"))

    async def test_unknown_event_type_returns_none(self):
        serializer = _make_serializer()
        await _setup(serializer)
        self.assertIsNone(await serializer.deserialize(json.dumps({"eventType": "made-up"})))


class TestBandwidthFrameSerializerSerialize(unittest.IsolatedAsyncioTestCase):
    """Outbound frame serialization."""

    async def test_audio_frame_pcmu(self):
        serializer = _make_serializer()
        # Align rates with bandwidth_sample_rate to bypass the resampler's
        # warm-up buffer; this test asserts framing/encoding, not resampling.
        await _setup(serializer, sample_rate=8000)

        audio = b"\x00\x00" * 800  # 100ms of silence at 8kHz
        frame = OutputAudioRawFrame(audio=audio, sample_rate=8000, num_channels=1)

        result = await serializer.serialize(frame)
        self.assertIsNotNone(result)
        msg = json.loads(result)
        self.assertEqual(msg["eventType"], "playAudio")
        self.assertEqual(msg["media"]["contentType"], "audio/pcmu")
        decoded = base64.b64decode(msg["media"]["payload"])
        # μ-law halves the byte count vs. 16-bit PCM: 1600 → 800.
        self.assertEqual(len(decoded), 800)

    async def test_audio_frame_pcm_24khz(self):
        serializer = _make_serializer(outbound_encoding="PCM", outbound_pcm_sample_rate=24000)
        # Align frame rate with target PCM rate so the resampler short-circuits.
        await _setup(serializer, sample_rate=24000)

        audio = b"\x00\x00" * 2400  # 100ms at 24kHz
        frame = OutputAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1)

        result = await serializer.serialize(frame)
        self.assertIsNotNone(result)
        msg = json.loads(result)
        self.assertEqual(msg["eventType"], "playAudio")
        self.assertIn("audio/pcm;rate=24000", msg["media"]["contentType"])
        self.assertIn("bit-depth=16", msg["media"]["contentType"])
        self.assertIn("endian=little", msg["media"]["contentType"])
        # PCM payload should round-trip the original (no resampling, no μ-law).
        decoded = base64.b64decode(msg["media"]["payload"])
        self.assertEqual(decoded, audio)

    async def test_interruption_frame_emits_clear(self):
        serializer = _make_serializer()
        await _setup(serializer)

        result = await serializer.serialize(InterruptionFrame())
        self.assertEqual(json.loads(result), {"eventType": "clear"})

    async def test_output_transport_message_passthrough(self):
        serializer = _make_serializer()
        await _setup(serializer)

        custom = {"eventType": "custom", "data": {"foo": "bar"}}
        result = await serializer.serialize(OutputTransportMessageFrame(message=custom))
        self.assertEqual(json.loads(result), custom)

    async def test_unhandled_frame_returns_none(self):
        serializer = _make_serializer()
        await _setup(serializer)
        self.assertIsNone(await serializer.serialize(StartFrame()))


class TestBandwidthFrameSerializerHangup(unittest.IsolatedAsyncioTestCase):
    """End-of-call hang-up REST invocation."""

    async def test_end_frame_triggers_hangup(self):
        serializer = _make_serializer(auto_hang_up=True)
        await _setup(serializer)

        # Mock aiohttp.ClientSession so we don't make a real HTTP call.
        with patch("aiohttp.ClientSession") as session_cls:
            response = MagicMock()
            response.status = 200
            response.text = AsyncMock(return_value="")

            post_ctx = MagicMock()
            post_ctx.__aenter__ = AsyncMock(return_value=response)
            post_ctx.__aexit__ = AsyncMock(return_value=None)

            session = MagicMock()
            session.post = MagicMock(return_value=post_ctx)
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=None)

            session_cls.return_value = session

            result = await serializer.serialize(EndFrame())

            self.assertIsNone(result)
            self.assertTrue(serializer._hangup_attempted)
            session.post.assert_called_once()
            call_args = session.post.call_args
            url = call_args.args[0]
            self.assertIn("voice.bandwidth.com/api/v2", url)
            self.assertIn("test-account-id", url)
            self.assertIn("test-call-id", url)
            self.assertEqual(call_args.kwargs["json"], {"state": "completed"})

    async def test_cancel_frame_triggers_hangup(self):
        serializer = _make_serializer(auto_hang_up=True)
        await _setup(serializer)

        with patch.object(serializer, "_hang_up_call", new=AsyncMock()) as mock_hangup:
            await serializer.serialize(CancelFrame())
            mock_hangup.assert_awaited_once()

    async def test_hangup_only_attempted_once(self):
        serializer = _make_serializer(auto_hang_up=True)
        await _setup(serializer)

        with patch.object(serializer, "_hang_up_call", new=AsyncMock()) as mock_hangup:
            await serializer.serialize(EndFrame())
            await serializer.serialize(EndFrame())
            mock_hangup.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
