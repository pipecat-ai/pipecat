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
        client_id="test-client-id",
        client_secret="test-client-secret",
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
        for required in ("call_id", "account_id", "client_id", "client_secret"):
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

    @staticmethod
    def _make_post_ctx(status: int, json_body: dict | None = None) -> MagicMock:
        """Build a mocked async context manager for ``session.post(...)``."""
        response = MagicMock()
        response.status = status
        response.json = AsyncMock(return_value=json_body or {})
        response.text = AsyncMock(return_value="")
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    @staticmethod
    def _make_session(post_side_effects: list[MagicMock]) -> MagicMock:
        session = MagicMock()
        session.post = MagicMock(side_effect=post_side_effects)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        return session

    async def test_end_frame_triggers_oauth_then_hangup(self):
        """EndFrame should fetch a Bearer token, then terminate the call with it."""
        serializer = _make_serializer(auto_hang_up=True)
        await _setup(serializer)

        token_ctx = self._make_post_ctx(
            200, {"access_token": "fake-jwt", "token_type": "Bearer", "expires_in": 3600}
        )
        hangup_ctx = self._make_post_ctx(200)
        session = self._make_session([token_ctx, hangup_ctx])

        with patch("aiohttp.ClientSession", return_value=session):
            result = await serializer.serialize(EndFrame())

        self.assertIsNone(result)
        self.assertTrue(serializer._hangup_attempted)
        self.assertEqual(session.post.call_count, 2)

        # First call: OAuth token request.
        token_call = session.post.call_args_list[0]
        self.assertIn("oauth2/token", token_call.args[0])
        self.assertEqual(token_call.kwargs["data"], {"grant_type": "client_credentials"})
        # Basic Auth header is set via the `auth` kwarg with client_id:client_secret.
        self.assertIsNotNone(token_call.kwargs.get("auth"))

        # Second call: call termination with Bearer token.
        hangup_call = session.post.call_args_list[1]
        url = hangup_call.args[0]
        self.assertIn("voice.bandwidth.com/api/v2", url)
        self.assertIn("test-account-id", url)
        self.assertIn("test-call-id", url)
        self.assertEqual(hangup_call.kwargs["json"], {"state": "completed"})
        self.assertEqual(hangup_call.kwargs["headers"]["Authorization"], "Bearer fake-jwt")

    async def test_token_fetch_failure_skips_hangup_request(self):
        """If OAuth fails, we should not attempt the call-termination request."""
        serializer = _make_serializer(auto_hang_up=True)
        await _setup(serializer)

        token_ctx = self._make_post_ctx(401, {"error": "invalid_client"})
        # Only one POST should happen — the second side_effect would raise StopIteration
        # if the code accidentally tries the hangup anyway, which would fail this test.
        session = self._make_session([token_ctx])

        with patch("aiohttp.ClientSession", return_value=session):
            await serializer.serialize(EndFrame())

        self.assertEqual(session.post.call_count, 1)

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
