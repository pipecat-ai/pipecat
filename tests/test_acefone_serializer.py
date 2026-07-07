#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Acefone/Smartflo Media Streams serializer.

Audio round-trips are made deterministic by keeping the wire sample rate equal
to the pipeline sample rate, which makes the stream resampler an identity
passthrough (no buffering latency).
"""

import base64
import json

import pytest

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    StartFrame,
    TextFrame,
)
from pipecat.serializers.acefone import AcefoneFrameSerializer, AcefoneMediaFormat

# RTVI message label filtered out by the base serializer.
RTVI_LABEL = "rtvi-ai"


async def _make_serializer(
    *,
    media_format: AcefoneMediaFormat = AcefoneMediaFormat.ULAW,
    wire_rate: int = 8000,
    pipeline_rate: int = 8000,
    **params,
) -> AcefoneFrameSerializer:
    """Build a serializer and run ``setup()`` so sample rates are resolved.

    ``sample_rate`` is pinned to ``pipeline_rate`` so that, when it matches
    ``wire_rate``, resampling is an identity passthrough.
    """
    serializer = AcefoneFrameSerializer(
        AcefoneFrameSerializer.InputParams(
            media_format=media_format,
            acefone_sample_rate=wire_rate,
            sample_rate=pipeline_rate,
            **params,
        )
    )
    await serializer.setup(
        StartFrame(audio_in_sample_rate=pipeline_rate, audio_out_sample_rate=pipeline_rate)
    )
    return serializer


class TestAcefoneFrameSerializer:
    """Tests for AcefoneFrameSerializer."""

    # ==================== Initialization Tests ====================

    def test_create_serializer_default_params(self):
        """Default params: μ-law wire format, 8kHz, no auto hang-up."""
        serializer = AcefoneFrameSerializer()

        assert serializer._params.media_format == AcefoneMediaFormat.ULAW
        assert serializer._params.acefone_sample_rate == 8000
        assert serializer._params.auto_hang_up is False
        assert serializer._stream_sid is None
        assert serializer._hangup_attempted is False
        # BaseObject initialized (super().__init__ ran).
        assert serializer.name.startswith("AcefoneFrameSerializer")

    def test_create_serializer_with_custom_params(self):
        """Custom params are stored on the serializer."""
        params = AcefoneFrameSerializer.InputParams(
            media_format=AcefoneMediaFormat.PCM,
            acefone_sample_rate=16000,
            stream_sid="MZstream",
            call_sid="CAcall",
            auto_hang_up=True,
        )
        serializer = AcefoneFrameSerializer(params)

        assert serializer._params.media_format == AcefoneMediaFormat.PCM
        assert serializer._stream_sid == "MZstream"
        assert serializer._call_sid == "CAcall"
        assert serializer._auto_hang_up is True

    def test_media_format_enum_values(self):
        """The wire-format enum exposes the provider's string codecs."""
        assert AcefoneMediaFormat.ULAW.value == "ulaw"
        assert AcefoneMediaFormat.PCM.value == "slin16"

    # ==================== Setup Tests ====================

    @pytest.mark.asyncio
    async def test_setup_uses_start_frame_rate(self):
        """When sample_rate is 0, setup adopts the StartFrame's input rate."""
        serializer = AcefoneFrameSerializer()  # sample_rate defaults to 0
        await serializer.setup(StartFrame(audio_in_sample_rate=24000, audio_out_sample_rate=24000))
        assert serializer._sample_rate == 24000

    @pytest.mark.asyncio
    async def test_setup_respects_sample_rate_override(self):
        """An explicit sample_rate wins over the StartFrame's rate."""
        serializer = AcefoneFrameSerializer(AcefoneFrameSerializer.InputParams(sample_rate=16000))
        await serializer.setup(StartFrame(audio_in_sample_rate=8000, audio_out_sample_rate=8000))
        assert serializer._sample_rate == 16000

    # ==================== Serialize: Audio ====================

    @pytest.mark.asyncio
    async def test_serialize_ulaw_audio(self):
        """Outbound audio is encoded to μ-law in a media event."""
        serializer = await _make_serializer(stream_sid="MZstream")
        pcm = bytes(320)  # 160 samples of 16-bit PCM

        result = await serializer.serialize(
            OutputAudioRawFrame(audio=pcm, sample_rate=8000, num_channels=1)
        )

        message = json.loads(result)
        assert message["event"] == "media"
        assert message["streamSid"] == "MZstream"
        # μ-law is one byte per 16-bit sample.
        decoded = base64.b64decode(message["media"]["payload"])
        assert len(decoded) == len(pcm) // 2

    @pytest.mark.asyncio
    async def test_serialize_pcm_audio(self):
        """Outbound audio in PCM mode is sent as raw PCM (identity at equal rate)."""
        serializer = await _make_serializer(
            media_format=AcefoneMediaFormat.PCM, wire_rate=16000, pipeline_rate=16000
        )
        pcm = bytes(range(0, 256)) * 2  # 512 bytes of PCM

        result = await serializer.serialize(
            OutputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)
        )

        message = json.loads(result)
        assert message["event"] == "media"
        assert base64.b64decode(message["media"]["payload"]) == pcm

    @pytest.mark.asyncio
    async def test_serialize_empty_audio_returns_none(self):
        """Empty audio produces no media event."""
        serializer = await _make_serializer()
        result = await serializer.serialize(
            OutputAudioRawFrame(audio=b"", sample_rate=8000, num_channels=1)
        )
        assert result is None

    # ==================== Serialize: Control ====================

    @pytest.mark.asyncio
    async def test_serialize_interruption(self):
        """An interruption maps to a clear event (barge-in)."""
        serializer = await _make_serializer(stream_sid="MZstream")
        result = await serializer.serialize(InterruptionFrame())
        assert json.loads(result) == {"event": "clear", "streamSid": "MZstream"}

    @pytest.mark.asyncio
    async def test_serialize_end_frame_hangs_up_once(self):
        """With auto_hang_up, an EndFrame emits an end event exactly once."""
        serializer = await _make_serializer(
            stream_sid="MZstream", call_sid="CAcall", auto_hang_up=True
        )

        result = await serializer.serialize(EndFrame())
        assert json.loads(result) == {
            "event": "end",
            "streamSid": "MZstream",
            "call_sid": "CAcall",
        }
        # A subsequent end/cancel does not emit a second hang-up.
        assert await serializer.serialize(EndFrame()) is None

    @pytest.mark.asyncio
    async def test_serialize_cancel_frame_hangs_up(self):
        """A CancelFrame also triggers the end event when auto_hang_up is set."""
        serializer = await _make_serializer(
            stream_sid="MZstream", call_sid="CAcall", auto_hang_up=True
        )
        result = await serializer.serialize(CancelFrame())
        assert json.loads(result)["event"] == "end"

    @pytest.mark.asyncio
    async def test_serialize_end_frame_without_auto_hangup(self):
        """Without auto_hang_up, an EndFrame is not turned into an end event."""
        serializer = await _make_serializer(stream_sid="MZstream", auto_hang_up=False)
        assert await serializer.serialize(EndFrame()) is None

    @pytest.mark.asyncio
    async def test_serialize_transport_message(self):
        """Transport messages are forwarded as their raw JSON payload."""
        serializer = await _make_serializer()
        frame = OutputTransportMessageFrame(message={"event": "custom", "data": 1})
        result = await serializer.serialize(frame)
        assert json.loads(result) == {"event": "custom", "data": 1}

    @pytest.mark.asyncio
    async def test_serialize_rtvi_message_ignored(self):
        """RTVI messages are filtered out by the base serializer."""
        serializer = await _make_serializer()
        frame = OutputTransportMessageFrame(message={"label": RTVI_LABEL, "type": "x"})
        assert await serializer.serialize(frame) is None

    @pytest.mark.asyncio
    async def test_serialize_unhandled_frame_returns_none(self):
        """Frames the serializer doesn't handle yield None."""
        serializer = await _make_serializer()
        assert await serializer.serialize(TextFrame("hello")) is None

    # ==================== Deserialize: Audio ====================

    @pytest.mark.asyncio
    async def test_deserialize_ulaw_media(self):
        """Inbound μ-law media becomes a PCM InputAudioRawFrame."""
        serializer = await _make_serializer()
        ulaw = bytes(range(0, 160))
        message = json.dumps(
            {"event": "media", "media": {"payload": base64.b64encode(ulaw).decode()}}
        )

        frame = await serializer.deserialize(message)

        assert isinstance(frame, InputAudioRawFrame)
        assert frame.sample_rate == 8000
        assert frame.num_channels == 1
        # μ-law decodes to 16-bit PCM: two bytes per input byte.
        assert len(frame.audio) == len(ulaw) * 2

    @pytest.mark.asyncio
    async def test_deserialize_pcm_media(self):
        """Inbound PCM media is passed through (identity at equal rate)."""
        serializer = await _make_serializer(
            media_format=AcefoneMediaFormat.PCM, wire_rate=16000, pipeline_rate=16000
        )
        pcm = bytes(range(0, 200)) * 2
        message = json.dumps(
            {"event": "media", "media": {"payload": base64.b64encode(pcm).decode()}}
        )

        frame = await serializer.deserialize(message)

        assert isinstance(frame, InputAudioRawFrame)
        assert frame.audio == pcm
        assert frame.sample_rate == 16000

    @pytest.mark.asyncio
    async def test_deserialize_empty_media_returns_none(self):
        """A media event carrying no audio yields no frame."""
        serializer = await _make_serializer()
        message = json.dumps({"event": "media", "media": {"payload": ""}})
        assert await serializer.deserialize(message) is None

    # ==================== Deserialize: DTMF ====================

    @pytest.mark.asyncio
    async def test_deserialize_dtmf_digit(self):
        """A DTMF event becomes an InputDTMFFrame."""
        serializer = await _make_serializer()
        message = json.dumps({"event": "dtmf", "dtmf": {"digit": "5"}})

        frame = await serializer.deserialize(message)

        assert isinstance(frame, InputDTMFFrame)
        assert frame.button.value == "5"

    @pytest.mark.asyncio
    async def test_deserialize_dtmf_invalid_digit_returns_none(self):
        """An unrecognized DTMF digit is dropped."""
        serializer = await _make_serializer()
        message = json.dumps({"event": "dtmf", "dtmf": {"digit": "X"}})
        assert await serializer.deserialize(message) is None

    # ==================== Deserialize: Lifecycle ====================

    @pytest.mark.asyncio
    async def test_deserialize_start_captures_stream_sid(self):
        """A start event captures the stream SID when one isn't set."""
        serializer = await _make_serializer()
        assert serializer._stream_sid is None

        result = await serializer.deserialize(
            json.dumps({"event": "start", "streamSid": "MZfromstart"})
        )

        assert result is None
        assert serializer._stream_sid == "MZfromstart"

    @pytest.mark.asyncio
    async def test_deserialize_start_does_not_overwrite_stream_sid(self):
        """A start event does not clobber an already-configured stream SID."""
        serializer = await _make_serializer(stream_sid="MZconfigured")

        await serializer.deserialize(json.dumps({"event": "start", "streamSid": "MZfromstart"}))

        assert serializer._stream_sid == "MZconfigured"

    @pytest.mark.asyncio
    async def test_deserialize_unknown_event_returns_none(self):
        """Unknown events are ignored."""
        serializer = await _make_serializer()
        assert await serializer.deserialize(json.dumps({"event": "mark"})) is None

    # ==================== Round Trip ====================

    @pytest.mark.asyncio
    async def test_pcm_round_trip(self):
        """PCM audio survives serialize -> deserialize unchanged at equal rates."""
        serializer = await _make_serializer(
            media_format=AcefoneMediaFormat.PCM,
            wire_rate=16000,
            pipeline_rate=16000,
            stream_sid="MZstream",
        )
        pcm = bytes(range(0, 256)) * 4

        serialized = await serializer.serialize(
            OutputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)
        )
        payload = json.loads(serialized)["media"]["payload"]
        frame = await serializer.deserialize(
            json.dumps({"event": "media", "media": {"payload": payload}})
        )

        assert isinstance(frame, InputAudioRawFrame)
        assert frame.audio == pcm


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
