#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for DeepgramFluxSageMakerTTSService."""

import asyncio
import json
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs

import pytest

pytest.importorskip("aws_sdk_sagemaker_runtime_http2")

from aws_sdk_sagemaker_runtime_http2.models import (  # noqa: E402
    ResponsePayloadPart,
    ResponseStreamEventPayloadPart,
)

from pipecat.frames.frames import (  # noqa: E402
    BotStoppedSpeakingFrame,
    ErrorFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.deepgram.flux.sagemaker.tts import (  # noqa: E402
    DeepgramFluxSageMakerTTSService,
)
from pipecat.tests.utils import SleepFrame, run_test  # noqa: E402

AUDIO_CHUNK_1 = b"\x00\x01" * 512
AUDIO_CHUNK_2 = b"\x02\x03" * 512


class FakeBidiClient:
    """A SageMaker BiDi client backed by a fake Flux TTS endpoint.

    Answers the Flux turn flow the way the service expects it: a Speak opens the
    turn, a Flush produces the turn's audio and its closing SpeechMetadata.
    """

    #: Every client built during a test, in construction order.
    instances: list["FakeBidiClient"] = []

    #: Whether payload parts are labelled with a data type. Endpoints that leave
    #: them unset exercise the service's decode-and-parse fallback.
    tag_data_type = True

    #: Whether a flushed turn is closed with SpeechMetadata. False leaves the
    #: turn open, the way it is while Flux is still synthesizing audio.
    end_turn = True

    def __init__(self, *, endpoint_name, region, model_invocation_path, model_query_string):
        self.endpoint_name = endpoint_name
        self.region = region
        self.model_invocation_path = model_invocation_path
        self.model_query_string = model_query_string
        self.messages: list[dict] = []
        self.is_active = False
        self._responses: asyncio.Queue = asyncio.Queue()
        self._speech_started = False
        FakeBidiClient.instances.append(self)

    async def start_session(self):
        self.is_active = True

    async def close_session(self):
        self.is_active = False
        await self._responses.put(None)

    async def send_json(self, data: dict):
        if not self.is_active:
            raise RuntimeError("BiDi session not active")

        self.messages.append(data)
        msg_type = data.get("type")

        if msg_type == "Speak" and not self._speech_started:
            self._speech_started = True
            await self._send_json_response(
                {"type": "Connected", "request_id": "test-request", "model_name": "flux-alexis-en"}
            )
            await self._send_json_response({"type": "SpeechStarted", "speech_id": "dg_sp_test"})
        elif msg_type == "Flush":
            # Flux sends the flush ack before the turn's remaining audio;
            # SpeechMetadata arrives only after all audio.
            await self._send_audio_response(AUDIO_CHUNK_1)
            await self._send_json_response({"type": "Flushed", "speech_id": "dg_sp_test"})
            await self._send_audio_response(AUDIO_CHUNK_2)
            if self.end_turn:
                await self._send_json_response(
                    {
                        "type": "SpeechMetadata",
                        "speech_id": "dg_sp_test",
                        "audio_duration_ms": 100,
                        "input_character_count": 17,
                        "billable_character_count": 17,
                    }
                )
        elif msg_type == "Interrupt":
            await self._send_json_response(
                {"type": "SpeechInterrupted", "speech_id": "dg_sp_test", "audio_played_ms": 40}
            )
        elif msg_type == "Configure":
            await self._send_json_response({"type": "ConfigureSuccess", "applied": data})

    async def receive_response(self):
        return await self._responses.get()

    async def _send_json_response(self, message: dict):
        await self._send_response(
            json.dumps(message).encode("utf-8"), "UTF8" if self.tag_data_type else None
        )

    async def _send_audio_response(self, audio: bytes):
        await self._send_response(audio, "BINARY" if self.tag_data_type else None)

    async def _send_response(self, payload: bytes, data_type: str | None):
        await self._responses.put(
            ResponseStreamEventPayloadPart(
                value=ResponsePayloadPart(bytes_=payload, data_type=data_type)
            )
        )


@pytest.fixture
def fake_client():
    """Replace the BiDi client with the fake endpoint for the duration of a test."""
    FakeBidiClient.instances = []
    FakeBidiClient.tag_data_type = True
    FakeBidiClient.end_turn = True
    with patch(
        "pipecat.services.deepgram.flux.sagemaker.tts.SageMakerBidiClient", new=FakeBidiClient
    ):
        yield FakeBidiClient


@pytest.mark.asyncio
async def test_flux_sagemaker_tts_protocol_roundtrip(fake_client):
    """Speak/Flush are sent, and audio (including post-Flushed audio) is emitted."""
    tts_service = DeepgramFluxSageMakerTTSService(
        endpoint_name="test-endpoint",
        region="us-east-2",
        sample_rate=24000,
        settings=DeepgramFluxSageMakerTTSService.Settings(voice="flux-alexis-en"),
    )

    down_frames, up_frames = await run_test(
        tts_service,
        frames_to_send=[
            TTSSpeakFrame(text="Hello from Flux."),
            SleepFrame(sleep=0.3),
            # With pause_frame_processing=True the transport's
            # BotStoppedSpeakingFrame resumes frame processing; there is no
            # transport in run_test, so send it explicitly (it is a system
            # frame, so it bypasses the paused queue).
            BotStoppedSpeakingFrame(),
        ],
    )

    frame_types = [type(frame) for frame in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSAudioRawFrame in frame_types
    assert TTSStoppedFrame in frame_types
    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)

    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert all(frame.sample_rate == 24000 for frame in audio_frames)
    assert all(frame.num_channels == 1 for frame in audio_frames)
    # Audio sent after Flushed but before SpeechMetadata must not be dropped.
    assert b"".join(frame.audio for frame in audio_frames) == AUDIO_CHUNK_1 + AUDIO_CHUNK_2

    client = fake_client.instances[0]
    assert client.endpoint_name == "test-endpoint"
    assert client.region == "us-east-2"
    assert client.model_invocation_path == "v2/speak"

    query = parse_qs(client.model_query_string)
    assert query["model"] == ["flux-alexis-en"]
    assert query["encoding"] == ["linear16"]
    assert query["sample_rate"] == ["24000"]
    assert "speed" not in query
    assert "expressivity" not in query

    types_sent = [m.get("type") for m in client.messages]
    assert "Flush" in types_sent
    speak_msg = next(m for m in client.messages if m.get("type") == "Speak")
    # In the default token streaming mode, text is sent verbatim.
    assert speak_msg["text"] == "Hello from Flux."


@pytest.mark.asyncio
async def test_flux_sagemaker_tts_untagged_payloads_are_routed_by_content(fake_client):
    """Payload parts with no data type are routed by whether they parse as JSON."""
    fake_client.tag_data_type = False

    tts_service = DeepgramFluxSageMakerTTSService(
        endpoint_name="test-endpoint",
        region="us-east-2",
        sample_rate=24000,
    )

    down_frames, up_frames = await run_test(
        tts_service,
        frames_to_send=[
            TTSSpeakFrame(text="Hello from Flux."),
            SleepFrame(sleep=0.3),
            BotStoppedSpeakingFrame(),
        ],
    )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    # TTSStoppedFrame comes from SpeechMetadata, so the JSON messages were
    # recognized as well as the audio.
    assert any(isinstance(frame, TTSStoppedFrame) for frame in down_frames)

    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert b"".join(frame.audio for frame in audio_frames) == AUDIO_CHUNK_1 + AUDIO_CHUNK_2


@pytest.mark.asyncio
async def test_flux_sagemaker_tts_interruption_sends_interrupt(fake_client):
    """On barge-in the service sends Interrupt and keeps the session open."""
    fake_client.end_turn = False

    tts_service = DeepgramFluxSageMakerTTSService(
        endpoint_name="test-endpoint",
        region="us-east-2",
        sample_rate=24000,
    )

    await run_test(
        tts_service,
        frames_to_send=[
            TTSSpeakFrame(text="Hello from Flux."),
            SleepFrame(sleep=0.2),
            InterruptionFrame(),
            SleepFrame(sleep=0.2),
        ],
    )

    # A single session serves the whole conversation: a barge-in cancels the
    # turn without reconnecting, so the cross-turn acoustic state survives it.
    assert len(fake_client.instances) == 1
    assert any(m.get("type") == "Interrupt" for m in fake_client.instances[0].messages)
    assert tts_service.get_active_audio_context_id() is None
    assert not tts_service.get_audio_contexts()


def _connected_service(**kwargs) -> tuple[DeepgramFluxSageMakerTTSService, FakeBidiClient]:
    """Build a service with a stub open session, without connecting."""
    tts_service = DeepgramFluxSageMakerTTSService(
        endpoint_name="test-endpoint",
        region="us-east-2",
        sample_rate=24000,
        **kwargs,
    )
    client = FakeBidiClient(
        endpoint_name="test-endpoint",
        region="us-east-2",
        model_invocation_path="v2/speak",
        model_query_string=tts_service._build_query_string(),
    )
    client.is_active = True
    tts_service._client = client
    return tts_service, client


@pytest.mark.asyncio
async def test_flux_sagemaker_tts_speed_update_sends_configure(fake_client):
    """A speed change is applied on the open session, without reconnecting."""
    tts_service, client = _connected_service()

    with (
        patch.object(tts_service, "_connect", new=AsyncMock()) as connect_spy,
        patch.object(tts_service, "_disconnect", new=AsyncMock()) as disconnect_spy,
    ):
        await tts_service._update_settings(DeepgramFluxSageMakerTTSService.Settings(speed=1.1))

    assert not disconnect_spy.called
    assert not connect_spy.called
    assert client.messages == [{"type": "Configure", "speed": 1.1}]
    assert tts_service._settings.speed == 1.1


@pytest.mark.asyncio
async def test_flux_sagemaker_tts_voice_update_reconnects(fake_client):
    """The voice is a query parameter, so a change reconnects."""
    tts_service, client = _connected_service()

    with (
        patch.object(tts_service, "_connect", new=AsyncMock()) as connect_spy,
        patch.object(tts_service, "_disconnect", new=AsyncMock()) as disconnect_spy,
    ):
        await tts_service._update_settings(
            DeepgramFluxSageMakerTTSService.Settings(voice="flux-alexis-en")
        )

    assert disconnect_spy.called
    assert connect_spy.called
    assert not client.messages
    # Deepgram passes the voice as its model, so metrics follow the voice.
    assert tts_service._settings.model == "flux-alexis-en"
    assert "model=flux-alexis-en" in tts_service._build_query_string()
