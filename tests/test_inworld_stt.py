#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Inworld speech-to-text service."""

import base64
import json
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.inworld.frames import InworldVoiceProfile, InworldVoiceProfileFrame
from pipecat.services.inworld.stt import (
    InworldRealtimeSTTService,
    InworldSTTService,
    language_to_inworld_stt_language,
)
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies


class _FakeWebsocket:
    def __init__(self, messages=None, *, state=State.OPEN):
        self._messages = messages or []
        self.state = state
        self.send = AsyncMock()
        self.close = AsyncMock()

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


def _realtime_service(**kwargs) -> InworldRealtimeSTTService:
    service = InworldRealtimeSTTService(
        api_key="encoded-key",
        sample_rate=16000,
        **kwargs,
    )
    service._sample_rate = 16000
    service._user_id = "user-1"
    return service


@pytest.mark.asyncio
async def test_inworld_stt_sends_documented_request_and_emits_transcription(aiohttp_client):
    """The service should send WAV audio and expose the Inworld result on the frame."""
    captured = {}

    async def handler(request):
        captured["headers"] = request.headers
        captured["body"] = await request.json()
        return web.json_response(
            {
                "transcription": {
                    "transcript": "  Hello from Inworld.  ",
                    "isFinal": True,
                    "wordTimestamps": [],
                },
                "voiceProfile": {
                    "age": [{"label": "adult", "confidence": 0.88}],
                    "emotion": [{"label": "calm", "confidence": 0.91}],
                    "pitch": [{"label": "medium", "confidence": 0.82}],
                    "vocalStyle": [{"label": "normal", "confidence": 0.95}],
                    "accent": [{"label": "en-US", "confidence": 0.76}],
                },
                "usage": {
                    "transcribedAudioMs": 1200,
                    "modelId": "inworld/inworld-stt-1",
                },
            }
        )

    app = web.Application()
    app.router.add_post("/stt/v1/transcribe", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")
    audio = b"RIFF-test-wav"

    async with aiohttp.ClientSession() as session:
        service = InworldSTTService(
            api_key="encoded-key",
            aiohttp_session=session,
            base_url=base_url,
            settings=InworldSTTService.Settings(
                language=Language.EN_US,
                prompts=["Pipecat", "Inworld"],
                enable_voice_profile=True,
                voice_profile_top_n=3,
            ),
        )
        service._sample_rate = 16000
        service._user_id = "user-1"

        frames = [frame async for frame in service.run_stt(audio)]

    assert captured["headers"]["Authorization"] == "Basic encoded-key"
    assert captured["headers"]["X-User-Agent"].startswith("pipecat/")
    assert captured["headers"]["X-Request-Id"]
    assert captured["body"] == {
        "transcribeConfig": {
            "modelId": "inworld/inworld-stt-1",
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "numberOfChannels": 1,
            "language": "en",
            "prompts": ["Pipecat", "Inworld"],
            "voiceProfileConfig": {
                "enableVoiceProfile": True,
                "topN": 3,
            },
        },
        "audioData": {"content": base64.b64encode(audio).decode("ascii")},
    }

    assert len(frames) == 2
    assert isinstance(frames[0], InworldVoiceProfileFrame)
    assert frames[0].user_id == "user-1"
    assert frames[0].voice_profile.emotion[0].label == "calm"
    assert frames[0].voice_profile.emotion[0].confidence == 0.91
    assert frames[0].voice_profile.vocal_style[0].label == "normal"

    assert isinstance(frames[1], TranscriptionFrame)
    assert frames[1].text == "Hello from Inworld."
    assert frames[1].user_id == "user-1"
    assert frames[1].language == Language.EN
    assert frames[1].result is not None
    assert frames[1].result["usage"]["transcribedAudioMs"] == 1200


@pytest.mark.asyncio
async def test_inworld_stt_omits_optional_settings_when_auto_detecting(aiohttp_client):
    """Auto-detection requests should omit language and empty prompts."""
    request_bodies = []

    async def handler(request):
        request_bodies.append(await request.json())
        return web.json_response({"transcription": {"transcript": ""}})

    app = web.Application()
    app.router.add_post("/stt/v1/transcribe", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        service = InworldSTTService(
            api_key="encoded-key",
            aiohttp_session=session,
            base_url=base_url,
        )
        service._sample_rate = 8000

        frames = [frame async for frame in service.run_stt(b"RIFF")]

    assert frames == []
    config = request_bodies[0]["transcribeConfig"]
    assert config["sampleRateHertz"] == 8000
    assert "language" not in config
    assert "prompts" not in config


@pytest.mark.asyncio
async def test_inworld_stt_emits_error_frame_for_api_errors(aiohttp_client):
    """Provider errors should enter Pipecat's normal non-fatal error path."""

    async def handler(request):
        return web.json_response(
            {"code": 8, "message": "rate limit exceeded", "details": []},
            status=429,
        )

    app = web.Application()
    app.router.add_post("/stt/v1/transcribe", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        service = InworldSTTService(
            api_key="encoded-key",
            aiohttp_session=session,
            base_url=base_url,
        )
        service._sample_rate = 16000

        frames = [frame async for frame in service.run_stt(b"RIFF")]

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert frames[0].fatal is False
    assert "Inworld API error (429)" in frames[0].error
    assert "rate limit exceeded" in frames[0].error
    assert isinstance(frames[0].exception, RuntimeError)


def test_inworld_stt_language_mapping_uses_base_codes():
    """Regional variants should resolve to the base codes accepted by Inworld STT."""
    assert language_to_inworld_stt_language(Language.EN_GB) == "en"
    assert language_to_inworld_stt_language(Language.PT_BR) == "pt"
    assert language_to_inworld_stt_language(Language.ZH_TW) == "zh"
    assert language_to_inworld_stt_language(Language.FIL) == "fil"


def test_inworld_voice_profile_accepts_snake_case_fields():
    """Voice Profile models should accept both documented response naming styles."""
    profile = InworldVoiceProfile.model_validate(
        {
            "vocal_style": [{"label": "whispering", "confidence": 0.97}],
            "emotion": [{"label": "tender", "confidence": 0.84}],
        }
    )

    assert profile.vocal_style[0].label == "whispering"
    assert profile.emotion[0].confidence == 0.84


@pytest.mark.asyncio
async def test_inworld_realtime_stt_connects_with_documented_config(monkeypatch):
    """The first WebSocket message should configure Inworld before audio is sent."""
    captured = {}
    websocket = _FakeWebsocket()

    async def fake_websocket_connect(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs["additional_headers"]
        return websocket

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect",
        fake_websocket_connect,
    )

    service = _realtime_service(
        base_url="api.inworld.ai",
        vad_force_turn_endpoint=False,
        settings=InworldRealtimeSTTService.Settings(
            language=Language.EN_US,
            prompts=["Pipecat"],
            enable_voice_profile=True,
            voice_profile_top_n=3,
            vad_threshold=0.5,
            min_end_of_turn_silence_when_confident=250,
            end_of_turn_confidence_threshold=0.4,
            inactivity_timeout_seconds=30,
        ),
    )

    await service._connect_websocket()

    assert captured["url"] == ("wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional")
    assert captured["headers"]["Authorization"] == "Basic encoded-key"
    assert captured["headers"]["X-Request-Id"]
    assert captured["headers"]["X-User-Agent"].startswith("pipecat/")
    config_message = json.loads(websocket.send.await_args.args[0])
    assert config_message == {
        "transcribeConfig": {
            "modelId": "inworld/inworld-stt-1",
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "numberOfChannels": 1,
            "language": "en",
            "prompts": ["Pipecat"],
            "voiceProfileConfig": {
                "enableVoiceProfile": True,
                "topN": 3,
            },
            "endOfTurnConfidenceThreshold": 0.4,
            "inactivityTimeoutSeconds": 30,
            "inworldSttV1Config": {
                "vadThreshold": 0.5,
                "minEndOfTurnSilenceWhenConfident": 250,
            },
        }
    }


@pytest.mark.asyncio
async def test_inworld_realtime_stt_streams_base64_audio():
    """Raw PCM frames should be sent in the documented audioChunk envelope."""
    service = _realtime_service()
    websocket = _FakeWebsocket()
    service._websocket = websocket
    audio = b"\x01\x02\x03\x04"

    frames = [frame async for frame in service.run_stt(audio)]

    assert frames == [None]
    assert json.loads(websocket.send.await_args.args[0]) == {
        "audioChunk": {"content": base64.b64encode(audio).decode("ascii")}
    }


@pytest.mark.asyncio
async def test_inworld_realtime_stt_sends_end_turn_from_pipecat_vad(monkeypatch):
    """Pipecat turn mode should map VAD stops to Inworld endTurn messages."""
    service = _realtime_service()
    websocket = _FakeWebsocket()
    service._websocket = websocket
    monkeypatch.setattr(service, "push_frame", AsyncMock())

    assert service._transcribe_config()["inworldSttV1Config"]["vadThreshold"] == 0

    await service.process_frame(
        VADUserStoppedSpeakingFrame(),
        FrameDirection.DOWNSTREAM,
    )

    assert json.loads(websocket.send.await_args.args[0]) == {"endTurn": {}}


def _instrument_realtime_service(monkeypatch, events, **kwargs):
    service = _realtime_service(**kwargs)

    async def fake_push_frame(frame, direction=None):
        events.append(("push", type(frame), frame))

    async def fake_broadcast_frame(frame_cls, **frame_kwargs):
        events.append(("broadcast", frame_cls, None))

    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "broadcast_frame", fake_broadcast_frame)
    monkeypatch.setattr(service, "_handle_transcription", AsyncMock())
    return service


@pytest.mark.asyncio
async def test_inworld_realtime_stt_emits_interim_final_and_voice_profile(monkeypatch):
    """Streaming results should preserve transcript state and Voice Profile data."""
    events = []
    service = _instrument_realtime_service(monkeypatch, events)

    await service._process_response(
        {
            "result": {
                "transcription": {
                    "transcript": "Hello",
                    "isFinal": False,
                }
            }
        }
    )
    final_result = {
        "transcription": {
            "transcript": "Hello from Inworld.",
            "isFinal": True,
            "voiceProfile": {
                "emotion": [{"label": "calm", "confidence": 0.91}],
                "vocalStyle": [{"label": "conversational", "confidence": 0.86}],
            },
        },
    }
    await service._process_response({"result": final_result})

    pushed = [event[2] for event in events if event[0] == "push"]
    assert [type(frame) for frame in pushed] == [
        InterimTranscriptionFrame,
        InworldVoiceProfileFrame,
        TranscriptionFrame,
    ]
    assert pushed[0].text == "Hello"
    assert pushed[1].voice_profile.emotion[0].label == "calm"
    assert pushed[1].voice_profile.vocal_style[0].label == "conversational"
    assert pushed[2].text == "Hello from Inworld."
    assert pushed[2].finalized is True
    assert pushed[2].result == final_result
    service._handle_transcription.assert_awaited_once_with("Hello from Inworld.", True, None)


def test_inworld_realtime_stt_recommends_turn_strategies_in_inworld_mode():
    """Inworld-owned endpointing should select the external turn strategies."""
    service = _realtime_service(
        vad_force_turn_endpoint=False,
        should_interrupt=False,
    )

    metadata = service.service_metadata_frame()

    assert isinstance(metadata.user_turn_strategies, ExternalUserTurnStrategies)
    assert metadata.user_turn_strategies.enable_interruptions is False
    assert metadata.ttfs_p99_latency == 0.0


@pytest.mark.asyncio
async def test_inworld_realtime_stt_proposes_inworld_turn_boundaries(monkeypatch):
    """Inworld turn mode should place the final transcript before the turn stop."""
    events = []
    service = _instrument_realtime_service(
        monkeypatch,
        events,
        vad_force_turn_endpoint=False,
    )

    await service._process_response({"result": {"speechStarted": {}}})
    await service._process_response({"result": {"speechStopped": {"silenceDurationMs": 750}}})
    await service._process_response(
        {
            "result": {
                "transcription": {
                    "transcript": "Hello.",
                    "isFinal": True,
                }
            }
        }
    )

    assert [(event[0], event[1]) for event in events] == [
        ("broadcast", ProposedUserStartedSpeakingFrame),
        ("push", TranscriptionFrame),
        ("broadcast", ProposedUserStoppedSpeakingFrame),
    ]
    assert service._user_turn_open is False


@pytest.mark.asyncio
async def test_inworld_realtime_stt_server_mode_ignores_vad_stop(monkeypatch):
    """Local VAD pauses should not end a turn owned by Inworld."""
    events = []
    service = _instrument_realtime_service(
        monkeypatch,
        events,
        vad_force_turn_endpoint=False,
    )
    websocket = _FakeWebsocket()
    service._websocket = websocket

    await service.process_frame(
        VADUserStartedSpeakingFrame(),
        FrameDirection.DOWNSTREAM,
    )
    await service.process_frame(
        VADUserStoppedSpeakingFrame(),
        FrameDirection.DOWNSTREAM,
    )

    websocket.send.assert_not_awaited()
    assert [(event[0], event[1]) for event in events] == [
        ("push", VADUserStartedSpeakingFrame),
        ("broadcast", ProposedUserStartedSpeakingFrame),
        ("push", VADUserStoppedSpeakingFrame),
    ]


@pytest.mark.asyncio
async def test_inworld_realtime_stt_disconnects_with_close_stream():
    """Graceful disconnect should send the required closeStream message."""
    service = _realtime_service()
    websocket = _FakeWebsocket()
    service._websocket = websocket

    await service._disconnect_websocket()

    assert json.loads(websocket.send.await_args.args[0]) == {"closeStream": {}}
    websocket.close.assert_awaited_once()
    assert service._websocket is None


@pytest.mark.asyncio
async def test_inworld_realtime_stt_settings_update_requests_reconnect(monkeypatch):
    """Runtime settings should reconnect because config is connection-scoped."""
    service = _realtime_service()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    await service._update_settings(
        InworldRealtimeSTTService.Settings(prompts=["Inworld", "Pipecat"])
    )

    assert service._settings.prompts == ["Inworld", "Pipecat"]
    reconnect.assert_awaited_once()
