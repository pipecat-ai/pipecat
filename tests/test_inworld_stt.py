#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Inworld speech-to-text service."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock

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
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_start import (
    TranscriptionUserTurnStartStrategy,
    VADUserTurnStartStrategy,
)
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
async def test_inworld_realtime_stt_restarts_completed_receive_task(monkeypatch):
    """A reopened or active socket should replace a completed receive task."""
    service = _realtime_service()
    websocket = _FakeWebsocket()
    service._websocket = websocket

    completed_task = MagicMock()
    completed_task.done.return_value = True
    service._receive_task = completed_task

    replacement_task = MagicMock()
    replacement_task.done.return_value = False

    def fake_create_task(coroutine, *, name):
        coroutine.close()
        assert name == "inworld_stt_receive"
        return replacement_task

    monkeypatch.setattr(service, "create_task", fake_create_task)

    frames = [frame async for frame in service.run_stt(b"\x01\x02")]

    assert frames == [None]
    assert service._receive_task is replacement_task
    websocket.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_inworld_realtime_stt_blocks_senders_during_socket_recovery(monkeypatch):
    """Recovery should keep senders and concurrent reconnect callers on one socket."""
    service = _realtime_service()
    recovery_started = asyncio.Event()
    release_recovery = asyncio.Event()
    recovered_websocket = _FakeWebsocket()

    async def fake_reconnect_websocket(self, attempt_number):
        assert self is service
        assert attempt_number == 1
        recovery_started.set()
        await release_recovery.wait()
        service._websocket = recovered_websocket
        return True

    monkeypatch.setattr(
        WebsocketSTTService,
        "_reconnect_websocket",
        fake_reconnect_websocket,
    )

    recovery = asyncio.create_task(service._try_reconnect(max_retries=1))
    await recovery_started.wait()
    assert service._connected_event.is_set() is False

    concurrent_recovery = asyncio.create_task(service._try_reconnect(max_retries=1))

    async def send_audio():
        return [frame async for frame in service.run_stt(b"\x01\x02")]

    send = asyncio.create_task(send_audio())
    await asyncio.sleep(0)
    assert concurrent_recovery.done() is False
    assert send.done() is False

    release_recovery.set()

    assert await recovery is True
    assert await concurrent_recovery is True
    assert await send == [None]
    recovered_websocket.send.assert_awaited_once()


def test_inworld_realtime_stt_automatic_mode_requires_server_vad():
    """Automatic mode should direct callers to manual mode instead of accepting VAD zero."""
    service = _realtime_service(
        settings=InworldRealtimeSTTService.Settings(vad_threshold=0),
    )

    with pytest.raises(ValueError, match="TurnDetectionMode.MANUAL"):
        service._transcribe_config()


def test_inworld_realtime_stt_manual_mode_disables_server_vad():
    """Manual mode should send the documented VAD-zero configuration."""
    service = _realtime_service(
        turn_detection_mode=InworldRealtimeSTTService.TurnDetectionMode.MANUAL,
    )

    config = service._transcribe_config()

    assert config["inworldSttV1Config"] == {"vadThreshold": 0}


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
    """Streaming results should emit the latest Voice Profile once with the final result."""
    events = []
    service = _instrument_realtime_service(monkeypatch, events)

    await service._process_response(
        {
            "result": {
                "transcription": {
                    "transcript": "Hello",
                    "isFinal": False,
                    "voiceProfile": {
                        "emotion": [{"label": "neutral", "confidence": 0.72}],
                    },
                }
            }
        }
    )
    final_result = {
        "transcription": {
            "transcript": "Hello from Inworld.",
            "isFinal": True,
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
    assert pushed[1].voice_profile.emotion[0].label == "neutral"
    assert pushed[2].text == "Hello from Inworld."
    assert pushed[2].finalized is True
    assert pushed[2].result == final_result
    service._handle_transcription.assert_awaited_once_with("Hello from Inworld.", True, None)


@pytest.mark.asyncio
async def test_inworld_realtime_stt_reports_response_processing_errors(monkeypatch):
    """Unexpected response failures should reach the pipeline as non-fatal errors."""
    service = _realtime_service()
    service._websocket = _FakeWebsocket(messages=[json.dumps({"result": {}})])
    failure = ValueError("invalid response")
    monkeypatch.setattr(service, "_process_response", AsyncMock(side_effect=failure))
    push_error = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error)

    await service._receive_messages()

    push_error.assert_awaited_once_with(
        error_msg="Error processing Inworld realtime STT message: invalid response",
        exception=failure,
    )


@pytest.mark.asyncio
async def test_inworld_realtime_stt_reports_documented_error_envelope(monkeypatch):
    """Provider error envelopes should emit an error and reset manual turn state."""
    service = _realtime_service(
        turn_detection_mode=InworldRealtimeSTTService.TurnDetectionMode.MANUAL,
    )
    service._manual_vad_turn_open = True
    service._manual_pending_finals = 1
    service._pending_voice_profile = InworldVoiceProfile()
    push_error = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error)

    await service._process_response(
        {"error": {"code": 8, "message": "rate limit exceeded", "details": []}}
    )

    push_error.assert_awaited_once_with(
        error_msg="Inworld realtime STT error (8): rate limit exceeded"
    )
    assert service._manual_vad_turn_open is False
    assert service._manual_pending_finals == 0
    assert service._pending_voice_profile is None


def test_inworld_realtime_stt_recommends_external_turn_strategies():
    """Inworld-owned endpointing should select the external turn strategies."""
    service = _realtime_service(
        should_interrupt=False,
    )

    metadata = service.service_metadata_frame()

    assert isinstance(metadata.user_turn_strategies, ExternalUserTurnStrategies)
    assert metadata.user_turn_strategies.enable_interruptions is False
    assert metadata.ttfs_p99_latency == 0.0


def test_inworld_realtime_stt_manual_mode_uses_pipecat_turn_strategies():
    """Manual endpointing should accept only VAD-driven turn starts."""
    service = _realtime_service(
        turn_detection_mode=InworldRealtimeSTTService.TurnDetectionMode.MANUAL,
        ttfs_p99_latency=0.4,
    )

    metadata = service.service_metadata_frame()

    assert metadata.user_turn_strategies is not None
    assert len(metadata.user_turn_strategies.start) == 1
    assert isinstance(metadata.user_turn_strategies.start[0], VADUserTurnStartStrategy)
    assert not any(
        isinstance(strategy, TranscriptionUserTurnStartStrategy)
        for strategy in metadata.user_turn_strategies.start
    )
    assert metadata.ttfs_p99_latency == 0.4


@pytest.mark.asyncio
async def test_inworld_realtime_stt_manual_mode_ignores_results_outside_vad_turn(monkeypatch):
    """Provider results should not create turns unless local VAD admitted the audio."""
    events = []
    service = _instrument_realtime_service(
        monkeypatch,
        events,
        turn_detection_mode=InworldRealtimeSTTService.TurnDetectionMode.MANUAL,
    )

    response = {
        "result": {
            "transcription": {
                "transcript": "assistant echo",
                "isFinal": True,
                "voiceProfile": {
                    "emotion": [{"label": "neutral", "confidence": 0.8}],
                },
            }
        }
    }
    await service._process_response(response)

    assert events == []
    service._handle_transcription.assert_not_awaited()


@pytest.mark.asyncio
async def test_inworld_realtime_stt_manual_mode_ignores_unmatched_vad_stop(monkeypatch):
    """A VAD stop without an admitted start should not finalize background audio."""
    events = []
    service = _instrument_realtime_service(
        monkeypatch,
        events,
        turn_detection_mode=InworldRealtimeSTTService.TurnDetectionMode.MANUAL,
    )
    websocket = _FakeWebsocket()
    service._websocket = websocket

    await service.process_frame(
        VADUserStoppedSpeakingFrame(),
        FrameDirection.DOWNSTREAM,
    )

    websocket.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_inworld_realtime_stt_proposes_inworld_turn_boundaries(monkeypatch):
    """Inworld should place the final transcript before the turn-stop proposal."""
    events = []
    service = _instrument_realtime_service(
        monkeypatch,
        events,
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
async def test_inworld_realtime_stt_ignores_local_vad_boundaries(monkeypatch):
    """Local VAD frames should not open or end a turn owned by Inworld."""
    events = []
    service = _instrument_realtime_service(
        monkeypatch,
        events,
    )
    websocket = _FakeWebsocket()
    service._websocket = websocket
    start_ttfb_metrics = AsyncMock()
    monkeypatch.setattr(service, "start_ttfb_metrics", start_ttfb_metrics)

    await service.process_frame(
        VADUserStartedSpeakingFrame(),
        FrameDirection.DOWNSTREAM,
    )
    await service.process_frame(
        VADUserStoppedSpeakingFrame(stop_secs=0.5),
        FrameDirection.DOWNSTREAM,
    )

    websocket.send.assert_not_awaited()
    start_ttfb_metrics.assert_not_awaited()
    assert [(event[0], event[1]) for event in events] == [
        ("push", VADUserStartedSpeakingFrame),
        ("push", VADUserStoppedSpeakingFrame),
    ]


@pytest.mark.asyncio
async def test_inworld_realtime_stt_manual_mode_sends_end_turn(monkeypatch):
    """A Pipecat VAD stop should request finalization without external proposals."""
    events = []
    service = _instrument_realtime_service(
        monkeypatch,
        events,
        turn_detection_mode=InworldRealtimeSTTService.TurnDetectionMode.MANUAL,
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
    await service._process_response({"result": {"speechStarted": {}}})
    await service._process_response(
        {
            "result": {
                "transcription": {
                    "transcript": "One complete turn.",
                    "isFinal": True,
                }
            }
        }
    )

    assert json.loads(websocket.send.await_args.args[0]) == {"endTurn": {}}
    assert [(event[0], event[1]) for event in events] == [
        ("push", VADUserStartedSpeakingFrame),
        ("push", VADUserStoppedSpeakingFrame),
        ("push", TranscriptionFrame),
    ]
    assert service._user_turn_open is False


@pytest.mark.asyncio
async def test_inworld_realtime_stt_manual_mode_keeps_next_vad_turn_open(monkeypatch):
    """A previous final should not close a newer local VAD turn."""
    events = []
    service = _instrument_realtime_service(
        monkeypatch,
        events,
        turn_detection_mode=InworldRealtimeSTTService.TurnDetectionMode.MANUAL,
    )
    websocket = _FakeWebsocket()
    service._websocket = websocket

    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    await service._process_response(
        {"result": {"transcription": {"transcript": "First turn.", "isFinal": True}}}
    )
    await service._process_response(
        {"result": {"transcription": {"transcript": "Second", "isFinal": False}}}
    )
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service._process_response(
        {"result": {"transcription": {"transcript": "Second turn.", "isFinal": True}}}
    )

    sent_messages = [json.loads(call.args[0]) for call in websocket.send.await_args_list]
    assert sent_messages == [{"endTurn": {}}, {"endTurn": {}}]
    transcriptions = [
        event[2]
        for event in events
        if event[0] == "push"
        and isinstance(event[2], (InterimTranscriptionFrame, TranscriptionFrame))
    ]
    assert [frame.text for frame in transcriptions] == [
        "First turn.",
        "Second",
        "Second turn.",
    ]
    assert service._manual_vad_turn_open is False
    assert service._manual_pending_finals == 0


@pytest.mark.asyncio
async def test_inworld_realtime_stt_manual_mode_preserves_vad_turn_across_retry(monkeypatch):
    """A recovered audio send should still be finalized by the next local VAD stop."""
    events = []
    service = _instrument_realtime_service(
        monkeypatch,
        events,
        turn_detection_mode=InworldRealtimeSTTService.TurnDetectionMode.MANUAL,
    )
    failed_websocket = _FakeWebsocket()
    failed_websocket.send.side_effect = [ConnectionError("socket lost"), None]
    recovered_websocket = _FakeWebsocket()
    service._websocket = failed_websocket

    async def fake_connect_websocket():
        service._websocket = recovered_websocket

    monkeypatch.setattr(service, "_connect_websocket", fake_connect_websocket)
    monkeypatch.setattr(service, "_verify_connection", AsyncMock(return_value=True))

    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    frames = [frame async for frame in service.run_stt(b"\x01\x02")]

    assert frames == [None]
    assert service._manual_vad_turn_open is True

    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    sent_messages = [json.loads(call.args[0]) for call in recovered_websocket.send.await_args_list]
    assert sent_messages == [
        {"audioChunk": {"content": base64.b64encode(b"\x01\x02").decode("ascii")}},
        {"endTurn": {}},
    ]


@pytest.mark.asyncio
async def test_inworld_realtime_stt_manual_mode_accepts_final_after_end_turn_retry(monkeypatch):
    """A retried endTurn should keep exactly one final response pending."""
    events = []
    service = _instrument_realtime_service(
        monkeypatch,
        events,
        turn_detection_mode=InworldRealtimeSTTService.TurnDetectionMode.MANUAL,
    )
    failed_websocket = _FakeWebsocket()
    failed_websocket.send.side_effect = [ConnectionError("socket lost"), None]
    recovered_websocket = _FakeWebsocket()
    service._websocket = failed_websocket

    async def fake_connect_websocket():
        service._websocket = recovered_websocket

    monkeypatch.setattr(service, "_connect_websocket", fake_connect_websocket)
    monkeypatch.setattr(service, "_verify_connection", AsyncMock(return_value=True))

    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert service._manual_pending_finals == 1
    assert [json.loads(call.args[0]) for call in recovered_websocket.send.await_args_list] == [
        {"endTurn": {}}
    ]

    await service._process_response(
        {"result": {"transcription": {"transcript": "Recovered final.", "isFinal": True}}}
    )

    transcriptions = [
        event[2]
        for event in events
        if event[0] == "push" and isinstance(event[2], TranscriptionFrame)
    ]
    assert [frame.text for frame in transcriptions] == ["Recovered final."]
    assert service._manual_pending_finals == 0


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
async def test_inworld_realtime_stt_full_disconnect_resets_turn_state(monkeypatch):
    """Intentional teardown should clear local and provider-owned turn state."""
    events = []
    service = _instrument_realtime_service(monkeypatch, events)
    websocket = _FakeWebsocket()
    service._websocket = websocket
    service._user_turn_open = True
    service._manual_vad_turn_open = True
    service._manual_pending_finals = 1
    service._need_reconnect = True

    await service._disconnect()

    assert service._user_turn_open is False
    assert service._manual_vad_turn_open is False
    assert service._manual_pending_finals == 0
    assert service._can_reconnect is True
    assert service._need_reconnect is False
    assert ("broadcast", ProposedUserStoppedSpeakingFrame, None) in events


@pytest.mark.asyncio
async def test_inworld_realtime_stt_recovery_aborts_during_full_disconnect(monkeypatch):
    """A queued error recovery should not reopen the socket after teardown starts."""
    service = _realtime_service()
    service._websocket = _FakeWebsocket()
    connect_websocket = AsyncMock()
    monkeypatch.setattr(service, "_connect_websocket", connect_websocket)
    monkeypatch.setattr(
        "pipecat.services.websocket_service.exponential_backoff_time",
        lambda attempt: 0,
    )

    await service._connection_lock.acquire()
    recovery = asyncio.create_task(service._try_reconnect(max_retries=1))
    while not service._reconnect_in_progress:
        await asyncio.sleep(0)

    disconnect = asyncio.create_task(service._disconnect())
    while not service._disconnecting:
        await asyncio.sleep(0)

    service._connection_lock.release()

    assert await recovery is False
    await disconnect
    connect_websocket.assert_not_awaited()
    assert service._websocket is None
    assert service._receive_task is None


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


@pytest.mark.asyncio
async def test_inworld_realtime_stt_defers_settings_until_inworld_final(monkeypatch):
    """Provider-owned turns should defer settings reconnects until their final result."""
    events = []
    service = _instrument_realtime_service(monkeypatch, events)
    do_reconnect = AsyncMock()
    monkeypatch.setattr(service, "_do_reconnect", do_reconnect)

    await service._process_response({"result": {"speechStarted": {}}})
    assert service._can_reconnect is False

    await service._update_settings(InworldRealtimeSTTService.Settings(prompts=["new hint"]))
    assert service._need_reconnect is True
    do_reconnect.assert_not_awaited()

    await service._process_response(
        {"result": {"transcription": {"transcript": "Complete.", "isFinal": True}}}
    )

    do_reconnect.assert_awaited_once()
    assert service._can_reconnect is True
    assert service._need_reconnect is False


@pytest.mark.asyncio
async def test_inworld_realtime_stt_empty_final_releases_vad_deferred_settings(monkeypatch):
    """An empty provider final should release reconnects deferred by local VAD."""
    events = []
    service = _instrument_realtime_service(monkeypatch, events)
    do_reconnect = AsyncMock()
    monkeypatch.setattr(service, "_do_reconnect", do_reconnect)

    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service._update_settings(InworldRealtimeSTTService.Settings(prompts=["new hint"]))
    await service._process_response(
        {"result": {"transcription": {"transcript": "", "isFinal": True}}}
    )

    do_reconnect.assert_awaited_once()
    assert service._can_reconnect is True
    assert service._need_reconnect is False


@pytest.mark.asyncio
async def test_inworld_realtime_stt_receive_task_schedules_deferred_reconnect(monkeypatch):
    """A provider final should not make the receive task disconnect itself."""
    events = []
    service = _instrument_realtime_service(monkeypatch, events)
    service._receive_task = asyncio.current_task()
    service._user_turn_open = True
    service._can_reconnect = False
    service._need_reconnect = True
    scheduled = []

    def fake_create_task(coroutine, *, name):
        scheduled.append((coroutine, name))
        return MagicMock()

    monkeypatch.setattr(service, "create_task", fake_create_task)

    await service._user_turn_stopped()

    assert service._can_reconnect is True
    assert service._need_reconnect is False
    assert len(scheduled) == 1
    assert scheduled[0][1] == "inworld_stt_settings_reconnect"
    scheduled[0][0].close()
