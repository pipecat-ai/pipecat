#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Protocol tests for Sarvam STT and TTS services hosted on Amazon SageMaker."""

import asyncio
import base64
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
    ErrorFrame,
    InterruptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402
from pipecat.services.sarvam.sagemaker.stt import (  # noqa: E402
    SarvamSageMakerSTTService,
)
from pipecat.services.sarvam.sagemaker.tts import (  # noqa: E402
    SarvamSageMakerTTSService,
)
from pipecat.services.tts_service import InterruptibleTTSService, TTSService  # noqa: E402
from pipecat.transcriptions.language import Language  # noqa: E402

PCM_AUDIO = b"\x00\x01\x02\x03" * 400
TTS_AUDIO = b"\x10\x11" * 512


class FakeBidiClient:
    """In-memory SageMaker BiDi transport that records UTF8 JSON payloads."""

    instances: list["FakeBidiClient"] = []

    def __init__(
        self,
        *,
        endpoint_name,
        region,
        model_invocation_path=None,
        model_query_string=None,
        **_kwargs,
    ):
        self.endpoint_name = endpoint_name
        self.region = region
        self.model_invocation_path = model_invocation_path
        self.model_query_string = model_query_string
        self.is_active = False
        self.sent_payloads: list[tuple[bytes, str | None]] = []
        self._responses: asyncio.Queue = asyncio.Queue()
        FakeBidiClient.instances.append(self)

    @property
    def messages(self) -> list[dict]:
        return [json.loads(payload.decode("utf-8")) for payload, _ in self.sent_payloads]

    async def start_session(self):
        self.is_active = True

    async def close_session(self):
        self.is_active = False

    async def send_data(self, payload: bytes, data_type: str | None = None):
        if not self.is_active:
            raise RuntimeError("BiDi session not active")
        self.sent_payloads.append((payload, data_type))

    async def send_json(self, message: dict):
        await self.send_data(json.dumps(message).encode("utf-8"), data_type="UTF8")

    async def receive_response(self):
        return await self._responses.get()

    async def queue_json(self, message: dict):
        await self._responses.put(
            ResponseStreamEventPayloadPart(
                value=ResponsePayloadPart(
                    bytes_=json.dumps(message).encode("utf-8"),
                    data_type="UTF8",
                )
            )
        )

    async def queue_raw_utf8(self, payload: bytes):
        await self._responses.put(
            ResponseStreamEventPayloadPart(
                value=ResponsePayloadPart(bytes_=payload, data_type="UTF8")
            )
        )

    async def end_responses(self):
        await self._responses.put(None)


@pytest.fixture(autouse=True)
def reset_fake_clients():
    FakeBidiClient.instances = []


def _stt_service(**kwargs) -> SarvamSageMakerSTTService:
    return SarvamSageMakerSTTService(
        endpoint_name="sarvam-saaras",
        region="ap-south-1",
        sample_rate=16000,
        input_audio_codec="wav",
        settings=SarvamSageMakerSTTService.Settings(
            model="saaras:v3",
            language=Language.EN_IN,
            vad_signals=False,
        ),
        **kwargs,
    )


def _tts_service(**kwargs) -> SarvamSageMakerTTSService:
    return SarvamSageMakerTTSService(
        endpoint_name="sarvam-bulbul",
        region="ap-south-1",
        sample_rate=24000,
        settings=SarvamSageMakerTTSService.Settings(
            model="bulbul:v3",
            voice="shubh",
            language=Language.EN_IN,
            pace=1.1,
        ),
        **kwargs,
    )


def _active_client() -> FakeBidiClient:
    client = FakeBidiClient(
        endpoint_name="endpoint",
        region="ap-south-1",
        model_invocation_path="invocations-bidirectional-stream",
        model_query_string="",
    )
    client.is_active = True
    return client


async def _consume(generator):
    return [frame async for frame in generator]


def _capture_frames(frames):
    async def capture(frame, *_args, **_kwargs):
        frames.append(frame)

    return capture


def _capture_frame_classes(frame_classes):
    async def capture(frame_class, *_args, **_kwargs):
        frame_classes.append(frame_class)

    return capture


def test_stt_query_parameters_are_url_encoded():
    service = SarvamSageMakerSTTService(
        endpoint_name="sarvam-saaras",
        region="ap-south-1",
        sample_rate=16000,
        input_audio_codec="wav",
        settings=SarvamSageMakerSTTService.Settings(
            model="saaras:v3",
            language=Language.HI_IN,
            vad_signals=True,
        ),
    )

    query_string = service._build_query_string()
    query = parse_qs(query_string)

    assert "saaras%3Av3" in query_string
    assert query["model"] == ["saaras:v3"]
    assert query["sample_rate"] == ["16000"]
    assert query["language-code"] == ["hi-IN"]
    assert query["vad_signals"] == ["true"]


def test_tts_query_parameters_are_url_encoded():
    service = _tts_service()

    query_string = service._build_query_string()
    query = parse_qs(query_string)

    assert "bulbul%3Av3" in query_string
    assert query["model"] == ["bulbul:v3"]
    assert set(query) == {"model"}


def test_stt_rejects_unsupported_streaming_sample_rate():
    with pytest.raises(ValueError, match="8000 or 16000"):
        SarvamSageMakerSTTService(
            endpoint_name="sarvam-saaras",
            region="ap-south-1",
            sample_rate=24000,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_rate": 48000}, "8000, 16000, 22050, or 24000"),
        ({"output_audio_codec": "flac"}, "does not support FLAC"),
    ],
)
def test_tts_rejects_unsupported_streaming_audio_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SarvamSageMakerTTSService(
            endpoint_name="sarvam-bulbul",
            region="ap-south-1",
            **kwargs,
        )


@pytest.mark.asyncio
async def test_services_use_default_model_invocation_path(monkeypatch):
    with (
        patch(
            "pipecat.services.sarvam.sagemaker.stt.SageMakerBidiClient",
            new=FakeBidiClient,
        ),
        patch(
            "pipecat.services.sarvam.sagemaker.tts.SageMakerBidiClient",
            new=FakeBidiClient,
        ),
    ):
        stt = _stt_service()
        tts = _tts_service()

        def discard_task(coroutine):
            coroutine.close()
            return None

        monkeypatch.setattr(stt, "create_task", discard_task)
        monkeypatch.setattr(tts, "create_task", discard_task)
        await stt._connect()
        await tts._connect()

        assert FakeBidiClient.instances[0].model_invocation_path in (None, "")
        assert FakeBidiClient.instances[1].model_invocation_path in (None, "")

        await stt._disconnect()
        await tts._disconnect()


@pytest.mark.asyncio
async def test_stt_sends_utf8_config_audio_and_flush_messages():
    service = _stt_service()
    client = _active_client()
    service._client = client

    await service._send_config()
    frames = await _consume(service.run_stt(PCM_AUDIO))
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert frames == [None]
    assert all(data_type == "UTF8" for _, data_type in client.sent_payloads)
    assert [message.get("type", "audio") for message in client.messages] == [
        "config",
        "audio",
        "flush",
    ]
    assert client.messages[0] == {"type": "config"}

    audio = client.messages[1]["audio"]
    assert audio == {
        "data": base64.b64encode(PCM_AUDIO).decode("utf-8"),
        "encoding": "audio/wav",
        "sample_rate": 16000,
    }
    assert client.messages[2] == {"type": "flush"}


@pytest.mark.asyncio
async def test_stt_maps_provider_events_and_transcript_to_pipecat_frames(monkeypatch):
    service = _stt_service()
    client = _active_client()
    service._client = client
    pushed = []
    broadcasted = []
    monkeypatch.setattr(service, "push_frame", _capture_frames(pushed))
    monkeypatch.setattr(service, "broadcast_frame", _capture_frame_classes(broadcasted))
    monkeypatch.setattr(service, "emit_stt_usage_metrics", AsyncMock())
    monkeypatch.setattr(service, "_handle_transcription", AsyncMock())

    await client.queue_json(
        {
            "type": "events",
            "data": {"signal_type": "START_SPEECH", "occured_at": "2026-10-13T00:00:00Z"},
        }
    )
    await client.queue_json(
        {
            "type": "events",
            "data": {"signal_type": "END_SPEECH", "occured_at": "2026-10-13T00:00:01Z"},
        }
    )
    transcript_message = {
        "type": "data",
        "data": {"transcript": "Hello from Saaras.", "language_code": "en-IN"},
    }
    await client.queue_json(transcript_message)
    await client.end_responses()

    await service._process_responses()

    assert broadcasted == [
        ProposedUserStartedSpeakingFrame,
        ProposedUserStoppedSpeakingFrame,
    ]
    assert len(pushed) == 1
    assert isinstance(pushed[0], TranscriptionFrame)
    assert pushed[0].text == "Hello from Saaras."
    assert pushed[0].language == Language.EN_IN
    assert pushed[0].finalized is True
    assert pushed[0].result == transcript_message


@pytest.mark.asyncio
async def test_stt_send_failure_yields_error_frame():
    service = _stt_service()
    client = _active_client()
    client.send_json = AsyncMock(side_effect=RuntimeError("write failed"))
    service._client = client

    frames = await _consume(service.run_stt(PCM_AUDIO))

    assert len(frames) == 2
    assert isinstance(frames[0], ErrorFrame)
    assert "write failed" in frames[0].error
    assert frames[1] is None


@pytest.mark.asyncio
async def test_stt_provider_error_and_invalid_json_are_reported(monkeypatch):
    service = _stt_service()
    client = _active_client()
    service._client = client
    push_error = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error)

    await client.queue_raw_utf8(b"{not-json")
    await client.queue_json(
        {
            "type": "error",
            "data": {"code": "invalid_audio", "message": "PCM payload was rejected"},
        }
    )
    await client.end_responses()

    await service._process_responses()

    assert push_error.await_count == 2
    error_messages = [call.kwargs["error_msg"] for call in push_error.await_args_list]
    assert any("Invalid Sarvam STT response" in message for message in error_messages)
    assert any("PCM payload was rejected" in message for message in error_messages)


@pytest.mark.asyncio
async def test_stt_invalid_response_structures_do_not_stop_receive_loop(monkeypatch):
    service = _stt_service()
    client = _active_client()
    service._client = client
    push_error = AsyncMock()
    pushed = []
    monkeypatch.setattr(service, "push_error", push_error)
    monkeypatch.setattr(service, "push_frame", _capture_frames(pushed))
    monkeypatch.setattr(service, "emit_stt_usage_metrics", AsyncMock())
    monkeypatch.setattr(service, "_handle_transcription", AsyncMock())

    await client.queue_json(["not", "an", "object"])
    await client.queue_json({"type": "data", "data": "not an object"})
    await client.queue_json(
        {
            "type": "data",
            "data": {"transcript": "Still listening.", "language_code": "en-IN"},
        }
    )
    await client.end_responses()

    await service._process_responses()

    assert push_error.await_count == 2
    assert len(pushed) == 1
    assert isinstance(pushed[0], TranscriptionFrame)
    assert pushed[0].text == "Still listening."


@pytest.mark.asyncio
async def test_stt_disconnect_closes_session_and_is_idempotent():
    service = _stt_service()
    client = _active_client()
    client.close_session = AsyncMock(wraps=client.close_session)
    service._client = client

    await service._disconnect()
    await service._disconnect()

    client.close_session.assert_awaited_once()
    assert service._client is None


@pytest.mark.asyncio
async def test_stt_connection_setting_update_reconnects(monkeypatch):
    service = _stt_service()
    disconnect = AsyncMock()
    connect = AsyncMock()
    monkeypatch.setattr(service, "_disconnect", disconnect)
    monkeypatch.setattr(service, "_connect", connect)

    await service._update_settings(SarvamSageMakerSTTService.Settings(language=Language.HI_IN))

    disconnect.assert_awaited_once()
    connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_sends_utf8_config_text_and_flush_messages():
    service = _tts_service()
    client = _active_client()
    service._client = client

    await service._send_config()
    frames = await _consume(service.run_tts("Namaste from Bulbul.", "ctx-1"))
    await service.flush_audio("ctx-1")

    assert frames == [None]
    assert all(data_type == "UTF8" for _, data_type in client.sent_payloads)
    assert [message["type"] for message in client.messages] == ["config", "text", "flush"]

    config = client.messages[0]["data"]
    assert config["language_code"] == "en-IN"
    assert config["speaker"] == "shubh"
    assert config["speech_sample_rate"] == "24000"
    assert config["output_audio_codec"] == "linear16"
    assert config["pace"] == 1.1
    assert config["model"] == "bulbul:v3"
    assert "min_buffer_size" not in config
    assert "max_chunk_length" not in config

    assert client.messages[1] == {
        "type": "text",
        "data": {"text": "Namaste from Bulbul."},
    }
    assert client.messages[2] == {"type": "flush"}


@pytest.mark.asyncio
async def test_tts_maps_audio_and_final_responses_to_audio_context(monkeypatch):
    service = _tts_service()
    client = _active_client()
    service._client = client
    appended = []
    removed = []
    monkeypatch.setattr(service, "get_active_audio_context_id", lambda: "ctx-1")
    monkeypatch.setattr(service, "audio_context_available", lambda _context_id: True)

    async def append(context_id, frame):
        appended.append((context_id, frame))

    async def remove(context_id):
        removed.append(context_id)

    monkeypatch.setattr(service, "append_to_audio_context", append)
    monkeypatch.setattr(service, "remove_audio_context", remove)
    monkeypatch.setattr(service, "stop_ttfb_metrics", AsyncMock())

    await client.queue_json(
        {
            "type": "audio",
            "data": {
                "audio": base64.b64encode(TTS_AUDIO).decode("utf-8"),
                "request_id": "request-123",
            },
        }
    )
    await client.queue_json(
        {
            "type": "event",
            "data": {"event_type": "final", "request_id": "request-123"},
        }
    )
    await client.end_responses()

    await service._process_responses()

    assert len(appended) == 2
    assert appended[0][0] == "ctx-1"
    assert isinstance(appended[0][1], TTSAudioRawFrame)
    assert appended[0][1].audio == TTS_AUDIO
    assert appended[0][1].sample_rate == 24000
    assert appended[0][1].num_channels == 1
    assert appended[0][1].context_id == "ctx-1"
    assert isinstance(appended[1][1], TTSStoppedFrame)
    assert appended[1][1].context_id == "ctx-1"
    assert removed == ["ctx-1"]


@pytest.mark.asyncio
async def test_tts_interruption_uses_interruptible_reconnect_contract(monkeypatch):
    service = _tts_service()
    assert isinstance(service, InterruptibleTTSService)
    service._bot_speaking = True
    disconnect = AsyncMock()
    connect = AsyncMock()
    monkeypatch.setattr(service, "_disconnect", disconnect)
    monkeypatch.setattr(service, "_connect", connect)

    with patch.object(TTSService, "_handle_interruption", new=AsyncMock()):
        await service._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    disconnect.assert_awaited_once()
    connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_disconnect_sends_close_before_closing_session():
    service = _tts_service()
    client = _active_client()
    client.close_session = AsyncMock(wraps=client.close_session)
    service._client = client

    await service._disconnect()

    assert client.messages == []
    client.close_session.assert_awaited_once()
    assert service._client is None


@pytest.mark.asyncio
async def test_tts_config_update_is_sent_without_reconnecting(monkeypatch):
    service = _tts_service()
    client = _active_client()
    service._client = client
    disconnect = AsyncMock()
    connect = AsyncMock()
    monkeypatch.setattr(service, "_disconnect", disconnect)
    monkeypatch.setattr(service, "_connect", connect)

    await service._update_settings(SarvamSageMakerTTSService.Settings(voice="priya"))

    disconnect.assert_not_awaited()
    connect.assert_not_awaited()
    assert client.messages[-1]["type"] == "config"
    assert client.messages[-1]["data"]["speaker"] == "priya"


@pytest.mark.asyncio
async def test_tts_send_failure_yields_error_and_reconnects(monkeypatch):
    service = _tts_service()
    client = _active_client()
    client.send_json = AsyncMock(side_effect=RuntimeError("write failed"))
    service._client = client
    disconnect = AsyncMock()
    connect = AsyncMock()
    monkeypatch.setattr(service, "_disconnect", disconnect)
    monkeypatch.setattr(service, "_connect", connect)

    frames = await _consume(service.run_tts("Hello.", "ctx-1"))

    assert isinstance(frames[0], ErrorFrame)
    assert "write failed" in frames[0].error
    assert isinstance(frames[1], TTSStoppedFrame)
    disconnect.assert_awaited_once()
    connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_tts_provider_error_is_reported_to_service_and_audio_context(monkeypatch):
    service = _tts_service()
    client = _active_client()
    service._client = client
    pushed_errors = []
    appended = []
    monkeypatch.setattr(service, "get_active_audio_context_id", lambda: "ctx-1")

    async def push_error(error_msg, **_kwargs):
        pushed_errors.append(error_msg)

    async def append(context_id, frame):
        appended.append((context_id, frame))

    monkeypatch.setattr(service, "push_error", push_error)
    monkeypatch.setattr(service, "append_to_audio_context", append)

    await client.queue_json(
        {
            "type": "error",
            "data": {"code": "synthesis_failed", "message": "voice is unavailable"},
        }
    )
    await client.end_responses()

    await service._process_responses()

    assert pushed_errors == ["TTS Error: voice is unavailable"]
    assert len(appended) == 1
    assert appended[0][0] == "ctx-1"
    assert isinstance(appended[0][1], ErrorFrame)
    assert appended[0][1].error == "TTS Error: voice is unavailable"


@pytest.mark.asyncio
async def test_tts_invalid_and_oversized_responses_do_not_stop_receive_loop(monkeypatch):
    service = _tts_service()
    client = _active_client()
    service._client = client
    push_error = AsyncMock()
    appended = []
    monkeypatch.setattr(service, "push_error", push_error)
    monkeypatch.setattr(service, "get_active_audio_context_id", lambda: "ctx-1")

    async def append(context_id, frame):
        appended.append((context_id, frame))

    monkeypatch.setattr(service, "append_to_audio_context", append)
    monkeypatch.setattr(service, "stop_ttfb_metrics", AsyncMock())

    await client.queue_json("not an object")
    await client.queue_raw_utf8(b"x" * (4 * 1024 * 1024 + 1))
    await client.queue_json(
        {
            "type": "audio",
            "data": {"audio": base64.b64encode(TTS_AUDIO).decode("utf-8")},
        }
    )
    await client.end_responses()

    await service._process_responses()

    assert push_error.await_count == 2
    assert len(appended) == 1
    assert isinstance(appended[0][1], TTSAudioRawFrame)
    assert appended[0][1].audio == TTS_AUDIO


@pytest.mark.asyncio
async def test_tts_concurrent_connects_create_one_session(monkeypatch):
    class SlowFakeBidiClient(FakeBidiClient):
        async def start_session(self):
            await asyncio.sleep(0.01)
            self.is_active = True

    with patch(
        "pipecat.services.sarvam.sagemaker.tts.SageMakerBidiClient",
        new=SlowFakeBidiClient,
    ):
        service = _tts_service()

        def discard_task(coroutine):
            coroutine.close()
            return None

        monkeypatch.setattr(service, "create_task", discard_task)
        await asyncio.gather(service._connect(), service._connect())

        assert len(FakeBidiClient.instances) == 1
        await service._disconnect()
