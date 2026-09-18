#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.protocol import State

from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.services.sarvam.tts import (
    TTS_MODEL_CONFIGS,
    WEBSOCKET_SAMPLE_RATES,
    SarvamHttpTTSService,
    SarvamTTSService,
    SarvamTTSSpeakerV3,
    SarvamTTSSpeakerV4Flash,
    _format_http_error,
    language_to_sarvam_language,
)
from pipecat.transcriptions.language import Language

V4_FLASH = "bulbul:v4-flash"


class _FakeWebsocket:
    def __init__(self, messages=None, *, state=State.OPEN):
        self._messages = messages or []
        self.state = state
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    async def close(self):
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


def _ws_service(**settings_kwargs) -> SarvamTTSService:
    settings_kwargs.setdefault("model", V4_FLASH)
    return SarvamTTSService(
        api_key="test-key", settings=SarvamTTSService.Settings(**settings_kwargs)
    )


def _sent_config(service: SarvamTTSService) -> dict:
    """Read back the config the service put on the wire."""
    messages = [json.loads(m) for m in service._websocket.sent]
    configs = [m["data"] for m in messages if m["type"] == "config"]
    assert configs, "no config message was sent"
    return configs[-1]


# --- Speaker catalogue ---------------------------------------------------------


def test_v4_flash_has_its_own_speaker_catalogue():
    """v4-flash rejects the v2 and v3 names, so it needs a separate catalogue."""
    v4_speakers = set(TTS_MODEL_CONFIGS[V4_FLASH].speakers)
    v3_speakers = {s.value for s in SarvamTTSSpeakerV3}

    assert len(v4_speakers) == 224
    assert not (v4_speakers & v3_speakers)
    assert SarvamTTSSpeakerV4Flash.SHUBH_EN_NARRATION_GENTLE in v4_speakers


def test_v4_flash_defaults_to_the_documented_speaker():
    """Omitting the voice has to land on the same default the API applies."""
    service = _ws_service()

    assert service._settings.voice == "shubh_en_narration_gentle"
    assert service._init_sample_rate == 24000


# --- Endpoint --------------------------------------------------------------------


@pytest.mark.parametrize("model", ["bulbul:v3-beta", "bulbul:v3", V4_FLASH])
def test_the_model_is_pinned_on_the_query_string(model):
    """The endpoint serves every model; the query string picks which one."""
    url = urlparse(_ws_service(model=model, voice=None)._websocket_url)

    assert url.path == "/text-to-speech/ws"
    assert parse_qs(url.query)["model"] == [model]


def test_an_unknown_model_is_rejected_with_the_allowed_values():
    with pytest.raises(ValueError, match=V4_FLASH):
        _ws_service(model="bulbul:v4")


# --- Sample rate ---------------------------------------------------------------


@pytest.mark.parametrize("sample_rate", WEBSOCKET_SAMPLE_RATES)
def test_streaming_sample_rates_are_accepted(sample_rate):
    service = SarvamTTSService(
        api_key="test-key",
        sample_rate=sample_rate,
        settings=SarvamTTSService.Settings(model=V4_FLASH),
    )

    assert service._init_sample_rate == sample_rate


@pytest.mark.parametrize("sample_rate", [32000, 44100, 48000])
def test_rates_the_websocket_rejects_fail_at_construction(sample_rate):
    """The socket only errors once synthesis starts, which is far too late."""
    with pytest.raises(ValueError, match="8000, 16000, 22050, 24000"):
        SarvamTTSService(
            api_key="test-key",
            sample_rate=sample_rate,
            settings=SarvamTTSService.Settings(model=V4_FLASH),
        )


# --- Voice parameters ----------------------------------------------------------


@pytest.mark.asyncio
async def test_v4_flash_sends_pitch_and_loudness():
    """v4-flash applies both as post-processing; the v3 models ignore them."""
    service = _ws_service(pitch=0.3, loudness=1.5, pace=1.2)
    service._websocket = _FakeWebsocket()

    await service._send_config()
    config = _sent_config(service)

    assert config["pitch"] == 0.3
    assert config["loudness"] == 1.5
    assert config["pace"] == 1.2


@pytest.mark.asyncio
async def test_v4_flash_drops_temperature():
    """The server pins it to 0.6, so sending a value would only mislead."""
    service = _ws_service(temperature=0.9)
    service._websocket = _FakeWebsocket()

    await service._send_config()

    assert "temperature" not in _sent_config(service)


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("pitch", 0.9, 0.5),
        ("pitch", -0.9, -0.5),
        ("loudness", 3.0, 2.5),
        ("loudness", 0.05, 0.1),
        ("pace", 2.5, 2.0),
        ("pace", 0.1, 0.5),
    ],
)
def test_voice_parameters_are_clamped_to_the_v4_flash_ranges(field, value, expected):
    service = _ws_service(**{field: value})

    assert getattr(service._settings, field) == expected


def test_v2_keeps_its_wider_pitch_range():
    """Clamping is per model, not a single range shared across the catalogue."""
    with pytest.deprecated_call():
        service = _ws_service(model="bulbul:v2", voice=None, pitch=0.7)

    assert service._settings.pitch == 0.7


def test_v4_flash_forces_preprocessing_on():
    service = _ws_service(enable_preprocessing=False)

    assert service._settings.enable_preprocessing is True


# --- Languages -----------------------------------------------------------------


@pytest.mark.parametrize(
    "language, expected",
    [
        (Language.EN_IN, "en-IN"),
        (Language.HI, "hi-IN"),
        (Language.AS_IN, "as-IN"),
        (Language.BRX, "brx-IN"),
        (Language.DOI_IN, "doi-IN"),
        (Language.KS, "ks-IN"),
        (Language.MNI_IN, "mni-IN"),
        (Language.NE, "ne-IN"),
        (Language.SA, "sa-IN"),
        (Language.SAT_IN, "sat-IN"),
        (Language.UR_IN, "ur-IN"),
        (Language.OR, "od-IN"),
    ],
)
def test_languages_map_to_sarvam_codes(language, expected):
    assert language_to_sarvam_language(language) == expected


# --- HTTP service --------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_payload_names_the_sample_rate_field_the_api_reads(monkeypatch):
    """The API reads `speech_sample_rate`; anything else leaves it at its default."""
    captured = {}

    class _Response:
        status = 200

        async def json(self):
            return {"audios": [""], "request_id": "test"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    def fake_post(url, json=None, headers=None):
        captured.update(json)
        return _Response()

    service = SarvamHttpTTSService(
        api_key="test-key",
        aiohttp_session=AsyncMock(post=fake_post),
        sample_rate=16000,
        settings=SarvamHttpTTSService.Settings(model=V4_FLASH),
    )
    service._sample_rate = 16000

    async for _ in service.run_tts("hello", "ctx"):
        pass

    assert captured["speech_sample_rate"] == 16000
    assert "sample_rate" not in captured


@pytest.mark.parametrize(
    "body, expected",
    [
        (
            json.dumps(
                {
                    "error": {
                        "message": "Speaker 'shubh' is not compatible with model bulbul:v4-flash.",
                        "code": "invalid_request_error",
                        "request_id": "20260918_abc",
                    }
                }
            ),
            "Sarvam TTS error (code=invalid_request_error, request_id=20260918_abc): "
            "Speaker 'shubh' is not compatible with model bulbul:v4-flash.",
        ),
        (
            json.dumps({"message": "This model is currently in beta and not available."}),
            "Sarvam TTS error (code=HTTP 422): This model is currently in beta and not available.",
        ),
        (
            "<html>Bad Gateway</html>",
            "Sarvam TTS error (HTTP 422): <html>Bad Gateway</html>",
        ),
    ],
)
def test_http_errors_keep_sarvams_own_message(body, expected):
    """Callers need the provider's wording — a speaker list, a gating notice."""
    assert _format_http_error(422, body) == expected


# --- WebSocket errors ----------------------------------------------------------


@pytest.mark.asyncio
async def test_websocket_errors_carry_the_code_and_request_id(monkeypatch):
    """A 422 gating error and a 400 bad speaker read alike without them."""
    error = {
        "type": "error",
        "data": {
            "request_id": "20260918_abc",
            "message": "This model is currently in beta and not available.",
            "code": 422,
            "details": {"model": V4_FLASH},
        },
    }
    service = _ws_service()
    service._websocket = _FakeWebsocket([json.dumps(error)])
    pushed = []
    monkeypatch.setattr(
        service, "push_error", AsyncMock(side_effect=lambda **kw: pushed.append(kw))
    )
    monkeypatch.setattr(service, "append_to_audio_context", AsyncMock())
    monkeypatch.setattr(service, "get_active_audio_context_id", lambda: "ctx")

    await service._receive_messages()

    assert len(pushed) == 1
    message = pushed[0]["error_msg"]
    assert "code=422" in message
    assert "request_id=20260918_abc" in message
    assert "This model is currently in beta and not available." in message
    assert str(V4_FLASH) in message


@pytest.mark.asyncio
async def test_audio_frames_are_appended_to_the_active_context(monkeypatch):
    """Guards the happy path the error test sits next to."""
    audio = b"\x00\x01" * 8
    message = {
        "type": "audio",
        "data": {
            "request_id": "r",
            "content_type": "audio/pcm",
            "audio": base64.b64encode(audio).decode(),
        },
    }
    service = _ws_service()
    service._sample_rate = 24000
    service._websocket = _FakeWebsocket([json.dumps(message)])
    appended = []
    monkeypatch.setattr(
        service, "append_to_audio_context", AsyncMock(side_effect=lambda _, f: appended.append(f))
    )
    monkeypatch.setattr(service, "get_active_audio_context_id", lambda: "ctx")
    monkeypatch.setattr(service, "stop_ttfb_metrics", AsyncMock())

    await service._receive_messages()

    assert len(appended) == 1
    frame = appended[0]
    assert isinstance(frame, TTSAudioRawFrame)
    assert frame.audio == audio
    assert frame.sample_rate == 24000
