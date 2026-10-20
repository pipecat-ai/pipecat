#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.protocol import State

from pipecat.services.sarvam.tts import SarvamHttpTTSService, SarvamTTSService

V4_FLASH = "bulbul:v4-flash"


class _FakeWebsocket:
    def __init__(self, *, state=State.OPEN):
        self.state = state
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    async def close(self):
        self.state = State.CLOSED


class _FakeResponse:
    status = 200

    async def json(self):
        return {"audios": [""], "request_id": "test"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeSession:
    """Records the JSON body of each POST."""

    def __init__(self):
        self.payloads = []

    def post(self, url, json=None, headers=None):
        self.payloads.append(json)
        return _FakeResponse()


def _service(**settings_kwargs) -> SarvamTTSService:
    settings_kwargs.setdefault("model", V4_FLASH)
    return SarvamTTSService(
        api_key="test-key", settings=SarvamTTSService.Settings(**settings_kwargs)
    )


def _http_service(session: _FakeSession, **settings_kwargs) -> SarvamHttpTTSService:
    settings_kwargs.setdefault("model", V4_FLASH)
    service = SarvamHttpTTSService(
        api_key="test-key",
        aiohttp_session=session,  # type: ignore[arg-type]
        settings=SarvamHttpTTSService.Settings(**settings_kwargs),
    )
    service._sample_rate = service._init_sample_rate
    return service


def _sent_config(service: SarvamTTSService) -> dict:
    """Read back the config the service put on the wire."""
    messages = [json.loads(m) for m in service._websocket.sent]
    configs = [m["data"] for m in messages if m["type"] == "config"]
    assert configs, "no config message was sent"
    return configs[-1]


def _url_model(service: SarvamTTSService) -> str:
    return parse_qs(urlparse(service._websocket_url).query)["model"][0]


async def _synthesize(service: SarvamHttpTTSService) -> dict:
    async for _ in service.run_tts("hello", "ctx"):
        pass
    return service._session.payloads[-1]


def test_v4_flash_defaults():
    """The service has to land on the same speaker and rate the API defaults to."""
    service = _service()

    assert service._settings.voice == "shubh_en_narration_gentle"
    assert service._init_sample_rate == 24000
    assert service._settings.enable_preprocessing is True


@pytest.mark.asyncio
async def test_v4_flash_sends_pitch_and_loudness():
    """v4-flash applies both; the v3 models ignore them."""
    service = _service(pitch=0.3, loudness=1.5, pace=1.2)
    service._websocket = _FakeWebsocket()

    await service._send_config()
    config = _sent_config(service)

    assert config["pitch"] == 0.3
    assert config["loudness"] == 1.5
    assert config["pace"] == 1.2


@pytest.mark.asyncio
@pytest.mark.parametrize("temperature", [0.6, 0.9])
async def test_v4_flash_drops_temperature(temperature):
    """The API pins it to 0.6, so sending any value would only mislead."""
    service = _service(temperature=temperature)
    service._websocket = _FakeWebsocket()

    await service._send_config()

    assert "temperature" not in _sent_config(service)


@pytest.mark.asyncio
async def test_v4_flash_drops_a_runtime_temperature_update():
    service = _service()
    service._websocket = _FakeWebsocket()

    await service._update_settings(SarvamTTSService.Settings(temperature=0.9))

    assert "temperature" not in _sent_config(service)


def test_v4_flash_clamps_pace_to_its_own_range():
    service = _service(pace=2.5)

    assert service._settings.pace == 2.0


@pytest.mark.asyncio
async def test_switching_to_v4_flash_reconnects_with_its_capabilities(monkeypatch):
    """The model is pinned on the URL, so the switch needs a new connection."""
    service = _service(model="bulbul:v3", voice=None)
    disconnect, connect = AsyncMock(), AsyncMock()
    monkeypatch.setattr(service, "_disconnect", disconnect)
    monkeypatch.setattr(service, "_connect", connect)

    await service._update_settings(SarvamTTSService.Settings(model=V4_FLASH))

    assert _url_model(service) == V4_FLASH
    assert service._config.supports_pitch is True
    # bulbul:v4-flash rejects the v3 default, so the voice moves with the model.
    assert service._settings.voice == "shubh_en_narration_gentle"
    disconnect.assert_awaited_once()
    connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_voice_switched_with_the_model_is_kept(monkeypatch):
    service = _service(model="bulbul:v3", voice=None)
    monkeypatch.setattr(service, "_disconnect", AsyncMock())
    monkeypatch.setattr(service, "_connect", AsyncMock())

    await service._update_settings(
        SarvamTTSService.Settings(model=V4_FLASH, voice="ritu_hi_edtech")
    )

    assert service._settings.voice == "ritu_hi_edtech"


@pytest.mark.asyncio
async def test_an_unsupported_model_update_is_ignored(monkeypatch):
    service = _service(model="bulbul:v3", voice=None)
    service._websocket = _FakeWebsocket()
    connect = AsyncMock()
    monkeypatch.setattr(service, "_connect", connect)

    changed = await service._update_settings(SarvamTTSService.Settings(model="bulbul:v9"))

    assert service._settings.model == "bulbul:v3"
    assert _url_model(service) == "bulbul:v3"
    assert "model" not in changed
    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_http_payload_carries_only_the_parameters_v4_flash_supports():
    session = _FakeSession()
    service = _http_service(session, pitch=0.3, loudness=1.5, temperature=0.9)

    payload = await _synthesize(service)

    assert payload["model"] == V4_FLASH
    assert payload["speaker"] == "shubh_en_narration_gentle"
    assert payload["pitch"] == 0.3
    assert payload["loudness"] == 1.5
    assert "temperature" not in payload


@pytest.mark.asyncio
async def test_switching_the_http_service_to_v4_flash_takes_its_capabilities():
    """`run_tts` builds the payload from the model config, so the switch has to swap it."""
    session = _FakeSession()
    service = _http_service(session, model="bulbul:v3", voice=None)

    await service._update_settings(
        SarvamHttpTTSService.Settings(model=V4_FLASH, pitch=0.3, loudness=1.5)
    )
    payload = await _synthesize(service)

    assert payload["model"] == V4_FLASH
    assert payload["speaker"] == "shubh_en_narration_gentle"
    assert payload["pitch"] == 0.3
    assert "temperature" not in payload
