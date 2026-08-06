#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest

from pipecat.services.cartesia.turns.stt import CartesiaTurnsSTTService


def _service(**kwargs) -> CartesiaTurnsSTTService:
    service = CartesiaTurnsSTTService(api_key="test-key", sample_rate=16000, **kwargs)
    # sample_rate is normally set from StartFrame, which these tests skip.
    service._sample_rate = 16000
    return service


def test_cartesia_turns_websocket_url_includes_keyterm():
    service = _service(settings=CartesiaTurnsSTTService.Settings(keyterm=["Cartesia", "Ink 2"]))

    parsed = urlparse(service._websocket_url())
    query = parse_qs(parsed.query)

    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.cartesia.ai"
    assert parsed.path == "/stt/turns/websocket"
    assert query["model"] == ["ink-2"]
    assert query["sample_rate"] == ["16000"]
    assert query["keyterm"] == ["Cartesia", "Ink 2"]


def test_cartesia_turns_websocket_url_encodes_keyterm_spaces_as_percent_20():
    service = _service(settings=CartesiaTurnsSTTService.Settings(keyterm=["Ink 2"]))

    assert "keyterm=Ink%202" in service._websocket_url()


def test_cartesia_turns_websocket_url_omits_keyterm_when_not_set():
    service = _service()

    query = parse_qs(urlparse(service._websocket_url()).query)

    assert "keyterm" not in query


def test_cartesia_turns_websocket_url_clamps_keyterms_to_limits():
    service = _service(
        settings=CartesiaTurnsSTTService.Settings(keyterm=[f"term{i}" for i in range(150)])
    )

    query = parse_qs(urlparse(service._websocket_url()).query)

    assert query["keyterm"] == [f"term{i}" for i in range(100)]


@pytest.mark.asyncio
async def test_cartesia_turns_update_keyterm_reconnects(monkeypatch):
    service = _service(settings=CartesiaTurnsSTTService.Settings(keyterm=["Cartesia"]))
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    await service._update_settings(CartesiaTurnsSTTService.Settings(keyterm=["Ink 2"]))

    assert service._settings.keyterm == ["Ink 2"]
    reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_cartesia_turns_update_model_does_not_reconnect(monkeypatch):
    service = _service()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    await service._update_settings(CartesiaTurnsSTTService.Settings(model="ink-3"))

    reconnect.assert_not_awaited()
