#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from urllib.parse import parse_qs, urlparse

from pipecat.services.cartesia.turns.stt import CartesiaTurnsSTTService


def test_cartesia_turns_websocket_url_includes_keyterm():
    service = CartesiaTurnsSTTService(
        api_key="test-key",
        sample_rate=16000,
        settings=CartesiaTurnsSTTService.Settings(keyterm=["Cartesia", "Ink 2"]),
    )
    # sample_rate is normally set from StartFrame; poke it directly since this
    # test calls _websocket_url() without running a full pipeline.
    service._sample_rate = 16000

    parsed = urlparse(service._websocket_url())
    query = parse_qs(parsed.query)

    assert parsed.scheme == "wss"
    assert parsed.netloc == "api.cartesia.ai"
    assert parsed.path == "/stt/turns/websocket"
    assert query["model"] == ["ink-2"]
    assert query["sample_rate"] == ["16000"]
    assert query["keyterm"] == ["Cartesia", "Ink 2"]


def test_cartesia_turns_websocket_url_omits_keyterm_when_not_set():
    service = CartesiaTurnsSTTService(api_key="test-key", sample_rate=16000)
    service._sample_rate = 16000

    query = parse_qs(urlparse(service._websocket_url()).query)

    assert "keyterm" not in query
