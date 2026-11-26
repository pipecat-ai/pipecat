#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.protocol import State

from pipecat.frames.frames import (
    EagerTranscriptionFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.services.cartesia.turns.stt import CartesiaTurnsSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import (
    EagerUserTurnStrategies,
    ExternalUserTurnStrategies,
)


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


def test_cartesia_turns_websocket_url_includes_turn_detection_thresholds():
    service = _service(
        settings=CartesiaTurnsSTTService.Settings(
            turn_start_threshold=0.7,
            turn_eager_end_threshold=0.5,
            turn_end_threshold=0.4,
            turn_end_timeout_ms=4500,
        )
    )

    query = parse_qs(urlparse(service._websocket_url()).query)

    assert query["turn_start_threshold"] == ["0.7"]
    assert query["turn_eager_end_threshold"] == ["0.5"]
    assert query["turn_end_threshold"] == ["0.4"]
    assert query["turn_end_timeout_ms"] == ["4500"]


def test_cartesia_turns_websocket_url_omits_unset_turn_detection_thresholds():
    service = _service(settings=CartesiaTurnsSTTService.Settings(turn_end_timeout_ms=8000))

    query = parse_qs(urlparse(service._websocket_url()).query)

    assert query["turn_end_timeout_ms"] == ["8000"]
    assert "turn_start_threshold" not in query
    assert "turn_eager_end_threshold" not in query
    assert "turn_end_threshold" not in query


def test_cartesia_turns_websocket_url_clamps_keyterms_to_limits():
    service = _service(
        settings=CartesiaTurnsSTTService.Settings(keyterm=[f"term{i}" for i in range(150)])
    )

    query = parse_qs(urlparse(service._websocket_url()).query)

    assert query["keyterm"] == [f"term{i}" for i in range(100)]


def test_cartesia_turns_websocket_url_includes_language_hints_as_base_codes():
    service = _service(
        settings=CartesiaTurnsSTTService.Settings(
            language_hints=[
                Language.HI,
                Language.EN_US,
                Language.EN,
                Language.PT_BR,
                Language.ES,
                Language.FR,
                Language.DE,
            ]
        )
    )

    query = parse_qs(urlparse(service._websocket_url()).query)

    assert query["language_hint"] == ["hi", "en", "pt", "es", "fr", "de"]


def test_cartesia_turns_rejects_more_than_six_distinct_language_hints():
    hints = [
        Language.EN,
        Language.ES,
        Language.PT,
        Language.FR,
        Language.DE,
        Language.HI,
        Language.JA,
    ]

    with pytest.raises(ValueError, match="language_hints accepts at most 6 distinct base codes"):
        _service(settings=CartesiaTurnsSTTService.Settings(language_hints=hints))


@pytest.mark.parametrize("language_hints", [None, []])
def test_cartesia_turns_websocket_url_omits_language_hints_when_not_set(language_hints):
    service = _service(settings=CartesiaTurnsSTTService.Settings(language_hints=language_hints))

    query = parse_qs(urlparse(service._websocket_url()).query)

    assert "language_hint" not in query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "language_hints, sent",
    [([Language.ES, Language.EN_GB], ["es", "en"]), ([], []), (None, [])],
)
async def test_cartesia_turns_update_language_hints_sends_config(monkeypatch, language_hints, sent):
    service = _service(settings=CartesiaTurnsSTTService.Settings(language_hints=[Language.HI]))
    service._websocket = MagicMock(state=State.OPEN, send=AsyncMock())
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    await service._update_settings(CartesiaTurnsSTTService.Settings(language_hints=language_hints))

    service._websocket.send.assert_awaited_once()
    assert json.loads(service._websocket.send.await_args.args[0]) == {
        "type": "config",
        "language_hints": sent,
    }
    reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_cartesia_turns_rejects_an_oversized_language_hint_update(monkeypatch):
    service = _service(settings=CartesiaTurnsSTTService.Settings(language_hints=[Language.HI]))
    service._websocket = MagicMock(state=State.OPEN, send=AsyncMock())
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)
    hints = [
        Language.EN,
        Language.ES,
        Language.PT,
        Language.FR,
        Language.DE,
        Language.HI,
        Language.JA,
    ]

    with pytest.raises(ValueError, match="language_hints accepts at most 6 distinct base codes"):
        await service._update_settings(
            CartesiaTurnsSTTService.Settings(keyterm=["Cartesia"], language_hints=hints)
        )

    service._websocket.send.assert_not_awaited()
    reconnect.assert_not_awaited()
    query = parse_qs(urlparse(service._websocket_url()).query)
    assert query["language_hint"] == ["hi"]
    assert "keyterm" not in query


@pytest.mark.asyncio
async def test_cartesia_turns_update_language_hints_without_a_connection_waits_for_the_url():
    service = _service()

    await service._update_settings(CartesiaTurnsSTTService.Settings(language_hints=[Language.JA]))

    query = parse_qs(urlparse(service._websocket_url()).query)
    assert query["language_hint"] == ["ja"]


@pytest.mark.asyncio
async def test_cartesia_turns_update_with_a_reconnect_carries_language_hints_in_the_url(
    monkeypatch,
):
    service = _service()
    service._websocket = MagicMock(state=State.OPEN, send=AsyncMock())
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    await service._update_settings(
        CartesiaTurnsSTTService.Settings(keyterm=["Cartesia"], language_hints=[Language.FR])
    )

    reconnect.assert_awaited_once()
    service._websocket.send.assert_not_awaited()
    assert parse_qs(urlparse(service._websocket_url()).query)["language_hint"] == ["fr"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "languages, expected",
    [
        (
            [{"language": "es", "confidence": 0.8}, {"language": "en", "confidence": 0.2}],
            Language.ES,
        ),
        ([{"language": "yue", "confidence": 0.9}], Language.YUE),
        ([], None),
        (None, None),
        ([{"language": "not-a-language", "confidence": 0.9}], None),
    ],
    ids=["strongest", "three-letter-code", "empty", "null", "unknown"],
)
async def test_cartesia_turns_labels_transcripts_with_the_detected_language(languages, expected):
    service = _service(settings=CartesiaTurnsSTTService.Settings(language_hints=[Language.HI]))
    service.push_frame = AsyncMock()
    service.broadcast_frame = AsyncMock()

    await service._handle_turn_end({"transcript": "hola", "languages": languages})

    frames = [call.args[0] for call in service.push_frame.await_args_list]
    transcription = next(f for f in frames if isinstance(f, TranscriptionFrame))
    assert transcription.language == expected


@pytest.mark.asyncio
async def test_cartesia_turns_uses_each_events_language_detection():
    service = _service(enable_eager_end_of_turn=True)
    service.push_frame = AsyncMock()
    service.broadcast_frame = AsyncMock()

    await service._handle_message(
        {
            "type": "turn.eager_end",
            "transcript": "bonjour",
            "languages": [{"language": "fr", "confidence": 0.9}],
        }
    )
    await service._handle_message({"type": "turn.resume"})
    await service._handle_message({"type": "turn.update", "transcript": "bonjour à tous"})
    await service._handle_message({"type": "turn.end", "transcript": "bonjour à tous"})

    transcriptions = [
        (type(frame), frame.language)
        for (frame,), _ in service.push_frame.await_args_list
        if isinstance(
            frame, (EagerTranscriptionFrame, InterimTranscriptionFrame, TranscriptionFrame)
        )
    ]
    assert transcriptions == [
        (EagerTranscriptionFrame, Language.FR),
        (InterimTranscriptionFrame, None),
        (TranscriptionFrame, None),
    ]


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


def test_cartesia_turns_recommends_external_strategies_by_default():
    strategies = _service().service_metadata_frame().user_turn_strategies

    assert isinstance(strategies, ExternalUserTurnStrategies)
    assert not isinstance(strategies, EagerUserTurnStrategies)


def test_cartesia_turns_recommends_eager_strategies_when_asked():
    strategies = _service(enable_eager_end_of_turn=True).service_metadata_frame()

    assert isinstance(strategies.user_turn_strategies, EagerUserTurnStrategies)


@pytest.mark.asyncio
async def test_cartesia_turns_reports_an_eager_end_of_turn_only_when_enabled():
    for enabled, expected in ((False, []), (True, [EagerTranscriptionFrame])):
        service = _service(enable_eager_end_of_turn=enabled)
        service.push_frame = AsyncMock()

        await service._handle_turn_eager_end({"transcript": "book a flight"})

        pushed = [type(call.args[0]) for call in service.push_frame.await_args_list]
        assert pushed == expected
