#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest

from pipecat.frames.frames import TTSAudioRawFrame
from pipecat.services.together.tts import TogetherTTSService
from pipecat.transcriptions.language import Language


def _service(**settings) -> TogetherTTSService:
    return TogetherTTSService(
        api_key="test-key",
        settings=TogetherTTSService.Settings(**settings) if settings else None,
    )


async def _emitted_audio(service: TogetherTTSService, deltas: list[bytes]) -> list[bytes]:
    """Feed audio deltas through the service and return the emitted frame payloads."""
    frames = []

    async def append(context_id, frame):
        if isinstance(frame, TTSAudioRawFrame):
            frames.append(frame.audio)

    with (
        patch.object(service, "stop_ttfb_metrics", AsyncMock()),
        patch.object(service, "get_active_audio_context_id", return_value="ctx"),
        patch.object(service, "append_to_audio_context", side_effect=append),
    ):
        for delta in deltas:
            await service._handle_audio_delta({"delta": base64.b64encode(delta).decode()})
    return frames


@pytest.mark.asyncio
async def test_odd_length_deltas_emit_whole_samples():
    service = _service()
    deltas = [bytes(range(7)), b"\x07", bytes(range(8, 13)), bytes(range(13, 16))]

    frames = await _emitted_audio(service, deltas)

    assert all(len(frame) % 2 == 0 for frame in frames)
    assert b"".join(frames) == bytes(range(16))


@pytest.mark.asyncio
async def test_single_byte_delta_is_held_back():
    service = _service()

    assert await _emitted_audio(service, [b"\x01"]) == []
    assert await _emitted_audio(service, [b"\x02\x03"]) == [b"\x01\x02"]


@pytest.mark.asyncio
async def test_audio_done_drops_held_byte():
    service = _service()
    await _emitted_audio(service, [b"\x01\x02\x03"])

    with (
        patch.object(service, "stop_all_metrics", AsyncMock()),
        patch.object(service, "get_active_audio_context_id", return_value=None),
    ):
        await service._handle_audio_done({"item_id": "item"})

    assert await _emitted_audio(service, [b"\x0a\x0b"]) == [b"\x0a\x0b"]


def test_websocket_url_sends_default_language():
    query = parse_qs(urlsplit(_service()._build_websocket_url()).query)

    assert query["model"] == ["hexgrad/Kokoro-82M"]
    assert query["voice"] == ["af_heart"]
    assert query["language"] == ["en"]
    assert "max_partial_length" not in query


def test_websocket_url_escapes_blended_voice():
    url = _service(voice="af_bella(2)+af_heart(1)")._build_websocket_url()

    assert "voice=af_bella%282%29%2Baf_heart%281%29" in url
    assert "model=hexgrad/Kokoro-82M" in url


@pytest.mark.parametrize(
    "language,expected",
    [
        (Language.FR, "fr"),
        (Language.EN_US, "en"),
        (Language.ZH_HK, "zh-hk"),
        ("pt-BR", "pt"),
    ],
)
def test_language_resolves_to_together_code(language, expected):
    query = parse_qs(urlsplit(_service(language=language)._build_websocket_url()).query)

    assert query["language"] == [expected]


def test_language_none_is_omitted():
    query = parse_qs(urlsplit(_service(language=None)._build_websocket_url()).query)

    assert "language" not in query
