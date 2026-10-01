#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from urllib.parse import parse_qs, urlsplit

import pytest

from pipecat.services.together.tts import TogetherTTSService
from pipecat.transcriptions.language import Language


def _service(**settings) -> TogetherTTSService:
    return TogetherTTSService(
        api_key="test-key",
        settings=TogetherTTSService.Settings(**settings) if settings else None,
    )


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
