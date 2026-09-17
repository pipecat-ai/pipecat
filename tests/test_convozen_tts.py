#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.services.convozen.tts import (
    ConvozenHttpTTSService,
    ConvozenHttpTTSSettings,
    language_to_convozen_language,
)
from pipecat.transcriptions.language import Language


def _service(sample_rate: int | None = None, **settings) -> ConvozenHttpTTSService:
    service = ConvozenHttpTTSService.__new__(ConvozenHttpTTSService)
    service._settings = ConvozenHttpTTSSettings(
        model="ragini-v1",
        voice="roohi",
        language="en",
        speed=1.0,
    )
    for key, value in settings.items():
        setattr(service._settings, key, value)
    service._sample_rate = sample_rate
    return service


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        (Language.EN, "en"),
        (Language.HI, "hi"),
        (Language.ML_IN, "ml"),
        (Language.DE, None),
    ],
)
def test_language_maps_to_two_letter_code(language, expected):
    assert language_to_convozen_language(language) == expected


def test_sample_rate_defaults_to_model_native_rate():
    assert _service()._request_sample_rate() == 24000
    assert _service(model="ragini-lite")._request_sample_rate() == 22050


def test_explicit_sample_rate_wins():
    assert _service(sample_rate=8000)._request_sample_rate() == 8000


def test_body_uses_speaker_not_voice():
    body = _service()._build_body("hello")
    assert body["speaker"] == "roohi"
    assert "voice" not in body


def test_body_has_no_format_field():
    # The endpoint has no format field; output is always WAV.
    assert "format" not in _service()._build_body("hello")


def test_body_fields():
    service = _service(language="hi", voice="amaya", speed=1.2)
    assert service._build_body("नमस्ते") == {
        "text": "नमस्ते",
        "language": "hi",
        "speaker": "amaya",
        "model": "ragini-v1",
        "sample_rate": "24000",
        "speed": "1.2",
        "stream": "true",
    }
