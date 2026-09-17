#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json

import pytest

from pipecat.services.convozen.stt import (
    ConvozenSTTService,
    ConvozenSTTSettings,
    language_to_convozen_language,
)
from pipecat.transcriptions.language import Language


def _service(**settings) -> ConvozenSTTService:
    service = ConvozenSTTService.__new__(ConvozenSTTService)
    service._settings = ConvozenSTTSettings(
        model="akshara-pro",
        language=Language.EN,
        lang_tags=None,
        keywords=None,
        blank_penalty=None,
        word_timestamps=False,
    )
    for key, value in settings.items():
        setattr(service._settings, key, value)
    return service


def _fields(form) -> dict:
    return {opts["name"]: value for opts, _headers, value in form._fields}


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        (Language.EN, "en"),
        (Language.HI, "hi"),
        (Language.HI_IN, "hi"),
        (Language.TA_LK, "ta"),
        (Language.BN_BD, "bn"),
        (Language.FR, None),
        (Language.JA, None),
    ],
)
def test_language_maps_to_two_letter_code(language, expected):
    assert language_to_convozen_language(language) == expected


def test_lang_tags_derived_from_language():
    assert _service(language="hi")._resolve_lang_tags() == ["hi"]


def test_lang_tags_omitted_for_unsupported_language():
    # The server rejects tags outside its set, so no hint is sent at all.
    assert _service(language="fr")._resolve_lang_tags() is None


def test_explicit_lang_tags_override_language():
    service = _service(language="hi", lang_tags=["hi", "en"])
    assert service._resolve_lang_tags() == ["hi", "en"]


def test_form_carries_audio_model_and_channels():
    fields = _fields(_service()._build_form(b"RIFFfake"))
    assert fields["file"] == b"RIFFfake"
    assert fields["model"] == "akshara-pro"
    assert fields["audio_channels"] == "mono"


def test_form_encodes_lists_as_json():
    service = _service(language="hi", keywords=["ConvoZen", "Akshara"])
    fields = _fields(service._build_form(b"audio"))
    assert fields["lang_tags"] == json.dumps(["hi"])
    assert fields["keywords"] == json.dumps(["ConvoZen", "Akshara"])


def test_form_omits_unset_optional_fields():
    fields = _fields(_service()._build_form(b"audio"))
    for absent in ("keywords", "blank_penalty", "word_timestamps"):
        assert absent not in fields
    # Diarization is not exposed: it returns a different response shape.
    assert "speaker_labels" not in fields
    assert "num_speakers" not in fields


def test_form_includes_optional_fields_when_set():
    service = _service(blank_penalty=1.5, word_timestamps=True)
    fields = _fields(service._build_form(b"audio"))
    assert fields["blank_penalty"] == "1.5"
    assert fields["word_timestamps"] == "true"
