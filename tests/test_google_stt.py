#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest
from google.cloud.speech_v2.types import cloud_speech

from pipecat.services.google.stt import (
    google_stt_model_supports_adaptation,
    normalize_google_speech_adaptation,
)


def test_google_stt_model_supports_adaptation():
    assert google_stt_model_supports_adaptation("latest_long") is True
    assert google_stt_model_supports_adaptation("telephony") is False
    assert google_stt_model_supports_adaptation("TELEPHONY") is False
    assert google_stt_model_supports_adaptation(None) is True


def test_normalize_google_speech_adaptation_accepts_native_message():
    adaptation = cloud_speech.SpeechAdaptation()

    normalized = normalize_google_speech_adaptation(adaptation)

    assert normalized is adaptation


def test_normalize_google_speech_adaptation_converts_phrase_set_references():
    normalized = normalize_google_speech_adaptation(
        {
            "phrase_set_references": [
                "projects/test/locations/global/phraseSets/support-terms",
            ]
        }
    )

    assert len(normalized.phrase_sets) == 1
    assert normalized.phrase_sets[0].phrase_set == (
        "projects/test/locations/global/phraseSets/support-terms"
    )


def test_normalize_google_speech_adaptation_converts_string_and_inline_phrase_sets():
    normalized = normalize_google_speech_adaptation(
        {
            "phrase_sets": [
                "projects/test/locations/global/phraseSets/catalog",
                {
                    "phrases": [
                        {"value": "pipecat", "boost": 15.0},
                        {"value": "voice pipeline"},
                    ]
                },
            ]
        }
    )

    assert normalized.phrase_sets[0].phrase_set == (
        "projects/test/locations/global/phraseSets/catalog"
    )
    assert normalized.phrase_sets[1].inline_phrase_set.phrases[0].value == "pipecat"
    assert normalized.phrase_sets[1].inline_phrase_set.phrases[0].boost == 15.0
    assert normalized.phrase_sets[1].inline_phrase_set.phrases[1].value == "voice pipeline"


def test_normalize_google_speech_adaptation_rejects_invalid_phrase_set_entries():
    with pytest.raises(ValueError, match="Invalid Google SpeechAdaptation phrase_set entry"):
        normalize_google_speech_adaptation({"phrase_sets": [123]})
