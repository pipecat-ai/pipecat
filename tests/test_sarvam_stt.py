#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

pytest.importorskip("sarvamai")

from pipecat.services.sarvam.stt import SarvamSTTService, language_to_sarvam_language
from pipecat.transcriptions.language import Language


@pytest.mark.parametrize(
    "language, expected",
    [
        (Language.HI_IN, "hi-IN"),
        (Language.UR_IN, "ur-IN"),
        (Language.KOK_IN, "kok-IN"),
        (Language.MAI_IN, "mai-IN"),
        (Language.SD_IN, "sd-IN"),
    ],
)
def test_language_to_sarvam_language_maps_enum_values(language, expected):
    assert language_to_sarvam_language(language) == expected


@pytest.mark.parametrize("language_code", ["ne-IN", "sat-IN"])
def test_get_language_string_passes_through_string_values(language_code):
    service = SarvamSTTService(api_key="test-key")
    service._settings.language = language_code

    assert service._get_language_string() == language_code


def test_get_language_string_resolves_enum_via_mapping():
    service = SarvamSTTService(api_key="test-key")
    service._settings.language = Language.HI_IN

    assert service._get_language_string() == "hi-IN"


def test_get_language_string_returns_model_default_when_unset():
    service = SarvamSTTService(api_key="test-key")
    service._settings.language = None

    assert service._get_language_string() == service._config.default_language
    assert service._config.default_language == "unknown"
