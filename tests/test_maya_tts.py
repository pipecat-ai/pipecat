#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

from pipecat.services.maya.tts import (
    MAYA_SAMPLE_RATE,
    MayaHttpTTSService,
    language_to_maya_language,
)
from pipecat.transcriptions.language import Language


@pytest.mark.parametrize(
    "language, expected",
    [
        (Language.HI, "hi"),
        (Language.HI_IN, "hi"),
        (Language.BN, "bn"),
        (Language.GU_IN, "gu"),
        (Language.KN, "kn"),
        (Language.ML, "ml"),
        (Language.MR, "mr"),
        (Language.OR, "or"),
        (Language.PA_IN, "pa"),
        (Language.TA, "ta"),
        (Language.TE_IN, "te"),
    ],
)
def test_language_mapping(language: Language, expected: str):
    assert language_to_maya_language(language) == expected


def test_unverified_language_falls_back_to_base_code():
    # Maya doesn't advertise English, but the base code is still returned so
    # callers get intelligible output rather than an error.
    assert language_to_maya_language(Language.EN_US) == "en"


def test_defaults():
    service = MayaHttpTTSService(api_key="key", aiohttp_session=None)
    assert service._settings.voice == "Ananya"
    assert service._settings.region == "IN"
    assert service._settings.model is None
    assert service._url == "https://tts.mayaresearch.ai/v1/tts"


def test_settings_override_and_language_is_stored_as_service_string():
    service = MayaHttpTTSService(
        api_key="key",
        aiohttp_session=None,
        base_url="https://tts.example.com/",
        settings=MayaHttpTTSService.Settings(voice="Arjun", language=Language.TA_IN, region="US"),
    )
    assert service._settings.voice == "Arjun"
    assert service._settings.language == "ta"
    assert service._settings.region == "US"
    # Trailing slash on base_url must not produce a double slash.
    assert service._url == "https://tts.example.com/v1/tts"


def test_native_sample_rate():
    assert MAYA_SAMPLE_RATE == 24000
