#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for runtime settings updates applied by GoogleSTTService."""

import pytest

from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.settings import STTSettings
from pipecat.transcriptions.language import Language


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr("pipecat.services.google.stt.default", lambda scopes: (object(), "project"))
    monkeypatch.setattr("google.cloud.speech_v2.SpeechAsyncClient", lambda **kwargs: object())
    return GoogleSTTService()


@pytest.mark.asyncio
@pytest.mark.parametrize("language", [Language.ES, "es-ES"])
async def test_base_settings_delta_changes_the_languages(service, language):
    delta = STTSettings(language=language)

    await service._update_settings(delta)

    assert service._get_language_codes() == ["es-ES"]
    # A switcher hands this same delta to its other services.
    assert delta.language == language
    assert not hasattr(delta, "languages")


@pytest.mark.asyncio
async def test_base_settings_delta_changes_the_model(service):
    await service._update_settings(STTSettings(model="telephony"))

    assert service._settings.model == "telephony"


@pytest.mark.asyncio
async def test_none_language_leaves_the_languages_unchanged(service):
    await service._update_settings(STTSettings(language=None))

    assert service._get_language_codes() == ["en-US"]
