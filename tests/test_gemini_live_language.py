#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for language settings in GeminiLiveLLMService."""

import pytest

from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transcriptions.language import Language


def _service(**settings) -> GeminiLiveLLMService:
    return GeminiLiveLLMService(
        api_key="test-key", settings=GeminiLiveLLMService.Settings(**settings)
    )


def test_constructor_converts_language_to_locale_code():
    service = _service(language=Language.ES)
    assert service._settings.language == "es-ES"


def test_constructor_keeps_language_string_as_given():
    service = _service(language="es-MX")
    assert service._settings.language == "es-MX"


@pytest.mark.asyncio
async def test_settings_update_converts_language_to_locale_code():
    service = _service()
    await service._update_settings(GeminiLiveLLMService.Settings(language=Language.FR))
    assert service._settings.language == "fr-FR"


@pytest.mark.asyncio
async def test_settings_update_keeps_language_string_as_given():
    service = _service()
    await service._update_settings(GeminiLiveLLMService.Settings(language="fr-CA"))
    assert service._settings.language == "fr-CA"
