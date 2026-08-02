#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Whisper STT model/language compatibility checking."""

from unittest.mock import MagicMock, patch

import pytest

from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language


def _build(supported, **settings):
    """Construct the service against a stand-in model, so nothing is downloaded."""
    with patch("pipecat.services.whisper.stt.WhisperModel") as mock_model:
        mock_model.return_value = MagicMock(supported_languages=supported)
        return WhisperSTTService(settings=WhisperSTTService.Settings(**settings))


def test_english_only_model_rejects_another_language():
    """The English-only models transcribe as English rather than refusing."""
    with pytest.raises(ValueError) as excinfo:
        _build(["en"], model="distil-medium.en", language=Language.ES)
    message = str(excinfo.value)
    assert "distil-medium.en" in message
    assert "large-v3-turbo" in message  # names a model that would work


def test_multilingual_model_accepts_the_language():
    service = _build(["en", "es", "zh"], model="large-v3-turbo", language=Language.ES)
    assert service._settings.language == "es"


def test_english_only_model_without_a_language_is_fine():
    """The default pairing — an English-only model left at its English default."""
    service = _build(["en"], model="distil-medium.en")
    assert service._settings.language == "en"


def test_model_that_does_not_publish_languages_is_not_second_guessed():
    service = _build(None, model="custom", language=Language.ES)
    assert service._settings.language == "es"
