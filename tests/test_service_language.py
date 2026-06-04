#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for language parameter handling in TTS and STT services.

Verifies that Language enums, raw strings (e.g. "de-DE"), and unrecognized
strings are all resolved correctly at both init time and runtime update time.
"""

from collections.abc import AsyncGenerator
from typing import Optional
from unittest.mock import patch

import pytest

from pipecat.frames.frames import Frame
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language

# ---------------------------------------------------------------------------
# Minimal concrete subclasses for testing
# ---------------------------------------------------------------------------

# A simple language map using only base codes (like ElevenLabs does).
_LANGUAGE_MAP = {
    Language.DE: "de",
    Language.EN: "en",
    Language.FR: "fr",
}


class _TestTTSService(TTSService):
    """Minimal concrete TTS service for testing language resolution."""

    class Settings(TTSSettings):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        yield  # pragma: no cover

    def language_to_service_language(self, language: Language) -> str | None:
        return resolve_language(language, _LANGUAGE_MAP, use_base_code=True)


class _TestSTTService(STTService):
    """Minimal concrete STT service for testing language resolution."""

    class Settings(STTSettings):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        yield  # pragma: no cover

    async def process_audio_frame(self, frame, direction):
        pass  # pragma: no cover

    def language_to_service_language(self, language: Language) -> str | None:
        return resolve_language(language, _LANGUAGE_MAP, use_base_code=True)


# ---------------------------------------------------------------------------
# TTS init tests
# ---------------------------------------------------------------------------


class TestTTSLanguageInit:
    """Test language resolution at TTS service init time."""

    def test_language_enum_base_code(self):
        """Language.DE (base code in map) resolves to 'de'."""
        svc = _TestTTSService(settings=_TestTTSService.Settings(language=Language.DE))
        assert svc._settings.language == "de"

    def test_language_enum_regional_code(self):
        """Language.DE_DE (regional, not in map) falls back to base code 'de'."""
        svc = _TestTTSService(settings=_TestTTSService.Settings(language=Language.DE_DE))
        assert svc._settings.language == "de"

    def test_raw_string_base_code(self):
        """Raw string 'de' is converted to Language.DE then resolved to 'de'."""
        svc = _TestTTSService(settings=_TestTTSService.Settings(language="de"))
        assert svc._settings.language == "de"

    def test_raw_string_regional_code(self):
        """Raw string 'de-DE' is converted to Language.DE_DE then resolved to 'de'."""
        svc = _TestTTSService(settings=_TestTTSService.Settings(language="de-DE"))
        assert svc._settings.language == "de"

    def test_raw_string_other_regional(self):
        """Raw string 'en-US' is converted to Language.EN_US then resolved to 'en'."""
        svc = _TestTTSService(settings=_TestTTSService.Settings(language="en-US"))
        assert svc._settings.language == "en"

    def test_raw_string_unrecognized(self):
        """Unrecognized raw string logs a debug message and is passed through as-is."""
        with patch("pipecat.services.tts_service.logger") as mock_logger:
            svc = _TestTTSService(settings=_TestTTSService.Settings(language="klingon"))
            assert svc._settings.language == "klingon"
            mock_logger.debug.assert_called_once()
            assert "klingon" in mock_logger.debug.call_args[0][0]

    def test_language_none(self):
        """None language is left as None."""
        svc = _TestTTSService(settings=_TestTTSService.Settings(language=None))
        assert svc._settings.language is None


# ---------------------------------------------------------------------------
# STT init tests
# ---------------------------------------------------------------------------


class TestSTTLanguageInit:
    """Test language resolution at STT service init time."""

    def test_language_enum_base_code(self):
        """Language.FR (base code in map) resolves to 'fr'."""
        svc = _TestSTTService(settings=_TestSTTService.Settings(language=Language.FR))
        assert svc._settings.language == "fr"

    def test_language_enum_regional_code(self):
        """Language.FR_FR (regional, not in map) falls back to base code 'fr'."""
        svc = _TestSTTService(settings=_TestSTTService.Settings(language=Language.FR_FR))
        assert svc._settings.language == "fr"

    def test_raw_string_base_code(self):
        """Raw string 'fr' is converted to Language.FR then resolved to 'fr'."""
        svc = _TestSTTService(settings=_TestSTTService.Settings(language="fr"))
        assert svc._settings.language == "fr"

    def test_raw_string_regional_code(self):
        """Raw string 'de-DE' is converted to Language.DE_DE then resolved to 'de'."""
        svc = _TestSTTService(settings=_TestSTTService.Settings(language="de-DE"))
        assert svc._settings.language == "de"

    def test_raw_string_unrecognized(self):
        """Unrecognized raw string logs a debug message and is passed through as-is."""
        with patch("pipecat.services.stt_service.logger") as mock_logger:
            svc = _TestSTTService(settings=_TestSTTService.Settings(language="klingon"))
            assert svc._settings.language == "klingon"
            mock_logger.debug.assert_called_once()
            assert "klingon" in mock_logger.debug.call_args[0][0]

    def test_language_none(self):
        """None language is left as None."""
        svc = _TestSTTService(settings=_TestSTTService.Settings(language=None))
        assert svc._settings.language is None


# ---------------------------------------------------------------------------
# TTS runtime update tests
# ---------------------------------------------------------------------------


class TestTTSLanguageUpdate:
    """Test language resolution during runtime settings updates."""

    @pytest.mark.asyncio
    async def test_update_language_enum_base_code(self):
        """Updating with Language.EN resolves to 'en'."""
        svc = _TestTTSService(settings=_TestTTSService.Settings(language=None))
        await svc._update_settings(_TestTTSService.Settings(language=Language.EN))
        assert svc._settings.language == "en"

    @pytest.mark.asyncio
    async def test_update_language_enum_regional_code(self):
        """Updating with Language.DE_DE falls back to base code 'de'."""
        svc = _TestTTSService(settings=_TestTTSService.Settings(language=None))
        await svc._update_settings(_TestTTSService.Settings(language=Language.DE_DE))
        assert svc._settings.language == "de"

    @pytest.mark.asyncio
    async def test_update_raw_string_base_code(self):
        """Updating with raw string 'de' resolves to 'de'."""
        svc = _TestTTSService(settings=_TestTTSService.Settings(language=None))
        await svc._update_settings(_TestTTSService.Settings(language="de"))
        assert svc._settings.language == "de"

    @pytest.mark.asyncio
    async def test_update_raw_string_regional_code(self):
        """Updating with raw string 'de-DE' resolves to 'de'."""
        svc = _TestTTSService(settings=_TestTTSService.Settings(language=None))
        await svc._update_settings(_TestTTSService.Settings(language="de-DE"))
        assert svc._settings.language == "de"

    @pytest.mark.asyncio
    async def test_update_raw_string_unrecognized(self):
        """Updating with unrecognized string logs debug message and passes through."""
        svc = _TestTTSService(settings=_TestTTSService.Settings(language=None))
        with patch("pipecat.services.tts_service.logger") as mock_logger:
            await svc._update_settings(_TestTTSService.Settings(language="klingon"))
            assert svc._settings.language == "klingon"
            mock_logger.debug.assert_called_once()
            assert "klingon" in mock_logger.debug.call_args[0][0]


# ---------------------------------------------------------------------------
# STT runtime update tests
# ---------------------------------------------------------------------------


class TestSTTLanguageUpdate:
    """Test language resolution during runtime settings updates."""

    @pytest.mark.asyncio
    async def test_update_language_enum_base_code(self):
        """Updating with Language.EN resolves to 'en'."""
        svc = _TestSTTService(settings=_TestSTTService.Settings(language=None))
        await svc._update_settings(_TestSTTService.Settings(language=Language.EN))
        assert svc._settings.language == "en"

    @pytest.mark.asyncio
    async def test_update_language_enum_regional_code(self):
        """Updating with Language.FR_FR falls back to base code 'fr'."""
        svc = _TestSTTService(settings=_TestSTTService.Settings(language=None))
        await svc._update_settings(_TestSTTService.Settings(language=Language.FR_FR))
        assert svc._settings.language == "fr"

    @pytest.mark.asyncio
    async def test_update_raw_string_base_code(self):
        """Updating with raw string 'fr' resolves to 'fr'."""
        svc = _TestSTTService(settings=_TestSTTService.Settings(language=None))
        await svc._update_settings(_TestSTTService.Settings(language="fr"))
        assert svc._settings.language == "fr"

    @pytest.mark.asyncio
    async def test_update_raw_string_regional_code(self):
        """Updating with raw string 'fr-FR' resolves to 'fr'."""
        svc = _TestSTTService(settings=_TestSTTService.Settings(language=None))
        await svc._update_settings(_TestSTTService.Settings(language="fr-FR"))
        assert svc._settings.language == "fr"

    @pytest.mark.asyncio
    async def test_update_raw_string_unrecognized(self):
        """Updating with unrecognized string logs debug message and passes through."""
        svc = _TestSTTService(settings=_TestSTTService.Settings(language=None))
        with patch("pipecat.services.stt_service.logger") as mock_logger:
            await svc._update_settings(_TestSTTService.Settings(language="klingon"))
            assert svc._settings.language == "klingon"
            mock_logger.debug.assert_called_once()
            assert "klingon" in mock_logger.debug.call_args[0][0]
