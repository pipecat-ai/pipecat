#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.services.reson8.stt import Reson8STTService, Reson8STTSettings
from pipecat.services.settings import NOT_GIVEN


class TestReson8STTSettings(unittest.TestCase):
    def test_default_settings_are_not_given(self):
        settings = Reson8STTSettings()
        assert isinstance(settings.phrases, type(NOT_GIVEN))
        assert isinstance(settings.bias_strength, type(NOT_GIVEN))
        assert isinstance(settings.include_timestamps, type(NOT_GIVEN))
        assert isinstance(settings.include_words, type(NOT_GIVEN))
        assert isinstance(settings.include_confidence, type(NOT_GIVEN))

    def test_settings_store_mode(self):
        settings = Reson8STTSettings(
            model=None,
            language="nl",
            phrases=["hello", "world"],
            bias_strength=1.5,
            include_timestamps=True,
            include_words=False,
            include_confidence=None,
        )
        assert settings.language == "nl"
        assert settings.phrases == ["hello", "world"]
        assert settings.bias_strength == 1.5
        assert settings.include_timestamps is True
        assert settings.include_words is False
        assert settings.include_confidence is None

    def test_settings_apply_update(self):
        store = Reson8STTSettings(
            model=None,
            language="nl",
            phrases=None,
            bias_strength=None,
            include_timestamps=None,
            include_words=None,
            include_confidence=None,
        )
        delta = Reson8STTSettings(phrases=["custom"])
        changed = store.apply_update(delta)
        assert store.phrases == ["custom"]
        assert "phrases" in changed
        assert store.language == "nl"


class TestReson8STTServiceBuildUrl(unittest.TestCase):
    def test_basic_url(self):
        service = Reson8STTService.__new__(Reson8STTService)
        service._api_url = "https://api.reson8.dev"
        service._sample_rate = 16000
        service._settings = Reson8STTSettings(
            model=None,
            language="nl",
            phrases=None,
            bias_strength=None,
            include_timestamps=None,
            include_words=None,
            include_confidence=None,
        )
        url = service._build_ws_url()
        assert url.startswith("wss://api.reson8.dev/v1/speech-to-text/realtime?")
        assert "language=nl" in url
        assert "encoding=pcm_s16le" in url
        assert "sample_rate=16000" in url
        assert "channels=1" in url
        assert "phrases" not in url

    def test_url_with_phrases(self):
        service = Reson8STTService.__new__(Reson8STTService)
        service._api_url = "https://api.reson8.dev"
        service._sample_rate = 16000
        service._settings = Reson8STTSettings(
            model=None,
            language="en",
            phrases=["hello", "world"],
            bias_strength=1.5,
            include_timestamps=True,
            include_words=True,
            include_confidence=True,
        )
        url = service._build_ws_url()
        assert "language=en" in url
        assert "phrases=hello" in url
        assert "bias_strength=1.5" in url
        assert "include_timestamps=true" in url
        assert "include_words=true" in url
        assert "include_confidence=true" in url

    def test_url_http_to_ws(self):
        service = Reson8STTService.__new__(Reson8STTService)
        service._api_url = "http://localhost:8080"
        service._sample_rate = 16000
        service._settings = Reson8STTSettings(
            model=None,
            language="nl",
            phrases=None,
            bias_strength=None,
            include_timestamps=None,
            include_words=None,
            include_confidence=None,
        )
        url = service._build_ws_url()
        assert url.startswith("ws://localhost:8080/")


if __name__ == "__main__":
    unittest.main()
