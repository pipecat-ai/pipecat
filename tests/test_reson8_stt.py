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
        assert isinstance(settings.include_timestamps, type(NOT_GIVEN))
        assert isinstance(settings.include_words, type(NOT_GIVEN))
        assert isinstance(settings.include_confidence, type(NOT_GIVEN))
        assert isinstance(settings.include_interim, type(NOT_GIVEN))
        assert isinstance(settings.custom_model_id, type(NOT_GIVEN))

    def test_settings_store_values(self):
        settings = Reson8STTSettings(
            model=None,
            language=None,
            include_timestamps=True,
            include_words=False,
            include_confidence=None,
            include_interim=True,
            custom_model_id="model-123",
        )
        assert settings.include_timestamps is True
        assert settings.include_words is False
        assert settings.include_confidence is None
        assert settings.include_interim is True
        assert settings.custom_model_id == "model-123"

    def test_settings_apply_update(self):
        store = Reson8STTSettings(
            model=None,
            language=None,
            include_timestamps=None,
            include_words=None,
            include_confidence=None,
            include_interim=True,
            custom_model_id=None,
        )
        delta = Reson8STTSettings(include_words=True)
        changed = store.apply_update(delta)
        assert store.include_words is True
        assert "include_words" in changed
        assert store.include_interim is True

    def test_settings_apply_update_custom_model_id(self):
        store = Reson8STTSettings(
            model=None,
            language=None,
            include_timestamps=None,
            include_words=None,
            include_confidence=None,
            include_interim=True,
            custom_model_id=None,
        )
        delta = Reson8STTSettings(custom_model_id="model-456")
        changed = store.apply_update(delta)
        assert store.custom_model_id == "model-456"
        assert "custom_model_id" in changed
        assert store.include_interim is True


class TestReson8STTServiceBuildUrl(unittest.TestCase):
    def test_basic_url(self):
        service = Reson8STTService.__new__(Reson8STTService)
        service._api_url = "https://api.reson8.dev"
        service._sample_rate = 16000
        service._settings = Reson8STTSettings(
            model=None,
            language=None,
            include_timestamps=None,
            include_words=None,
            include_confidence=None,
            include_interim=None,
            custom_model_id=None,
        )
        url = service._build_ws_url()
        assert url.startswith("wss://api.reson8.dev/v1/speech-to-text/realtime?")
        assert "encoding=pcm_s16le" in url
        assert "sample_rate=16000" in url
        assert "channels=1" in url
        assert "include_interim" not in url
        assert "custom_model_id" not in url

    def test_url_with_all_params(self):
        service = Reson8STTService.__new__(Reson8STTService)
        service._api_url = "https://api.reson8.dev"
        service._sample_rate = 16000
        service._settings = Reson8STTSettings(
            model=None,
            language="en",
            include_timestamps=True,
            include_words=True,
            include_confidence=True,
            include_interim=True,
            custom_model_id="model-123",
        )
        url = service._build_ws_url()
        assert "language=en" in url
        assert "include_timestamps=true" in url
        assert "include_words=true" in url
        assert "include_confidence=true" in url
        assert "include_interim=true" in url
        assert "custom_model_id=model-123" in url

    def test_url_http_to_ws(self):
        service = Reson8STTService.__new__(Reson8STTService)
        service._api_url = "http://localhost:8080"
        service._sample_rate = 16000
        service._settings = Reson8STTSettings(
            model=None,
            language=None,
            include_timestamps=None,
            include_words=None,
            include_confidence=None,
            include_interim=None,
            custom_model_id=None,
        )
        url = service._build_ws_url()
        assert url.startswith("ws://localhost:8080/")

    def test_url_with_custom_model_only(self):
        service = Reson8STTService.__new__(Reson8STTService)
        service._api_url = "https://api.reson8.dev"
        service._sample_rate = 16000
        service._settings = Reson8STTSettings(
            model=None,
            language=None,
            include_timestamps=None,
            include_words=None,
            include_confidence=None,
            include_interim=None,
            custom_model_id="my-custom-model",
        )
        url = service._build_ws_url()
        assert "custom_model_id=my-custom-model" in url
        assert "include_interim" not in url

    def test_url_with_language(self):
        service = Reson8STTService.__new__(Reson8STTService)
        service._api_url = "https://api.reson8.dev"
        service._sample_rate = 16000
        service._settings = Reson8STTSettings(
            model=None,
            language="nl",
            include_timestamps=None,
            include_words=None,
            include_confidence=None,
            include_interim=True,
            custom_model_id=None,
        )
        url = service._build_ws_url()
        assert "language=nl" in url
        assert "include_interim=true" in url

    def test_url_without_language(self):
        service = Reson8STTService.__new__(Reson8STTService)
        service._api_url = "https://api.reson8.dev"
        service._sample_rate = 16000
        service._settings = Reson8STTSettings(
            model=None,
            language=None,
            include_timestamps=None,
            include_words=None,
            include_confidence=None,
            include_interim=None,
            custom_model_id=None,
        )
        url = service._build_ws_url()
        assert "language" not in url

    def test_url_with_interim_only(self):
        service = Reson8STTService.__new__(Reson8STTService)
        service._api_url = "https://api.reson8.dev"
        service._sample_rate = 16000
        service._settings = Reson8STTSettings(
            model=None,
            language=None,
            include_timestamps=None,
            include_words=None,
            include_confidence=None,
            include_interim=True,
            custom_model_id=None,
        )
        url = service._build_ws_url()
        assert "include_interim=true" in url
        assert "custom_model_id" not in url
        assert "include_timestamps" not in url


if __name__ == "__main__":
    unittest.main()
