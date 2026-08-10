#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Moonshine STT service."""

import importlib
import sys
import types

import pytest

from pipecat.frames.frames import TranscriptionFrame
from pipecat.transcriptions.language import Language

# Architectures published per language, best first. Streaming architectures ship
# for English only; the other languages have one or two models each.
FAKE_MODELS = {
    "ar": ["base"],
    "es": ["base"],
    "en": ["medium-streaming", "small-streaming", "base", "tiny-streaming", "tiny"],
    "ja": ["base", "tiny"],
    "ko": ["tiny"],
    "vi": ["base"],
    "uk": ["base"],
    "zh": ["base"],
}


@pytest.fixture()
def moonshine_module(monkeypatch):
    class FakeTranscriber:
        instances = []

        def __init__(self, model_path, model_arch):
            self.model_path = model_path
            self.model_arch = model_arch
            self.instances.append(self)

        def transcribe_without_streaming(self, audio, sample_rate):
            self.audio = audio
            self.sample_rate = sample_rate
            line = types.SimpleNamespace(text=" hello there ")
            return types.SimpleNamespace(lines=[line])

    def get_model_for_language(language, model_arch=None):
        if language not in FAKE_MODELS:
            raise ValueError(f"Language not found: {language}")
        available = FAKE_MODELS[language]
        if model_arch is None:
            model_arch = available[0]
        if model_arch not in available:
            raise ValueError(f"Model not found for language: {language} and arch: {model_arch}")
        return f"/models/{language}-{model_arch}", model_arch

    def string_to_model_arch(model_arch_string):
        if model_arch_string not in FAKE_MODELS["en"]:
            raise ValueError(f"Invalid model architecture string: {model_arch_string}")
        return model_arch_string

    moonshine_voice = types.ModuleType("moonshine_voice")
    moonshine_voice.Transcriber = FakeTranscriber
    moonshine_voice.get_model_for_language = get_model_for_language
    moonshine_voice.model_arch_to_string = lambda model_arch: model_arch
    moonshine_voice.string_to_model_arch = string_to_model_arch
    moonshine_voice.supported_languages = lambda: list(FAKE_MODELS)
    moonshine_voice.supported_languages_friendly = lambda: ", ".join(FAKE_MODELS)

    monkeypatch.setitem(sys.modules, "moonshine_voice", moonshine_voice)
    sys.modules.pop("pipecat.services.moonshine.stt", None)

    module = importlib.import_module("pipecat.services.moonshine.stt")
    FakeTranscriber.instances = []
    yield module, FakeTranscriber

    sys.modules.pop("pipecat.services.moonshine.stt", None)


def test_defaults_load_the_english_streaming_model(moonshine_module):
    stt, fake_transcriber = moonshine_module

    service = stt.MoonshineSTTService()

    assert service._settings.model == "small-streaming"
    assert service._settings.language == "en"
    assert fake_transcriber.instances[-1].model_path == "/models/en-small-streaming"
    assert fake_transcriber.instances[-1].model_arch == "small-streaming"


def test_regional_variants_resolve_to_the_base_code(moonshine_module):
    stt, fake_transcriber = moonshine_module

    service = stt.MoonshineSTTService(
        settings=stt.MoonshineSTTService.Settings(language=Language.EN_GB)
    )

    assert service._settings.language == "en"
    assert fake_transcriber.instances[-1].model_path == "/models/en-small-streaming"


def test_unavailable_architecture_falls_back_to_the_language_default(moonshine_module):
    stt, fake_transcriber = moonshine_module

    service = stt.MoonshineSTTService(
        settings=stt.MoonshineSTTService.Settings(language=Language.ES_MX)
    )

    assert service._settings.language == "es"
    assert service._settings.model == "base"
    assert fake_transcriber.instances[-1].model_path == "/models/es-base"


def test_requested_architecture_is_kept_when_available(moonshine_module):
    stt, fake_transcriber = moonshine_module

    service = stt.MoonshineSTTService(
        settings=stt.MoonshineSTTService.Settings(
            model=stt.Model.TINY,
            language=Language.JA,
        )
    )

    assert service._settings.model == "tiny"
    assert fake_transcriber.instances[-1].model_path == "/models/ja-tiny"


def test_unsupported_language_raises(moonshine_module):
    stt, _ = moonshine_module

    with pytest.raises(ValueError, match="does not support language 'fr'"):
        stt.MoonshineSTTService(settings=stt.MoonshineSTTService.Settings(language=Language.FR))


def test_unknown_model_raises(moonshine_module):
    stt, _ = moonshine_module

    with pytest.raises(ValueError, match="Invalid model architecture string"):
        stt.MoonshineSTTService(settings=stt.MoonshineSTTService.Settings(model="huge"))


def test_language_to_moonshine_language(moonshine_module):
    stt, _ = moonshine_module

    assert stt.language_to_moonshine_language(Language.EN_US) == "en"
    assert stt.language_to_moonshine_language(Language.ZH_TW) == "zh"
    assert stt.language_to_moonshine_language(Language.UK) == "uk"


def test_moonshine_language_to_frame_language(moonshine_module):
    stt, _ = moonshine_module

    assert stt.moonshine_language_to_frame_language("es") == Language.ES
    assert stt.moonshine_language_to_frame_language(None) is None
    assert stt.moonshine_language_to_frame_language("not-a-language") is None


def test_service_exposes_metrics_and_language_mapping(moonshine_module):
    stt, _ = moonshine_module
    service = stt.MoonshineSTTService()

    assert service.can_generate_metrics() is True
    assert service.language_to_service_language(Language.EN_US) == "en"


@pytest.mark.asyncio
async def test_run_stt_yields_a_transcription_frame(moonshine_module):
    stt, fake_transcriber = moonshine_module
    service = stt.MoonshineSTTService(
        settings=stt.MoonshineSTTService.Settings(language=Language.ES)
    )

    frames = [frame async for frame in service.run_stt(b"\x00\x00" * 160)]

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "hello there"
    assert frames[0].language == Language.ES
    assert fake_transcriber.instances[-1].sample_rate == stt.MOONSHINE_SAMPLE_RATE


@pytest.mark.asyncio
async def test_language_update_reloads_the_model(moonshine_module):
    stt, fake_transcriber = moonshine_module
    service = stt.MoonshineSTTService()

    await service._update_settings(stt.MoonshineSTTService.Settings(language=Language.UK))

    assert service._settings.language == "uk"
    assert service._settings.model == "base"
    assert service._transcriber is fake_transcriber.instances[-1]
    assert service._transcriber.model_path == "/models/uk-base"


@pytest.mark.asyncio
async def test_failed_reload_keeps_the_loaded_model(moonshine_module):
    stt, _ = moonshine_module
    service = stt.MoonshineSTTService()
    loaded = service._transcriber

    errors = []

    async def push_error(error_msg, exception=None, fatal=False):
        errors.append(error_msg)

    service.push_error = push_error

    def failing_transcriber(model_path, model_arch):
        raise RuntimeError("boom")

    stt.Transcriber = failing_transcriber

    await service._update_settings(stt.MoonshineSTTService.Settings(language=Language.JA))

    assert service._transcriber is loaded
    assert errors == ["Moonshine model load error: boom"]
