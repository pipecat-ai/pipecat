#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Qwen3-ASR STT service."""

import importlib
import sys
import types

import pytest

from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
from pipecat.transcriptions.language import Language


@pytest.fixture()
def qwen_asr_module(monkeypatch):
    class FakeResult:
        def __init__(self, text):
            self.text = text

    class FakeQwen3ASRModel:
        instances = []

        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs
            self.transcribe_calls = []
            FakeQwen3ASRModel.instances.append(self)

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return cls(model_name, **kwargs)

        def transcribe(self, audio, language=None):
            self.transcribe_calls.append({"audio": audio, "language": language})
            return [FakeResult(" hello world ")]

    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bfloat16"

    qwen_asr_mod = types.ModuleType("qwen_asr")
    qwen_asr_mod.Qwen3ASRModel = FakeQwen3ASRModel

    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "qwen_asr", qwen_asr_mod)
    sys.modules.pop("pipecat.services.qwen.stt", None)

    module = importlib.import_module("pipecat.services.qwen.stt")
    FakeQwen3ASRModel.instances = []
    yield module, FakeQwen3ASRModel

    sys.modules.pop("pipecat.services.qwen.stt", None)


def test_qwen3_stt_defaults(qwen_asr_module):
    stt, fake_model = qwen_asr_module

    service = stt.Qwen3STTService()

    assert service._settings.model == "Qwen/Qwen3-ASR-1.7B"
    assert service._settings.language == Language.EN
    assert service._device == "cuda:0"
    assert service._settings.max_new_tokens == 256
    assert fake_model.instances[-1].model_name == "Qwen/Qwen3-ASR-1.7B"
    assert fake_model.instances[-1].kwargs["dtype"] == "bfloat16"
    assert fake_model.instances[-1].kwargs["device_map"] == "cuda:0"


def test_qwen3_stt_settings_override(qwen_asr_module):
    stt, fake_model = qwen_asr_module

    service = stt.Qwen3STTService(
        model=stt.Model.ASR_0_6B,
        device="cuda:1",
        settings=stt.Qwen3STTService.Settings(
            language=Language.ZH,
            max_new_tokens=128,
        ),
    )

    assert service._settings.model == "Qwen/Qwen3-ASR-0.6B"
    assert service._settings.language == Language.ZH
    assert service._device == "cuda:1"
    assert service._settings.max_new_tokens == 128
    assert fake_model.instances[-1].kwargs["max_new_tokens"] == 128


def test_qwen3_stt_model_string_accepted(qwen_asr_module):
    stt, fake_model = qwen_asr_module

    service = stt.Qwen3STTService(model="Qwen/Qwen3-ASR-8B")

    assert service._settings.model == "Qwen/Qwen3-ASR-8B"
    assert fake_model.instances[-1].model_name == "Qwen/Qwen3-ASR-8B"


def test_language_to_service_language(qwen_asr_module):
    stt, _ = qwen_asr_module
    service = stt.Qwen3STTService()

    assert service.language_to_service_language(Language.EN) == "English"
    assert service.language_to_service_language(Language.ZH) == "Chinese"
    assert service.language_to_service_language(Language.JA) == "Japanese"
    assert service.language_to_service_language(Language.DE) == "German"


def test_can_generate_metrics(qwen_asr_module):
    stt, _ = qwen_asr_module
    service = stt.Qwen3STTService()

    assert service.can_generate_metrics() is True


def test_wants_wav_segments_is_false(qwen_asr_module):
    stt, _ = qwen_asr_module
    service = stt.Qwen3STTService()

    assert service.wants_wav_segments is False


@pytest.mark.asyncio
async def test_run_stt_yields_transcription_frame(qwen_asr_module):
    stt, fake_model = qwen_asr_module
    service = stt.Qwen3STTService(
        settings=stt.Qwen3STTService.Settings(language=Language.EN)
    )

    frames = [frame async for frame in service.run_stt(b"\x00\x00" * 160)]

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "hello world"
    assert frames[0].language == Language.EN
    call = fake_model.instances[-1].transcribe_calls[-1]
    assert call["language"] == "English"
    assert call["audio"][1] == stt._SAMPLE_RATE


@pytest.mark.asyncio
async def test_run_stt_passes_correct_language(qwen_asr_module):
    stt, fake_model = qwen_asr_module
    service = stt.Qwen3STTService(
        settings=stt.Qwen3STTService.Settings(language=Language.JA)
    )

    await service.run_stt(b"\x00\x00" * 160).__aiter__().__anext__()

    call = fake_model.instances[-1].transcribe_calls[-1]
    assert call["language"] == "Japanese"


@pytest.mark.asyncio
async def test_run_stt_returns_error_frame_when_model_missing(qwen_asr_module):
    stt, _ = qwen_asr_module
    service = stt.Qwen3STTService()
    service._qwen_model = None

    frames = [frame async for frame in service.run_stt(b"\x00\x00" * 160)]

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert frames[0].error == "Qwen3-ASR model not available"


@pytest.mark.asyncio
async def test_run_stt_returns_error_frame_on_inference_failure(qwen_asr_module):
    stt, fake_model = qwen_asr_module
    service = stt.Qwen3STTService()

    def fail(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    fake_model.instances[-1].transcribe = fail

    frames = [frame async for frame in service.run_stt(b"\x00\x00" * 160)]

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert "CUDA out of memory" in frames[0].error


@pytest.mark.asyncio
async def test_run_stt_skips_empty_transcript(qwen_asr_module):
    stt, fake_model = qwen_asr_module
    service = stt.Qwen3STTService()

    fake_model.instances[-1].transcribe = lambda *a, **kw: []

    frames = [frame async for frame in service.run_stt(b"\x00\x00" * 160)]

    assert frames == []
