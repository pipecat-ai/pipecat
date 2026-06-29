import importlib
import sys
import types

import pytest

from pipecat.frames.frames import TranscriptionFrame
from pipecat.transcriptions.language import Language


@pytest.fixture()
def funasr_module(monkeypatch):
    class FakeAutoModel:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.generate_calls = []
            self.instances.append(self)

        def generate(self, **kwargs):
            self.generate_calls.append(kwargs)
            return [{"text": " raw text "}]

    def rich_transcription_postprocess(text: str) -> str:
        return text.replace("raw", "clean")

    funasr = types.ModuleType("funasr")
    funasr.AutoModel = FakeAutoModel
    funasr_utils = types.ModuleType("funasr.utils")
    postprocess_utils = types.ModuleType("funasr.utils.postprocess_utils")
    postprocess_utils.rich_transcription_postprocess = rich_transcription_postprocess

    monkeypatch.setitem(sys.modules, "funasr", funasr)
    monkeypatch.setitem(sys.modules, "funasr.utils", funasr_utils)
    monkeypatch.setitem(sys.modules, "funasr.utils.postprocess_utils", postprocess_utils)
    sys.modules.pop("pipecat.services.funasr.stt", None)

    module = importlib.import_module("pipecat.services.funasr.stt")
    yield module, FakeAutoModel

    sys.modules.pop("pipecat.services.funasr.stt", None)


def test_funasr_stt_defaults_use_stt_settings(funasr_module):
    stt, fake_auto_model = funasr_module

    service = stt.FunASRSTTService()

    assert service._settings.model == "iic/SenseVoiceSmall"
    assert service._settings.language == "en"
    assert service._settings.use_itn is True
    assert fake_auto_model.instances[-1].kwargs == {
        "model": "iic/SenseVoiceSmall",
        "device": "cpu",
        "disable_update": True,
    }


def test_funasr_stt_applies_settings_overrides(funasr_module):
    stt, fake_auto_model = funasr_module

    service = stt.FunASRSTTService(
        device="cuda",
        settings=stt.FunASRSTTService.Settings(
            model="custom/sensevoice",
            language=Language.ZH,
            use_itn=False,
        ),
    )

    assert service._settings.model == "custom/sensevoice"
    assert service._settings.language == "zh"
    assert service._settings.use_itn is False
    assert fake_auto_model.instances[-1].kwargs == {
        "model": "custom/sensevoice",
        "device": "cuda",
        "disable_update": True,
    }


def test_language_to_funasr_language_accepts_enums_and_strings(funasr_module):
    stt, _ = funasr_module

    assert stt.language_to_funasr_language(Language.EN_US) == "en"
    assert stt.language_to_funasr_language("zh-CN") == "zh"
    assert stt.language_to_funasr_language("yue") == "yue"
    assert stt.language_to_funasr_language("de") == "auto"
    assert stt.language_to_funasr_language(None) == "auto"


@pytest.mark.asyncio
async def test_run_stt_uses_settings_for_generation(funasr_module):
    stt, fake_auto_model = funasr_module
    service = stt.FunASRSTTService(
        settings=stt.FunASRSTTService.Settings(
            model="custom/sensevoice",
            language=Language.ZH,
            use_itn=False,
        )
    )

    frames = [frame async for frame in service.run_stt(b"\x00\x00" * 160)]

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "clean text"
    assert frames[0].language == Language.ZH
    generate_call = fake_auto_model.instances[-1].generate_calls[-1]
    assert generate_call["language"] == "zh"
    assert generate_call["use_itn"] is False
