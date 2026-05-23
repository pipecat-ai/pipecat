#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for KittenTTSService."""

import sys
import types

import numpy as np
import pytest

from pipecat.frames.frames import (
    AggregatedTextFrame,
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.kittentts.tts import KittenTTSService
from pipecat.tests.utils import run_test


class FakeKittenTTS:
    instances: list["FakeKittenTTS"] = []

    def __init__(self, model_name: str, cache_dir: str | None = None, backend: str | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.backend = backend
        self.calls = []
        FakeKittenTTS.instances.append(self)

    def generate_stream(self, text: str, *, voice: str, speed: float, clean_text: bool):
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "clean_text": clean_text,
            }
        )
        yield np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)


class ErrorKittenTTS(FakeKittenTTS):
    def generate_stream(self, text: str, *, voice: str, speed: float, clean_text: bool):
        raise RuntimeError("synthesis failed")
        yield


@pytest.fixture(autouse=True)
def fake_kittentts(monkeypatch):
    FakeKittenTTS.instances = []
    module = types.ModuleType("kittentts")
    module.KittenTTS = FakeKittenTTS
    monkeypatch.setitem(sys.modules, "kittentts", module)
    return module


@pytest.mark.asyncio
async def test_run_kittentts_success() -> None:
    service = KittenTTSService(
        speed=1.2,
        clean_text=False,
        cache_dir="/tmp/kittentts-cache",
        backend="cpu",
        settings=KittenTTSService.Settings(
            model="KittenML/kitten-tts-nano-0.8",
            voice="expr-voice-5-m",
        ),
    )

    down_frames, _ = await run_test(
        service,
        frames_to_send=[TTSSpeakFrame(text="Hello from KittenTTS.")],
    )

    frame_types = [type(frame) for frame in down_frames]
    assert AggregatedTextFrame in frame_types
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types
    assert TTSTextFrame in frame_types

    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert audio_frames
    assert all(frame.sample_rate == 24000 for frame in audio_frames)
    assert all(frame.num_channels == 1 for frame in audio_frames)

    assert len(FakeKittenTTS.instances) == 1
    fake_model = FakeKittenTTS.instances[0]
    assert fake_model.model_name == "KittenML/kitten-tts-nano-0.8"
    assert fake_model.cache_dir == "/tmp/kittentts-cache"
    assert fake_model.backend == "cpu"
    assert fake_model.calls == [
        {
            "text": "Hello from KittenTTS.",
            "voice": "expr-voice-5-m",
            "speed": 1.2,
            "clean_text": False,
        }
    ]


@pytest.mark.asyncio
async def test_run_kittentts_error(fake_kittentts) -> None:
    fake_kittentts.KittenTTS = ErrorKittenTTS
    service = KittenTTSService()

    _, up_frames = await run_test(
        service,
        frames_to_send=[TTSSpeakFrame(text="Error case.", append_to_context=False)],
        expected_up_frames=[ErrorFrame],
    )

    assert isinstance(up_frames[0], ErrorFrame)
    assert "synthesis failed" in up_frames[0].error


def test_audio_to_pcm16_clips_and_converts() -> None:
    from pipecat.services.kittentts import tts as kittentts_tts

    pcm = kittentts_tts._audio_to_pcm16(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    samples = np.frombuffer(pcm, dtype="<i2")

    assert samples.tolist() == [-32767, -32767, 0, 32767, 32767]
