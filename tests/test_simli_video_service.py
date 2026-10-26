#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SimliVideoService: Trinity-2 framing and the audio it animates from.

``av`` and ``simli`` are optional extras the test environment doesn't install,
so they are stubbed: the av fakes pass audio through unchanged, and the fake
Simli client echoes the audio it is sent back as the avatar's voice.
"""

import asyncio
import importlib
import sys
import types
from dataclasses import asdict, dataclass

import numpy as np
import pytest
from pydantic import ValidationError


class _FakeAudioFrame:
    def __init__(self, array: np.ndarray, layout: str):
        self._array = array
        self.layout = layout
        self.sample_rate = 0

    @classmethod
    def from_ndarray(cls, array: np.ndarray, layout: str):
        return cls(array, layout)

    def to_ndarray(self) -> np.ndarray:
        return self._array


class _FakeAudioResampler:
    def __init__(self, format: str, layout: str, rate: int):
        self.layout = layout
        self.rate = rate

    def resample(self, frame):
        return [] if frame is None else [frame]


class _FakeSimliClient:
    """Records what the service sends and echoes its audio back."""

    def __init__(self, api_key, config, simliURL, enableSFU):
        self.config = config
        self.sent: list[bytes] = []
        self.sent_immediately: list[bytes] = []
        self.buffer_clears = 0
        self._echo: asyncio.Queue = asyncio.Queue()

    async def start(self):
        return self

    async def stop(self):
        pass

    async def sendSilence(self):
        pass

    async def send(self, data: bytes):
        self.sent.append(data)
        self._echo.put_nowait(data)

    async def sendImmediate(self, data: bytes):
        self.sent_immediately.append(data)
        self._echo.put_nowait(data)

    async def clearBuffer(self):
        self.buffer_clears += 1

    async def getAudioStreamIterator(self):
        while True:
            data = await self._echo.get()
            yield _FakeAudioFrame(np.frombuffer(data, dtype=np.int16)[None, :], "mono")

    async def getVideoStreamIterator(self, targetFormat: str = "rgb24"):
        await asyncio.Event().wait()
        yield


@dataclass
class _SimliConfig20:
    """simli-ai 2.0.x's SimliConfig: no framing fields, sent as ``__dict__``."""

    faceId: str
    handleSilence: bool = True
    maxSessionLength: int = 600
    maxIdleTime: int = 30
    model: str = "fasttalk"

    def request(self) -> dict:
        return dict(self.__dict__)


@dataclass
class _SimliConfigWithFraming:
    """A SimliConfig that declares the framing fields and leaves unset ones out."""

    faceId: str
    handleSilence: bool = True
    maxSessionLength: int = 600
    maxIdleTime: int = 30
    model: str = "fasttalk"
    streaming_width: int | None = None
    streaming_height: int | None = None
    avatar_position_x: float | None = None
    avatar_position_y: float | None = None
    camera: dict | None = None

    def request(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _load_simli_video(monkeypatch, config_class=_SimliConfig20):
    av = types.ModuleType("av")
    av_audio = types.ModuleType("av.audio")
    av_audio_frame = types.ModuleType("av.audio.frame")
    av_audio_frame.AudioFrame = _FakeAudioFrame
    av_audio_resampler = types.ModuleType("av.audio.resampler")
    av_audio_resampler.AudioResampler = _FakeAudioResampler
    simli = types.ModuleType("simli")
    simli.SimliClient = _FakeSimliClient
    simli.SimliConfig = config_class

    monkeypatch.setitem(sys.modules, "av", av)
    monkeypatch.setitem(sys.modules, "av.audio", av_audio)
    monkeypatch.setitem(sys.modules, "av.audio.frame", av_audio_frame)
    monkeypatch.setitem(sys.modules, "av.audio.resampler", av_audio_resampler)
    monkeypatch.setitem(sys.modules, "simli", simli)
    monkeypatch.delitem(sys.modules, "pipecat.services.simli.video", raising=False)
    return importlib.import_module("pipecat.services.simli.video")


@pytest.fixture
def simli_video(monkeypatch):
    module = _load_simli_video(monkeypatch)
    yield module
    sys.modules.pop("pipecat.services.simli.video", None)


@pytest.fixture
def simli_video_with_framing_fields(monkeypatch):
    module = _load_simli_video(monkeypatch, _SimliConfigWithFraming)
    yield module
    sys.modules.pop("pipecat.services.simli.video", None)


def _framing(module):
    return module.SimliVideoService.Framing(
        streaming_width=1366,
        streaming_height=768,
        avatar_position_x=0.25,
        avatar_position_y=1.0,
        camera=module.SimliVideoService.Camera(zoom=1.2, orbit_y_off=15),
    )


#
# Framing
#


def test_framing_goes_into_session_request(simli_video):
    service = simli_video.SimliVideoService(
        api_key="key", face_id="face", framing=_framing(simli_video)
    )

    request = service._simli_client.config.request()

    assert request["faceId"] == "face"
    assert request["streaming_width"] == 1366
    assert request["streaming_height"] == 768
    assert request["avatar_position_x"] == 0.25
    assert request["avatar_position_y"] == 1.0
    # Only the camera fields that were set, so the rest stay neutral.
    assert request["camera"] == {"zoom": 1.2, "orbit_y_off": 15}
    # Unset fields keep Simli's defaults.
    assert "background_id" not in request
    assert "foreground_id" not in request


def test_framing_fills_declared_config_fields(simli_video_with_framing_fields):
    module = simli_video_with_framing_fields
    service = module.SimliVideoService(api_key="key", face_id="face", framing=_framing(module))

    config = service._simli_client.config
    assert config.streaming_width == 1366
    assert config.streaming_height == 768
    request = config.request()
    assert request["avatar_position_x"] == 0.25
    assert request["camera"] == {"zoom": 1.2, "orbit_y_off": 15}


def test_background_and_foreground_ids(simli_video):
    framing = simli_video.SimliVideoService.Framing(background_id="bg-id", foreground_id="fg-id")
    service = simli_video.SimliVideoService(api_key="key", face_id="face", framing=framing)

    request = service._simli_client.config.request()

    assert request["background_id"] == "bg-id"
    assert request["foreground_id"] == "fg-id"
    assert "streaming_width" not in request


def test_no_framing_leaves_session_request_unchanged(simli_video):
    service = simli_video.SimliVideoService(api_key="key", face_id="face")

    assert service._simli_client.config.request() == {
        "faceId": "face",
        "handleSilence": True,
        "maxSessionLength": 605,
        "maxIdleTime": 35,
        "model": "fasttalk",
    }


@pytest.mark.parametrize(
    "fields",
    [
        {"streaming_width": 256, "streaming_height": 768},
        {"avatar_position_x": 1.5},
        {"avatar_position_y": -0.1},
    ],
)
def test_framing_rejects_out_of_range_values(simli_video, fields):
    with pytest.raises(ValidationError):
        simli_video.SimliVideoService.Framing(**fields)
