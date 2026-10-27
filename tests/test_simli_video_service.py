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

from pipecat.frames.frames import (
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    SpeechOutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.tests.utils import SleepFrame, run_test

SPEECH = np.full(160, 1000, dtype=np.int16).tobytes()
SILENCE = bytes(320)


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
class _CameraSetting:
    """simli-ai 2.1's CameraSetting: neutral by default."""

    pos_x_off: float = 0.0
    pos_y_off: float = 0.0
    pos_z_off: float = 0.0
    rot_x_off: float = 0.0
    rot_y_off: float = 0.0
    rot_z_off: float = 0.0
    zoom: float = 1.0
    orbit_x_off: float = 0.0
    orbit_y_off: float = 0.0


@dataclass
class _SimliConfig:
    """simli-ai 2.1's SimliConfig: the request leaves out the fields left as None."""

    faceId: str
    handleSilence: bool = True
    maxSessionLength: int = 600
    maxIdleTime: int = 30
    model: str = "fasttalk"
    streaming_width: int | None = None
    streaming_height: int | None = None
    avatar_position_x: float | None = None
    avatar_position_y: float | None = None
    camera: _CameraSetting | None = None
    background_id: str | None = None
    foreground_id: str | None = None

    def toRequest(self) -> dict:
        return {key: value for key, value in asdict(self).items() if value is not None}


class _FakeVideoFrame:
    width, height, pts = 2, 2, 0

    def to_rgb(self):
        return self

    def to_image(self):
        return self

    def tobytes(self):
        return bytes(12)


class _IdleStreamingSimliClient(_FakeSimliClient):
    """Like Simli with handleSilence: idle video and silence from the moment it connects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for _ in range(3):
            self._echo.put_nowait(SILENCE)

    async def getVideoStreamIterator(self, targetFormat: str = "rgb24"):
        for _ in range(3):
            yield _FakeVideoFrame()
        await asyncio.Event().wait()


def _load_simli_video(monkeypatch, client_class=_FakeSimliClient):
    av = types.ModuleType("av")
    av_audio = types.ModuleType("av.audio")
    av_audio_frame = types.ModuleType("av.audio.frame")
    av_audio_frame.AudioFrame = _FakeAudioFrame
    av_audio_resampler = types.ModuleType("av.audio.resampler")
    av_audio_resampler.AudioResampler = _FakeAudioResampler
    simli = types.ModuleType("simli")
    simli.CameraSetting = _CameraSetting
    simli.SimliClient = client_class
    simli.SimliConfig = _SimliConfig

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


def _framing(module):
    return module.SimliVideoService.Framing(
        streaming_width=1366,
        streaming_height=768,
        avatar_position_x=0.25,
        avatar_position_y=1.0,
        camera=module.SimliVideoService.Camera(zoom=1.2, orbit_y_off=15),
    )


def _audio_frames(frames, frame_type):
    return [f for f in frames if type(f) is frame_type]


#
# Framing
#


def test_framing_goes_into_session_request(simli_video):
    service = simli_video.SimliVideoService(
        api_key="key", face_id="face", framing=_framing(simli_video)
    )

    request = service._simli_client.config.toRequest()

    assert request["faceId"] == "face"
    assert request["streaming_width"] == 1366
    assert request["streaming_height"] == 768
    assert request["avatar_position_x"] == 0.25
    assert request["avatar_position_y"] == 1.0
    # The SDK's CameraSetting: the fields that were set, the rest neutral.
    assert request["camera"] == asdict(_CameraSetting(zoom=1.2, orbit_y_off=15))
    # Unset fields keep Simli's defaults.
    assert "background_id" not in request
    assert "foreground_id" not in request


def test_background_and_foreground_ids(simli_video):
    framing = simli_video.SimliVideoService.Framing(background_id="bg-id", foreground_id="fg-id")
    service = simli_video.SimliVideoService(api_key="key", face_id="face", framing=framing)

    request = service._simli_client.config.toRequest()

    assert request["background_id"] == "bg-id"
    assert request["foreground_id"] == "fg-id"
    assert "streaming_width" not in request


def test_no_framing_leaves_session_request_unchanged(simli_video):
    service = simli_video.SimliVideoService(api_key="key", face_id="face")

    assert service._simli_client.config.toRequest() == {
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


#
# Audio
#


@pytest.mark.asyncio
async def test_speech_output_audio_animates_avatar(simli_video):
    """A speech stream (e.g. OpenAILiveLLMService) reaches Simli, and the avatar's
    voice goes on as a speech stream."""
    service = simli_video.SimliVideoService(api_key="key", face_id="face")

    down, _ = await run_test(
        service,
        frames_to_send=[
            SpeechOutputAudioRawFrame(audio=SPEECH, sample_rate=24000, num_channels=1),
            SpeechOutputAudioRawFrame(audio=SILENCE, sample_rate=24000, num_channels=1),
            SleepFrame(sleep=0.1),
        ],
    )

    assert service._simli_client.sent == [SPEECH, SILENCE]
    voice = _audio_frames(down, SpeechOutputAudioRawFrame)
    assert [f.audio for f in voice] == [SPEECH]
    assert voice[0].sample_rate == 24000
    # Only the avatar's voice goes on: the bot's isn't also pushed.
    assert len([f for f in down if isinstance(f, OutputAudioRawFrame)]) == 1


@pytest.mark.asyncio
async def test_tts_audio_animates_avatar(simli_video):
    """TTS audio goes to Simli once, and the avatar's voice comes back as TTS audio
    with silence left out, as before."""
    service = simli_video.SimliVideoService(api_key="key", face_id="face")

    down, _ = await run_test(
        service,
        frames_to_send=[
            TTSAudioRawFrame(audio=SPEECH, sample_rate=24000, num_channels=1),
            TTSAudioRawFrame(audio=SILENCE, sample_rate=24000, num_channels=1),
            SleepFrame(sleep=0.1),
        ],
    )

    assert service._simli_client.sent == [SPEECH, SILENCE]
    assert [f.audio for f in _audio_frames(down, TTSAudioRawFrame)] == [SPEECH]
    assert not _audio_frames(down, SpeechOutputAudioRawFrame)


@pytest.mark.asyncio
async def test_trinity_buffers_tts_audio_until_tts_stops(simli_video):
    service = simli_video.SimliVideoService(api_key="key", face_id="face", is_trinity_avatar=True)

    await run_test(
        service,
        frames_to_send=[
            TTSAudioRawFrame(audio=SPEECH, sample_rate=24000, num_channels=1),
            TTSAudioRawFrame(audio=SPEECH, sample_rate=24000, num_channels=1),
            TTSStoppedFrame(),
            SleepFrame(sleep=0.1),
        ],
    )

    assert service._simli_client.sent == []
    assert service._simli_client.sent_immediately == [SPEECH + SPEECH]


@pytest.mark.asyncio
async def test_trinity_sends_speech_stream_as_it_comes(simli_video):
    """A speech stream plays in real time, so it isn't held back to fill
    the Trinity buffer, and an interruption still clears Simli's buffer."""
    service = simli_video.SimliVideoService(api_key="key", face_id="face", is_trinity_avatar=True)

    await run_test(
        service,
        frames_to_send=[
            SpeechOutputAudioRawFrame(audio=SPEECH, sample_rate=24000, num_channels=1),
            SpeechOutputAudioRawFrame(audio=SPEECH, sample_rate=24000, num_channels=1),
            SleepFrame(sleep=0.1),
            InterruptionFrame(),
            SleepFrame(sleep=0.1),
        ],
    )

    assert service._simli_client.sent == [SPEECH, SPEECH]
    assert service._simli_client.sent_immediately == []
    assert service._simli_client.buffer_clears == 1


#
# Before the bot speaks
#


@pytest.fixture
def simli_video_idle_stream(monkeypatch):
    module = _load_simli_video(monkeypatch, client_class=_IdleStreamingSimliClient)
    yield module
    sys.modules.pop("pipecat.services.simli.video", None)


@pytest.mark.asyncio
async def test_idle_video_streams_before_the_bot_speaks(simli_video_idle_stream):
    """The avatar shows as soon as Simli is connected, not from the bot's first audio."""
    service = simli_video_idle_stream.SimliVideoService(api_key="key", face_id="face")

    down, _ = await run_test(service, frames_to_send=[SleepFrame(sleep=0.1)])

    assert len([f for f in down if isinstance(f, OutputImageRawFrame)]) == 3


@pytest.mark.asyncio
async def test_idle_audio_is_read_but_not_pushed(simli_video_idle_stream):
    """Simli's idle silence is read as it comes, so none of it waits to be read
    when the bot first speaks, and none of it goes on."""
    service = simli_video_idle_stream.SimliVideoService(api_key="key", face_id="face")

    down, _ = await run_test(service, frames_to_send=[SleepFrame(sleep=0.1)])

    assert service._simli_client._echo.empty()
    assert not [f for f in down if isinstance(f, OutputAudioRawFrame)]
