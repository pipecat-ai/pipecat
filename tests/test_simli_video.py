#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for SimliVideoService frame consumption."""

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import OutputImageRawFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.utils.asyncio.task_manager import TaskManager


@pytest.fixture()
def simli_video_module(monkeypatch):
    """Import the Simli video service with its optional deps stubbed out.

    ``av`` and ``simli`` are optional extras and are not installed in the test
    environment, so stub them before importing the service module.
    """
    av = types.ModuleType("av")
    av_audio = types.ModuleType("av.audio")
    av_audio_frame = types.ModuleType("av.audio.frame")
    av_audio_frame.AudioFrame = MagicMock()
    av_audio_resampler = types.ModuleType("av.audio.resampler")
    av_audio_resampler.AudioResampler = MagicMock()

    simli = types.ModuleType("simli")
    simli.SimliClient = MagicMock()
    simli.SimliConfig = MagicMock()

    monkeypatch.setitem(sys.modules, "av", av)
    monkeypatch.setitem(sys.modules, "av.audio", av_audio)
    monkeypatch.setitem(sys.modules, "av.audio.frame", av_audio_frame)
    monkeypatch.setitem(sys.modules, "av.audio.resampler", av_audio_resampler)
    monkeypatch.setitem(sys.modules, "simli", simli)
    sys.modules.pop("pipecat.services.simli.video", None)

    module = importlib.import_module("pipecat.services.simli.video")
    yield module

    sys.modules.pop("pipecat.services.simli.video", None)


def _make_video_frame():
    """Build a fake av video frame matching the interface the service reads."""
    frame = MagicMock()
    frame.width = 2
    frame.height = 2
    frame.pts = 123
    frame.to_rgb.return_value.to_image.return_value.tobytes.return_value = b"\x00" * 12
    return frame


async def _setup_service(service):
    task_manager = TaskManager(loop=asyncio.get_running_loop())
    await service.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=task_manager,
            pipeline_worker=SimpleNamespace(app_resources=None),
        )
    )


@pytest.mark.asyncio
async def test_video_consumer_does_not_wait_for_resampler(simli_video_module):
    """Video consumption must not be gated on the audio resampler event.

    The resampler event is only set once the first TTS audio frame arrives. The
    video consumer has no dependency on the resampler, so it must start draining
    the Simli video stream immediately, even when no TTS audio is ever produced.
    """
    service = simli_video_module.SimliVideoService(api_key="test", face_id="test")
    await _setup_service(service)

    async def video_iterator():
        yield _make_video_frame()

    service._simli_client.getVideoStreamIterator = MagicMock(return_value=video_iterator())

    pushed = []

    async def capture(frame, direction=None):
        pushed.append(frame)

    service.push_frame = capture

    # The resampler event is intentionally left unset (no TTS audio has arrived).
    assert not service._pipecat_resampler_event.is_set()

    await asyncio.wait_for(service._consume_and_process_video(), timeout=2.0)

    image_frames = [f for f in pushed if isinstance(f, OutputImageRawFrame)]
    assert len(image_frames) == 1
    assert image_frames[0].size == (2, 2)
