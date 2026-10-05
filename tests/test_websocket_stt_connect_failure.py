#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for a websocket STT service whose connect failed."""

import asyncio
import importlib
import time
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame, Frame, InputAudioRawFrame
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.tests.utils import SleepFrame, run_test

SAMPLE_RATE = 16000


class Websocket:
    """An open websocket that never receives anything."""

    state = State.OPEN

    async def send(self, message):
        pass

    async def ping(self):
        pass

    async def close(self):
        self.state = State.CLOSED

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)


class ConnectFailingSTTService(WebsocketSTTService):
    """Connects the way websocket STT services do, failing the first ``failures`` times.

    A failed connect reports an error and leaves no websocket, and the receive
    loop starts either way.
    """

    def __init__(self, *, failures: int, **kwargs):
        super().__init__(settings=STTSettings(model=None, language=None), **kwargs)
        self._failures = failures
        self.connect_attempts = 0
        self.received_without_websocket = False
        self._receive_task: asyncio.Task | None = None
        self._reconnect_backoff_min_wait = 0.05
        self._reconnect_backoff_max_wait = 0.05

    async def setup(self, setup: FrameProcessorSetup):
        await super().setup(setup)
        await self._connect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        yield None

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        self.connect_attempts += 1
        if self.connect_attempts <= self._failures:
            self._websocket = None
            await self.push_error(
                error_msg="Unable to connect", exception=ConnectionError("HTTP 503")
            )
            return
        self._websocket = Websocket()

    async def _disconnect_websocket(self):
        self._websocket = None

    async def _receive_messages(self):
        if self._websocket is None:
            self.received_without_websocket = True
            raise ConnectionError("not connected")
        async for _ in self._websocket:
            pass


async def _run(service: ConnectFailingSTTService, seconds: float = 0.5):
    frames: list[Frame] = []
    for _ in range(int(seconds * 50)):
        frames += [
            InputAudioRawFrame(audio=b"\0\0" * 320, sample_rate=SAMPLE_RATE, num_channels=1),
            SleepFrame(sleep=0.02),
        ]
    return await run_test(
        service,
        frames_to_send=frames,
        expected_up_frames=None,
        pipeline_params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE),
    )


@pytest.mark.asyncio
async def test_failed_connect_is_retried_like_a_drop():
    service = ConnectFailingSTTService(failures=1, sample_rate=SAMPLE_RATE)

    await _run(service)

    assert service.connect_attempts == 2
    assert service.is_usable is True
    assert service.received_without_websocket is False


@pytest.mark.asyncio
async def test_connect_that_keeps_failing_gives_up_as_permanent():
    """Unusable once the reconnect attempts run out, so a `ServiceSwitcher` can fail over."""
    service = ConnectFailingSTTService(failures=100, sample_rate=SAMPLE_RATE)

    _, up = await _run(service)

    assert service.connect_attempts == 4
    assert service.is_usable is False
    errors = [frame for frame in up if isinstance(frame, ErrorFrame)]
    assert "failed to reconnect after 3 attempts" in errors[-1].error


@pytest.mark.asyncio
async def test_connected_service_is_not_reconnected():
    service = ConnectFailingSTTService(failures=0, sample_rate=SAMPLE_RATE)

    await _run(service)

    assert service.connect_attempts == 1
    assert service.is_usable is True


@pytest.mark.asyncio
async def test_failed_connect_is_not_retried_without_reconnect_on_error():
    service = ConnectFailingSTTService(
        failures=100, sample_rate=SAMPLE_RATE, reconnect_on_error=False
    )

    await _run(service)

    assert service.connect_attempts == 1


@pytest.mark.asyncio
async def test_teardown_stops_a_reconnect_waiting_on_backoff():
    service = ConnectFailingSTTService(failures=100, sample_rate=SAMPLE_RATE)
    service._reconnect_backoff_min_wait = 30
    service._reconnect_backoff_max_wait = 30

    start = time.monotonic()
    await _run(service, 0.2)

    assert time.monotonic() - start < 5
    assert service.connect_attempts == 2


SERVICES = [
    ("assemblyai.stt", "AssemblyAISTTService", {}),
    ("aws.stt", "AWSTranscribeSTTService", {"region": "us-east-1"}),
    ("gladia.stt", "GladiaSTTService", {}),
    ("meta.stt", "MetaSTTService", {}),
    ("openai.stt", "OpenAIRealtimeSTTService", {}),
    ("soniox.stt", "SonioxSTTService", {}),
    ("together.stt", "TogetherSTTService", {}),
    ("xai.stt", "XAISTTService", {}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("module", "name", "kwargs"), SERVICES)
async def test_failed_connect_starts_the_receive_loop(module, name, kwargs):
    """The receive loop is what retries, so it starts even without a websocket."""
    cls = getattr(importlib.import_module(f"pipecat.services.{module}"), name)
    service = cls(api_key="test-key", **kwargs)
    service._session_url = "wss://example.invalid"  # Gladia opens its session over HTTP first.
    started = []

    def create_task(coroutine, name=None):
        started.append(coroutine.__name__)
        coroutine.close()

    async def push_error(*args, **kwargs):
        pass

    async def fail_connect(uri, **kwargs):
        raise ConnectionRefusedError(111, "Connection refused")

    service.create_task = create_task
    service.push_error = push_error
    with patch("pipecat.services.websocket_service.websocket_connect", fail_connect):
        await service._connect()

    assert service._websocket is None
    assert "_receive_task_handler" in started
