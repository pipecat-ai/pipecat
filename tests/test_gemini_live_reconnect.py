#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for GeminiLiveLLMService handshake error recovery, resumption fallback, and failure cleanup."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest
import websockets.exceptions
from google.genai import errors as genai_errors
from google.genai.types import LiveConnectConfig, SessionResumptionConfig
from websockets.frames import Close

from pipecat.frames.frames import ErrorFrame, InputAudioRawFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.google.gemini_live.llm import (
    MAX_CONSECUTIVE_FAILURES,
    GeminiLiveLLMService,
)


class _HealthySession:
    """Minimal fake AsyncSession that records client content seeds and stays open until cancelled."""

    def __init__(self, unblock_receive: asyncio.Event | None = None):
        self.client_content_calls: list[dict[str, Any]] = []
        self.realtime_calls: list[dict[str, Any]] = []
        self.closed = False
        self.receiving = asyncio.Event()
        self._unblock = unblock_receive or asyncio.Event()

    async def send_client_content(self, **kwargs):
        self.client_content_calls.append(kwargs)

    async def send_realtime_input(self, **kwargs):
        self.realtime_calls.append(kwargs)

    async def close(self):
        self.closed = True

    def receive(self):
        async def _gen():
            self.receiving.set()
            await self._unblock.wait()
            if False:
                yield None

        return _gen()


class _CrashingReceiveSession(_HealthySession):
    """Fake AsyncSession whose receive stream raises ConnectionClosedError(1011) immediately."""

    def receive(self):
        async def _gen():
            raise websockets.exceptions.ConnectionClosedError(
                Close(1011, "internal server error"),
                None,
            )
            yield

        return _gen()

    async def send_realtime_input(self, **kwargs):
        raise websockets.exceptions.ConnectionClosedError(
            Close(1011, "socket already closed"),
            None,
        )


def _make_service() -> GeminiLiveLLMService:
    """Construct an isolated GeminiLiveLLMService with metric/task helpers stubbed."""
    service = GeminiLiveLLMService(api_key="test-key")

    async def _noop(*args, **kwargs):
        pass

    service.start_ttfb_metrics = _noop
    service.stop_all_metrics = _noop
    return service


@pytest.mark.asyncio
async def test_rejected_resumption_handle_on_connect_falls_back_to_fresh_session():
    """A 1007 setup error during live.connect clears the stale resumption handle and re-seeds context."""
    service = _make_service()
    service._context = LLMContext(messages=[{"role": "user", "content": "Hello there"}])
    service._session_resumption_handle = "expired-handle-abc"

    connect_handles: list[str | None] = []
    healthy_session = _HealthySession()

    @asynccontextmanager
    async def fake_connect(*, model, config: LiveConnectConfig):
        handle = config.session_resumption.handle if config.session_resumption else None
        connect_handles.append(handle)
        if len(connect_handles) == 1:
            genai_errors.APIError.raise_error(1007, "session resumption handle expired", None)
        yield healthy_session

    service._client = SimpleNamespace(
        aio=SimpleNamespace(live=SimpleNamespace(connect=fake_connect))
    )

    spawned_configs: list[LiveConnectConfig] = []

    async def fake_connect_call(session_resumption_handle: str | None = None):
        cfg = LiveConnectConfig(
            session_resumption=SessionResumptionConfig(handle=session_resumption_handle)
        )
        spawned_configs.append(cfg)
        task = asyncio.create_task(service._connection_task_handler(config=cfg))
        await healthy_session.receiving.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    service._connect = fake_connect_call

    initial_cfg = LiveConnectConfig(
        session_resumption=SessionResumptionConfig(handle="expired-handle-abc")
    )
    await service._connection_task_handler(config=initial_cfg)

    assert connect_handles == ["expired-handle-abc", None]
    assert service._session_resumption_handle is None
    assert service._consecutive_failures == 1
    assert len(healthy_session.client_content_calls) == 1


@pytest.mark.asyncio
async def test_transient_connect_error_preserves_handle_until_final_retry():
    """Transient handshake errors keep the resumption handle on the first retry and clear it on the last."""
    service = _make_service()
    service._session_resumption_handle = "valid-handle-123"

    reconnect_handles: list[str | None] = []

    @asynccontextmanager
    async def transient_fail_connect(*, model, config: LiveConnectConfig):
        raise TimeoutError("opening handshake timed out")
        yield

    service._client = SimpleNamespace(
        aio=SimpleNamespace(live=SimpleNamespace(connect=transient_fail_connect))
    )

    async def fake_reconnect():
        reconnect_handles.append(service._session_resumption_handle)

    service._reconnect = fake_reconnect

    cfg = LiveConnectConfig(session_resumption=SessionResumptionConfig(handle="valid-handle-123"))

    await service._connection_task_handler(config=cfg)
    assert reconnect_handles == ["valid-handle-123"]
    assert service._session_resumption_handle == "valid-handle-123"

    await service._connection_task_handler(config=cfg)
    assert reconnect_handles == ["valid-handle-123", None]
    assert service._session_resumption_handle is None


@pytest.mark.asyncio
async def test_max_consecutive_failures_marks_permanent_and_clears_session():
    """Exhausting MAX_CONSECUTIVE_FAILURES sets force_treat_as_permanent=True and prevents Send error floods."""
    service = _make_service()
    pushed_frames: list[tuple[ErrorFrame, bool]] = []

    async def capture_error_frame(error: ErrorFrame, force_treat_as_permanent: bool = False):
        pushed_frames.append((error, force_treat_as_permanent))
        if force_treat_as_permanent:
            await service.set_usable(False)

    service.push_error_frame = capture_error_frame

    @asynccontextmanager
    async def connect_crashing_session(*, model, config: LiveConnectConfig):
        yield _CrashingReceiveSession()

    service._client = SimpleNamespace(
        aio=SimpleNamespace(live=SimpleNamespace(connect=connect_crashing_session))
    )
    service._consecutive_failures = MAX_CONSECUTIVE_FAILURES - 1

    await service._connection_task_handler(config=LiveConnectConfig())

    assert len(pushed_frames) == 1
    err_frame, is_permanent = pushed_frames[0]
    assert "Max consecutive failures" in err_frame.error
    assert is_permanent is True
    assert service.is_usable is False
    assert service._session is None
    assert service._ready_for_realtime_input is False

    for _ in range(5):
        await service._send_user_audio(
            InputAudioRawFrame(audio=b"\x01\x02" * 160, sample_rate=16000, num_channels=1)
        )

    assert len(pushed_frames) == 1


@pytest.mark.asyncio
async def test_disconnect_from_inside_connection_task_skips_self_cancel():
    """Calling _disconnect from inside _connection_task clears the task reference without self-cancelling."""
    service = _make_service()
    cancel_calls: list[asyncio.Task] = []

    async def record_cancel(task: asyncio.Task, timeout: float | None = None):
        cancel_calls.append(task)

    service.cancel_task = record_cancel

    async def run_inside_connection_task():
        service._connection_task = asyncio.current_task()
        await service._disconnect()

    task = asyncio.create_task(run_inside_connection_task())
    await task

    assert cancel_calls == []
    assert service._connection_task is None
