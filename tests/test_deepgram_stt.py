#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import contextlib
import io
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from deepgram.core import ApiError
from loguru import logger

from pipecat.services.deepgram.stt import DeepgramSTTService, _derive_deepgram_urls
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.network import QuickFailureTracker
from tests.frame_processor_helpers import frame_processor_setup


def _make_bare_service() -> DeepgramSTTService:
    """Build a DeepgramSTTService without running __init__, wiring just enough
    for _connection_handler() to run: a real create_task/cancel_task pair (so
    the keepalive task is properly started and torn down) and mocked
    push_error/_build_connect_kwargs.
    """
    service = DeepgramSTTService.__new__(DeepgramSTTService)
    service._name = "DeepgramSTTService"
    service._connection = None
    service._connection_settled = asyncio.Event()
    service._quick_failure_tracker = QuickFailureTracker()
    service._build_connect_kwargs = MagicMock(return_value={})
    service.push_error = AsyncMock()
    service.create_task = lambda coro, name=None: asyncio.create_task(coro)

    async def fake_cancel_task(task, timeout=None):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    service.cancel_task = fake_cancel_task
    return service


def _failing_connect_cm(exc: Exception):
    class _CM:
        async def __aenter__(self):
            raise exc

        async def __aexit__(self, *args):
            return False

    return _CM()


def _dropping_connect_cm(exc: Exception):
    """A connect that completes the handshake and then loses the connection."""

    class _CM:
        async def __aenter__(self):
            connection = MagicMock()
            connection.start_listening = AsyncMock(side_effect=exc)
            return connection

        async def __aexit__(self, *args):
            return False

    return _CM()


@pytest.mark.parametrize(
    "base_url, expected_ws, expected_http",
    [
        # Secure schemes
        ("wss://mydeepgram.com", "wss://mydeepgram.com", "https://mydeepgram.com"),
        ("https://mydeepgram.com", "wss://mydeepgram.com", "https://mydeepgram.com"),
        # Insecure schemes (air-gapped deployments)
        ("ws://mydeepgram.com", "ws://mydeepgram.com", "http://mydeepgram.com"),
        ("http://mydeepgram.com", "ws://mydeepgram.com", "http://mydeepgram.com"),
        # Bare hostname defaults to secure
        ("mydeepgram.com", "wss://mydeepgram.com", "https://mydeepgram.com"),
        # With port
        ("ws://localhost:8080", "ws://localhost:8080", "http://localhost:8080"),
        ("wss://localhost:443", "wss://localhost:443", "https://localhost:443"),
        ("localhost:8080", "wss://localhost:8080", "https://localhost:8080"),
        # With path
        ("wss://host/v1/listen", "wss://host/v1/listen", "https://host/v1/listen"),
        ("http://host/v1/listen", "ws://host/v1/listen", "http://host/v1/listen"),
    ],
)
def test_derive_deepgram_urls(base_url, expected_ws, expected_http):
    ws_url, http_url = _derive_deepgram_urls(base_url)
    assert ws_url == expected_ws
    assert http_url == expected_http


def test_derive_deepgram_urls_unknown_scheme_warns():
    sink = io.StringIO()
    handler_id = logger.add(sink, format="{message}")
    try:
        ws_url, http_url = _derive_deepgram_urls("ftp://mydeepgram.com")
        # Falls back to secure
        assert ws_url == "wss://mydeepgram.com"
        assert http_url == "https://mydeepgram.com"
        assert "Unrecognized scheme" in sink.getvalue()
    finally:
        logger.remove(handler_id)


@pytest.mark.asyncio
async def test_run_stt_send_media_exception_clears_connection():
    """send_media() failure should log a warning and clear self._connection."""
    service = DeepgramSTTService.__new__(DeepgramSTTService)
    service._name = "DeepgramSTTService"

    mock_connection = MagicMock()
    mock_connection.send_media = AsyncMock(side_effect=Exception("websocket closed"))
    service._connection = mock_connection

    sink = io.StringIO()
    handler_id = logger.add(sink, format="{message}")
    try:
        async for _ in service.run_stt(b"\x00" * 160):
            pass

        assert service._connection is None
        assert "send_media failed" in sink.getvalue()
    finally:
        logger.remove(handler_id)


@pytest.mark.asyncio
async def test_run_stt_skips_send_when_connection_is_none():
    """When self._connection is None, run_stt should silently skip."""
    service = DeepgramSTTService.__new__(DeepgramSTTService)
    service._connection = None

    # Should not raise
    async for _ in service.run_stt(b"\x00" * 160):
        pass

    assert service._connection is None


@pytest.mark.asyncio
async def test_connection_handler_gives_up_immediately_on_4xx_api_error():
    """A 4xx ApiError (e.g. invalid API key) should stop retrying after a
    single attempt and report the error."""
    service = _make_bare_service()
    mock_client = MagicMock()
    mock_client.listen.v1.connect = MagicMock(
        return_value=_failing_connect_cm(ApiError(status_code=401, body="invalid credentials"))
    )
    service._client = mock_client

    await service._connection_handler()

    assert mock_client.listen.v1.connect.call_count == 1
    service.push_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_connection_handler_gives_up_after_max_quick_failures(monkeypatch):
    """Repeated fast failures (e.g. network errors) should stop retrying after
    max_consecutive_failures in a row, with backoff between attempts."""
    monkeypatch.setattr("pipecat.services.deepgram.stt.exponential_backoff_time", lambda attempt: 0)
    service = _make_bare_service()
    max_failures = service._quick_failure_tracker.max_consecutive_failures
    mock_client = MagicMock()
    mock_client.listen.v1.connect = MagicMock(
        side_effect=[_failing_connect_cm(ConnectionError("boom")) for _ in range(max_failures)]
    )
    service._client = mock_client

    await service._connection_handler()

    assert mock_client.listen.v1.connect.call_count == max_failures
    # One push_error per failed attempt, plus a final give-up error.
    assert service.push_error.await_count == max_failures + 1


@pytest.mark.asyncio
async def test_connection_handler_resets_quick_failure_count_after_stable_connection(
    monkeypatch,
):
    """A connection that stays up longer than min_stable_duration should reset
    the quick-failure counter, so a prior near-miss doesn't count against the
    next round of failures."""
    monkeypatch.setattr("pipecat.services.deepgram.stt.exponential_backoff_time", lambda attempt: 0)
    service = _make_bare_service()
    # Simulate having already accumulated near-cap quick failures before a
    # stable connection came up.
    service._quick_failure_tracker.count = (
        service._quick_failure_tracker.max_consecutive_failures - 1
    )

    # Patch the module-level `time` name binding (not the real `time` module,
    # which asyncio's own event loop clock relies on). Only the attempt that
    # connects is timed, from the handshake to the drop.
    monotonic_values = iter([0, 10])
    fake_time = MagicMock()
    fake_time.monotonic.side_effect = lambda: next(monotonic_values)
    monkeypatch.setattr("pipecat.services.deepgram.stt.time", fake_time)

    mock_client = MagicMock()
    mock_client.listen.v1.connect = MagicMock(
        side_effect=[
            _dropping_connect_cm(ConnectionError("stable then dropped")),
            _failing_connect_cm(ConnectionError("quick 1")),
            _failing_connect_cm(ConnectionError("quick 2")),
            _failing_connect_cm(ConnectionError("quick 3")),
        ]
    )
    service._client = mock_client

    await service._connection_handler()

    # If the counter had NOT been reset after the stable connection, giving up
    # would have happened after just 1 more quick failure (2 total attempts).
    assert mock_client.listen.v1.connect.call_count == 4


@pytest.mark.asyncio
async def test_connection_handler_gives_up_on_handshakes_that_fail_slowly(monkeypatch):
    """A handshake that hangs before failing is a failure, however long it took.

    Timing the attempt rather than the connection reads these as healthy and
    retries them forever.
    """
    monkeypatch.setattr("pipecat.services.deepgram.stt.exponential_backoff_time", lambda attempt: 0)
    service = _make_bare_service()
    max_failures = service._quick_failure_tracker.max_consecutive_failures

    # Every attempt takes far longer than min_stable_duration before failing.
    ticks = iter([0, 10, 10, 20, 20, 30, 30, 40, 40, 50])
    fake_time = MagicMock()
    fake_time.monotonic.side_effect = lambda: next(ticks)
    monkeypatch.setattr("pipecat.services.deepgram.stt.time", fake_time)

    mock_client = MagicMock()
    mock_client.listen.v1.connect = MagicMock(
        side_effect=[_failing_connect_cm(ConnectionError("timed out")) for _ in range(8)]
    )
    service._client = mock_client

    await service._connection_handler()

    assert mock_client.listen.v1.connect.call_count == max_failures


@pytest.mark.asyncio
async def test_connect_returns_once_the_connection_is_given_up_on():
    """Connecting happens while the service is set up, so a connection that is
    never going to come up has to finish setting up rather than hold it open."""
    service = _make_bare_service()
    mock_client = MagicMock()
    mock_client.listen.v1.connect = MagicMock(
        return_value=_failing_connect_cm(ApiError(status_code=401, body="invalid credentials"))
    )
    service._client = mock_client

    await asyncio.wait_for(service._connect(), timeout=5)

    assert service._connection is None


@pytest.mark.asyncio
async def test_connection_handler_backs_off_after_non_quick_failure(monkeypatch):
    """A failure that isn't a quick failure (lasted >= min_stable_duration)
    must still back off before retrying, instead of busy-looping with no delay."""
    sleep_calls = []

    async def fake_sleep(duration):
        sleep_calls.append(duration)
        if len(sleep_calls) >= 2:
            # Stand in for the task being cancelled, e.g. by _disconnect(),
            # so the `while True` loop under test terminates.
            raise asyncio.CancelledError

    monkeypatch.setattr("pipecat.services.deepgram.stt.asyncio.sleep", fake_sleep)
    service = _make_bare_service()

    fake_time = MagicMock()
    # Each attempt "lasts" 10s (>= min_stable_duration), so is never a quick failure.
    times = iter([0, 10, 10, 20, 20, 30])
    fake_time.monotonic.side_effect = lambda: next(times)
    monkeypatch.setattr("pipecat.services.deepgram.stt.time", fake_time)

    mock_client = MagicMock()
    mock_client.listen.v1.connect = MagicMock(
        side_effect=[
            _failing_connect_cm(ConnectionError("drop 1")),
            _failing_connect_cm(ConnectionError("drop 2")),
            _failing_connect_cm(ConnectionError("drop 3")),
        ]
    )
    service._client = mock_client

    with contextlib.suppress(asyncio.CancelledError):
        await service._connection_handler()

    assert sleep_calls == [4, 4]  # exponential_backoff_time's min_wait, not skipped


def _results_message(transcript: str, is_final: bool):
    from deepgram.listen.v1.types import ListenV1Results

    return ListenV1Results.model_validate(
        {
            "type": "Results",
            "channel_index": [0, 1],
            "duration": 1.2,
            "start": 0.0,
            "is_final": is_final,
            "speech_final": is_final,
            "channel": {
                "alternatives": [{"transcript": transcript, "confidence": 0.99, "words": []}]
            },
            "metadata": {
                "request_id": "req-123",
                "model_info": {"name": "n", "version": "v", "arch": "a"},
                "model_uuid": "u",
            },
        }
    )


@pytest.mark.asyncio
async def test_final_transcript_emits_usage_before_transcription_frame(monkeypatch):
    from pipecat.frames.frames import InterimTranscriptionFrame, MetricsFrame, TranscriptionFrame
    from pipecat.metrics.metrics import STTUsageMetricsData

    service = DeepgramSTTService(api_key="test-key")
    service._setup = frame_processor_setup(TaskManager(), enable_usage_metrics=True)
    pushed_frames = []

    async def fake_push_frame(frame, direction=None):
        pushed_frames.append(frame)

    monkeypatch.setattr(service, "push_frame", fake_push_frame)

    # Simulate audio previously submitted to the service.
    service._stt_usage_pending_seconds = 1.25

    # Interim results must not emit usage.
    await service._on_message(_results_message("hello", is_final=False))
    assert [type(f) for f in pushed_frames] == [InterimTranscriptionFrame]

    # A final transcript emits usage before the TranscriptionFrame so tracing
    # can attach it to the span the frame closes.
    await service._on_message(_results_message("hello world", is_final=True))

    frame_types = [type(f) for f in pushed_frames]
    assert frame_types == [InterimTranscriptionFrame, MetricsFrame, TranscriptionFrame]

    data = pushed_frames[1].data[0]
    assert isinstance(data, STTUsageMetricsData)
    assert data.value.audio_seconds == 1.25
    assert service._stt_usage_pending_seconds == 0.0


@pytest.mark.asyncio
async def test_connection_handler_does_not_reconnect_after_cancel():
    """A cancelled ``_connection_handler`` must die, not loop and reconnect.

    ``_connection_handler`` is a ``while True`` reconnect loop whose ``finally``
    block awaits ``cancel_task(keepalive_task)``. Pipeline teardown can cancel
    the connection task while it is suspended in that ``finally`` — right after
    a mid-call network drop, with the keepalive blocked on the dead socket —
    and a handler that survived would reconnect to Deepgram unsupervised.

    The service is wired to a real ``TaskManager`` by hand so that the handler
    runs its own ``finally`` against the real ``cancel_task``; the fake SDK
    client keeps the network out of it.
    """
    task_manager = TaskManager(loop=asyncio.get_running_loop())

    service = DeepgramSTTService(api_key="fake-key-offline-test")
    service.create_task = lambda coro, name="deepgram-test": task_manager.create_task(coro, name)
    service.cancel_task = task_manager.cancel_task

    drop_event = asyncio.Event()
    connect_calls = 0

    class FakeConnection:
        def __init__(self, drops: bool):
            self._drops = drops

        def on(self, *args, **kwargs):
            pass

        async def start_listening(self):
            if self._drops:
                await drop_event.wait()
                raise ConnectionError("simulated mid-call network drop")
            await asyncio.Event().wait()  # reconnected socket: idle forever

        async def send_close_stream(self, *args, **kwargs):
            pass

        async def send_keep_alive(self, *args, **kwargs):
            pass

    def fake_connect(**kwargs):
        nonlocal connect_calls
        connect_calls += 1
        connection = FakeConnection(drops=connect_calls == 1)

        @asynccontextmanager
        async def cm():
            yield connection

        return cm()

    service._client = SimpleNamespace(
        listen=SimpleNamespace(v1=SimpleNamespace(connect=fake_connect))
    )

    # Keepalive whose cancellation takes a while to complete — models the
    # real keepalive blocked mid ``send_keep_alive()`` on a just-dropped
    # socket. This holds the handler inside its finally's
    # ``await cancel_task(keepalive_task)``, the window where the race lands.
    keepalive_cancel_delivered = asyncio.Event()
    release_keepalive_cleanup = asyncio.Event()

    async def stubborn_keepalive():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            keepalive_cancel_delivered.set()
            await release_keepalive_cleanup.wait()
            raise

    service._keepalive_handler = stubborn_keepalive

    await service._connect()  # spawns the real _connection_handler
    connection_task = service._connection_task
    await asyncio.sleep(0.05)  # handler inside start_listening, keepalive parked
    assert connect_calls == 1

    # 1. The connection drops mid-call: handler enters `except Exception`,
    #    then `finally`, and suspends at `await cancel_task(keepalive_task)`.
    drop_event.set()
    await keepalive_cancel_delivered.wait()

    # 2. Pipeline teardown cancels the connection task in that exact window.
    connection_task.cancel()
    await asyncio.sleep(0.05)

    try:
        assert connection_task.cancelled() or connection_task.done(), (
            "connection handler survived an explicit cancel: its own "
            "CancelledError was swallowed inside the finally's cancel_task"
        )
        assert connect_calls == 1, (
            f"connection handler RECONNECTED after being cancelled "
            f"(connect_calls={connect_calls}) — orphaned-reconnect zombie"
        )
    finally:
        release_keepalive_cleanup.set()
        connection_task.cancel()
        await asyncio.gather(connection_task, return_exceptions=True)
        remaining = list(task_manager.current_tasks())
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
