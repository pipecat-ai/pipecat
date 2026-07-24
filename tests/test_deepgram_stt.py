#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import contextlib
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from deepgram.core import ApiError
from loguru import logger

from pipecat.services.deepgram.stt import DeepgramSTTService, _derive_deepgram_urls
from pipecat.utils.network import QuickFailureTracker


def _make_bare_service() -> DeepgramSTTService:
    """Build a DeepgramSTTService without running __init__, wiring just enough
    for _connection_handler() to run: a real create_task/cancel_task pair (so
    the keepalive task is properly started and torn down) and mocked
    push_error/_build_connect_kwargs.
    """
    service = DeepgramSTTService.__new__(DeepgramSTTService)
    service._name = "DeepgramSTTService"
    service._connection = None
    service._connection_ready = asyncio.Event()
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
    single attempt and report a non-fatal error."""
    service = _make_bare_service()
    mock_client = MagicMock()
    mock_client.listen.v1.connect = MagicMock(
        return_value=_failing_connect_cm(ApiError(status_code=401, body="invalid credentials"))
    )
    service._client = mock_client

    await service._connection_handler()

    assert mock_client.listen.v1.connect.call_count == 1
    service.push_error.assert_awaited_once()
    _, kwargs = service.push_error.call_args
    assert kwargs.get("fatal", False) is False


@pytest.mark.asyncio
async def test_connection_handler_gives_up_after_max_quick_failures(monkeypatch):
    """Repeated fast failures (e.g. network errors) should stop retrying after
    max_consecutive_failures in a row, with backoff between attempts, and none
    of the reported errors should be fatal."""
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
    for call in service.push_error.await_args_list:
        assert call.kwargs.get("fatal", False) is False


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
    # which asyncio's own event loop clock relies on).
    monotonic_values = iter(
        [
            0,
            10,  # attempt 1: 10s elapsed -> stable, resets counter to 0
            10,
            10.1,  # attempt 2: quick failure -> count 1
            10.1,
            10.2,  # attempt 3: quick failure -> count 2
            10.2,
            10.3,  # attempt 4: quick failure -> count 3, give up
        ]
    )
    fake_time = MagicMock()
    fake_time.monotonic.side_effect = lambda: next(monotonic_values)
    monkeypatch.setattr("pipecat.services.deepgram.stt.time", fake_time)

    mock_client = MagicMock()
    mock_client.listen.v1.connect = MagicMock(
        side_effect=[
            _failing_connect_cm(ConnectionError("stable then dropped")),
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
