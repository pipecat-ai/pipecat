#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for how GeminiLiveLLMService reports connection and send failures.

The service must report a session that fails to open, a connection that keeps
failing and a failed send as errors the pipeline can act on, so
``ProcessorUnusablePolicy`` applies instead of the bot staying silent.
"""

from typing import Any

import pytest
from google.genai import errors
from google.genai.types import LiveConnectConfig

from pipecat.services.google.gemini_live.llm import (
    MAX_CONSECUTIVE_FAILURES,
    GeminiLiveLLMService,
)


class _FailingConnect:
    """Async context manager whose ``__aenter__`` fails like a rejected setup."""

    def __init__(self, error: Exception):
        self._error = error

    async def __aenter__(self):
        raise self._error

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, error: Exception):
        self.aio = self
        self.live = self
        self._error = error

    def connect(self, **kwargs):
        return _FailingConnect(self._error)


def _api_error() -> Exception:
    try:
        errors.APIError.raise_error(1008, "model not available for bidiGenerateContent", None)
    except errors.APIError as exc:
        return exc
    raise AssertionError("raise_error did not raise")


def _make_service(monkeypatch) -> tuple[GeminiLiveLLMService, list[dict[str, Any]], list[str]]:
    service = GeminiLiveLLMService(api_key="test-key")
    pushed: list[dict[str, Any]] = []
    reconnects: list[str] = []

    async def push_error(**kwargs):
        pushed.append(kwargs)

    async def reconnect():
        reconnects.append("reconnect")

    monkeypatch.setattr(service, "push_error", push_error)
    monkeypatch.setattr(service, "_reconnect", reconnect)
    return service, pushed, reconnects


@pytest.mark.asyncio
async def test_failure_to_open_session_is_retried_then_reported_as_permanent(monkeypatch):
    service, pushed, reconnects = _make_service(monkeypatch)
    error = _api_error()
    service._client = _FakeClient(error)

    for attempt in range(1, MAX_CONSECUTIVE_FAILURES + 1):
        await service._connection_task_handler(config=LiveConnectConfig())
        if attempt < MAX_CONSECUTIVE_FAILURES:
            assert pushed == []
            assert len(reconnects) == attempt

    assert len(reconnects) == MAX_CONSECUTIVE_FAILURES - 1
    assert len(pushed) == 1
    assert pushed[0]["exception"] is error
    assert pushed[0]["force_treat_as_permanent"] is True


@pytest.mark.asyncio
async def test_failure_to_open_session_while_disconnecting_is_ignored(monkeypatch):
    service, pushed, reconnects = _make_service(monkeypatch)
    service._client = _FakeClient(_api_error())
    service._disconnecting = True

    await service._connection_task_handler(config=LiveConnectConfig())

    assert pushed == []
    assert reconnects == []
    assert service._consecutive_failures == 0


@pytest.mark.asyncio
async def test_giving_up_after_max_failures_marks_the_service_unusable(monkeypatch):
    service, pushed, _ = _make_service(monkeypatch)
    error = _api_error()

    results = [
        await service._handle_connection_error(error) for _ in range(MAX_CONSECUTIVE_FAILURES)
    ]

    assert results == [True] * (MAX_CONSECUTIVE_FAILURES - 1) + [False]
    assert len(pushed) == 1
    assert pushed[0]["force_treat_as_permanent"] is True


@pytest.mark.asyncio
async def test_send_error_is_reported_as_permanent(monkeypatch):
    service, pushed, _ = _make_service(monkeypatch)
    service._session = object()
    error = RuntimeError("socket closed")

    await service._handle_send_error(error)

    assert len(pushed) == 1
    assert pushed[0]["exception"] is error
    assert pushed[0]["force_treat_as_permanent"] is True


@pytest.mark.asyncio
async def test_send_error_while_disconnecting_is_ignored(monkeypatch):
    service, pushed, _ = _make_service(monkeypatch)
    service._session = object()
    service._disconnecting = True

    await service._handle_send_error(RuntimeError("socket closed"))

    assert pushed == []
