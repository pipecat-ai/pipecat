#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for Gemini Live message dispatch ordering."""

import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

try:
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

    google_available = True
except Exception:
    google_available = False


class _SingleMessageTurn:
    def __init__(self, message):
        self._message = message
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        return self._message


class _SingleMessageSession:
    def __init__(self, message):
        self._message = message
        self._received = False

    def receive(self):
        if self._received:
            raise RuntimeError("stop test session")
        self._received = True
        return _SingleMessageTurn(self._message)


class _MockConnect:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


def _make_service():
    with patch.object(GeminiLiveLLMService, "create_client"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return GeminiLiveLLMService(api_key="test-key", model="test-model")


def _make_message(
    *,
    model_turn=None,
    turn_complete=False,
    output_transcription=None,
    grounding_metadata=None,
    usage_metadata=None,
):
    server_content = SimpleNamespace(
        interrupted=False,
        model_turn=model_turn,
        turn_complete=turn_complete,
        input_transcription=None,
        output_transcription=output_transcription,
        grounding_metadata=grounding_metadata,
    )
    return SimpleNamespace(
        server_content=server_content,
        usage_metadata=usage_metadata,
        tool_call=None,
        session_resumption_update=None,
    )


async def _run_single_message(service, message):
    session = _SingleMessageSession(message)
    service._client = SimpleNamespace(
        aio=SimpleNamespace(live=SimpleNamespace(connect=lambda **kwargs: _MockConnect(session)))
    )
    service._disconnecting = True
    await service._connection_task_handler(config=SimpleNamespace())


def _async_recorder(order, name):
    async def _record(message):
        order.append(name)

    return _record


@pytest.mark.asyncio
@pytest.mark.skipif(not google_available, reason="Google dependencies not installed")
async def test_output_transcription_is_processed_before_turn_complete():
    service = _make_service()
    order = []

    service._handle_msg_model_turn = AsyncMock(side_effect=_async_recorder(order, "model_turn"))
    service._handle_msg_output_transcription = AsyncMock(
        side_effect=_async_recorder(order, "output_transcription")
    )
    service._handle_msg_turn_complete = AsyncMock(
        side_effect=_async_recorder(order, "turn_complete")
    )
    service._handle_msg_usage_metadata = AsyncMock(
        side_effect=_async_recorder(order, "usage_metadata")
    )

    message = _make_message(
        model_turn=object(),
        output_transcription=object(),
        turn_complete=True,
        usage_metadata=object(),
    )
    await _run_single_message(service, message)

    assert order == ["model_turn", "output_transcription", "turn_complete", "usage_metadata"]


@pytest.mark.asyncio
@pytest.mark.skipif(not google_available, reason="Google dependencies not installed")
async def test_bundled_grounding_metadata_is_not_emitted_immediately():
    service = _make_service()
    order = []

    service._handle_msg_output_transcription = AsyncMock(
        side_effect=_async_recorder(order, "output_transcription")
    )
    service._handle_msg_grounding_metadata = AsyncMock(
        side_effect=_async_recorder(order, "grounding_metadata")
    )
    service._handle_msg_turn_complete = AsyncMock(
        side_effect=_async_recorder(order, "turn_complete")
    )
    service._handle_msg_usage_metadata = AsyncMock(
        side_effect=_async_recorder(order, "usage_metadata")
    )

    message = _make_message(
        output_transcription=object(),
        grounding_metadata=object(),
        turn_complete=True,
        usage_metadata=object(),
    )
    await _run_single_message(service, message)

    assert order == ["output_transcription", "turn_complete", "usage_metadata"]
    service._handle_msg_grounding_metadata.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.skipif(not google_available, reason="Google dependencies not installed")
async def test_standalone_grounding_metadata_is_still_emitted():
    service = _make_service()
    order = []

    service._handle_msg_grounding_metadata = AsyncMock(
        side_effect=_async_recorder(order, "grounding_metadata")
    )
    service._handle_msg_turn_complete = AsyncMock(
        side_effect=_async_recorder(order, "turn_complete")
    )
    service._handle_msg_usage_metadata = AsyncMock(
        side_effect=_async_recorder(order, "usage_metadata")
    )

    message = _make_message(
        grounding_metadata=object(),
        turn_complete=True,
        usage_metadata=object(),
    )
    await _run_single_message(service, message)

    assert order == ["grounding_metadata", "turn_complete", "usage_metadata"]
