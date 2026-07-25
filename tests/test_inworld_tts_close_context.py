#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for InworldTTSService end-of-turn close_context race handling.

When Inworld synthesis is slow, closing the context at end of turn lets the
server's ``contextClosed`` ack overtake the first ``audioChunk`` and drop the
whole utterance. These tests pin the deferral behavior that avoids that: the
close is held until the first chunk arrives, and a deferred-close context stays
recreatable so audio outliving the idle timeout still plays.
"""

import base64
import json
from unittest.mock import AsyncMock

import pytest

from pipecat.services.inworld.tts import InworldTTSService
from pipecat.services.tts_service import WebsocketTTSService


def _service() -> InworldTTSService:
    service = InworldTTSService(api_key="test-key", settings=InworldTTSService.Settings())
    service._websocket = AsyncMock()
    return service


class _FakeWebsocket:
    """Async-iterable websocket yielding preloaded messages, recording sends."""

    def __init__(self, messages):
        self._messages = messages
        self.sent = []

    async def send(self, message):
        self.sent.append(message)

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for m in self._messages:
            yield m


def _audio_chunk_message(context_id: str) -> str:
    audio = base64.b64encode(b"\x00\x01" * 32).decode()
    return json.dumps({"result": {"contextId": context_id, "audioChunk": {"audioContent": audio}}})


def _context_closed_message(context_id: str) -> str:
    return json.dumps({"result": {"contextId": context_id, "contextClosed": {}}})


@pytest.mark.asyncio
async def test_close_deferred_when_no_audio_yet():
    """End of turn with no audio yet defers the close instead of sending it."""
    service = _service()
    service._close_context = AsyncMock()
    service._turn_context_id = "c1"

    await service.on_turn_context_completed()

    service._close_context.assert_not_awaited()
    assert "c1" in service._contexts_pending_close


@pytest.mark.asyncio
async def test_close_immediate_when_audio_already_streaming():
    """End of turn after audio has started closes immediately (unchanged path)."""
    service = _service()
    service._close_context = AsyncMock()
    service._turn_context_id = "c1"
    service._contexts_with_first_audio.add("c1")

    await service.on_turn_context_completed()

    service._close_context.assert_awaited_once_with("c1")
    assert "c1" not in service._contexts_pending_close


@pytest.mark.asyncio
async def test_deferred_close_sent_on_first_audio_chunk():
    """The deferred close fires once the first audioChunk arrives."""
    service = _service()
    service._close_context = AsyncMock()
    service.append_to_audio_context = AsyncMock()
    service._turn_context_id = "c1"

    await service.on_turn_context_completed()
    assert "c1" in service._contexts_pending_close
    service._close_context.assert_not_awaited()

    service._get_websocket = lambda: _FakeWebsocket([_audio_chunk_message("c1")])
    await service._receive_messages()

    service._close_context.assert_awaited_once_with("c1")
    assert "c1" not in service._contexts_pending_close
    assert "c1" in service._contexts_with_first_audio


def test_deferred_close_context_is_recreatable():
    """A context with a pending close may be recreated even after the turn ends."""
    service = _service()
    service._turn_context_id = None
    service._contexts_pending_close.add("c1")

    assert service._can_recreate_audio_context("c1") is True
    assert service._can_recreate_audio_context("other") is False


def test_turn_context_still_recreatable_by_default():
    """The base recreate guard (current turn) is preserved through the override."""
    service = _service()
    service._turn_context_id = "c1"

    assert service._can_recreate_audio_context("c1") is True


@pytest.mark.asyncio
async def test_interruption_clears_pending_close():
    """A barge-in drops deferred state so a late chunk can't resurrect the context."""
    service = _service()
    service._close_context = AsyncMock()
    service._maybe_push_fallback_text = AsyncMock()
    service._contexts_pending_close.add("c1")
    service._contexts_with_first_audio.add("c1")

    await service.on_audio_context_interrupted("c1")

    assert "c1" not in service._contexts_pending_close
    assert "c1" not in service._contexts_with_first_audio
    assert service._can_recreate_audio_context("c1") is False


@pytest.mark.asyncio
async def test_disconnect_clears_deferred_state(monkeypatch):
    """Disconnect drops any lingering per-context tracking so it can't leak."""
    service = _service()
    service._receive_task = None
    service._keepalive_task = None
    service._contexts_pending_close.add("c1")
    service._contexts_with_first_audio.add("c1")
    monkeypatch.setattr(WebsocketTTSService, "_disconnect", AsyncMock())
    service._disconnect_websocket = AsyncMock()

    await service._disconnect()

    assert not service._contexts_pending_close
    assert not service._contexts_with_first_audio


@pytest.mark.asyncio
async def test_context_closed_clears_tracking_state():
    """Server-side contextClosed cleans up per-context tracking sets."""
    service = _service()
    service.append_to_audio_context = AsyncMock()
    service.remove_audio_context = AsyncMock()
    service._maybe_push_fallback_text = AsyncMock()
    service._contexts_pending_close.add("c1")
    service._contexts_with_first_audio.add("c1")

    service._get_websocket = lambda: _FakeWebsocket([_context_closed_message("c1")])
    await service._receive_messages()

    assert "c1" not in service._contexts_pending_close
    assert "c1" not in service._contexts_with_first_audio
