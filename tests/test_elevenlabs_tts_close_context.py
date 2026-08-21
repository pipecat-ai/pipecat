#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ElevenLabsTTSService end-of-turn close_context race handling.

When ElevenLabs synthesis is slow, closing the context at end of turn lets the
server's `isFinal` ack overtake the first audio chunk and drop the whole
utterance. These tests pin the deferral behavior that avoids that: the close
is held until the first audio chunk arrives, and turns that never open a
context (function-call-only turns) send no close at all.
"""

import base64
import json

import pytest
from websockets.protocol import State

from pipecat.services.elevenlabs.tts import ElevenLabsTTSService


class _FakeWebSocket:
    """Stand-in for the ElevenLabs websocket: records sends, replays messages."""

    def __init__(self, messages: list[str] | None = None):
        self.state = State.OPEN
        self.sent: list[dict] = []
        self._messages = messages or []

    async def send(self, data: str):
        self.sent.append(json.loads(data))

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for message in self._messages:
            yield message


def _make_service(**kwargs) -> ElevenLabsTTSService:
    return ElevenLabsTTSService(
        api_key="test-key",
        settings=ElevenLabsTTSService.Settings(voice="test-voice"),
        **kwargs,
    )


def _audio_message(context_id: str) -> str:
    audio = base64.b64encode(b"\x00\x01" * 32).decode()
    return json.dumps({"audio": audio, "contextId": context_id})


def _close_messages(ws: _FakeWebSocket) -> list[dict]:
    return [m for m in ws.sent if m.get("close_context") is True]


@pytest.mark.asyncio
async def test_close_deferred_when_no_audio_yet():
    """End of turn with no audio yet defers the close instead of sending it."""
    service = _make_service()
    ws = _FakeWebSocket()
    service._websocket = ws
    await service.create_audio_context("ctx-1")
    service._turn_context_id = "ctx-1"

    await service.on_turn_context_completed()

    assert _close_messages(ws) == []
    assert "ctx-1" in service._contexts_pending_close


@pytest.mark.asyncio
async def test_close_immediate_when_audio_already_streaming():
    """End of turn after audio has started closes immediately (unchanged path)."""
    service = _make_service()
    ws = _FakeWebSocket()
    service._websocket = ws
    await service.create_audio_context("ctx-1")
    service._turn_context_id = "ctx-1"
    service._contexts_with_first_audio.add("ctx-1")

    await service.on_turn_context_completed()

    assert _close_messages(ws) == [{"context_id": "ctx-1", "close_context": True}]
    assert "ctx-1" not in service._contexts_pending_close


@pytest.mark.asyncio
async def test_no_close_sent_for_function_call_only_turn():
    """A turn that never sends text never opens a context, so no close is sent."""
    service = _make_service()
    ws = _FakeWebSocket()
    service._websocket = ws
    # No create_audio_context call: text never reached run_tts for this turn.
    service._turn_context_id = "ctx-1"

    await service.on_turn_context_completed()

    assert ws.sent == []
    assert "ctx-1" not in service._contexts_pending_close


@pytest.mark.asyncio
async def test_deferred_close_sent_on_first_audio_chunk():
    """The deferred close fires once the first audio chunk arrives."""
    service = _make_service()
    ws = _FakeWebSocket([_audio_message("ctx-1")])
    service._websocket = ws
    await service.create_audio_context("ctx-1")
    service._turn_context_id = "ctx-1"

    await service.on_turn_context_completed()
    assert "ctx-1" in service._contexts_pending_close
    assert _close_messages(ws) == []

    await service._receive_messages()

    assert _close_messages(ws) == [{"context_id": "ctx-1", "close_context": True}]
    assert "ctx-1" not in service._contexts_pending_close
    assert "ctx-1" in service._contexts_with_first_audio


@pytest.mark.asyncio
async def test_interruption_clears_pending_close():
    """A barge-in drops deferred state so a late chunk can't resurrect the context."""
    service = _make_service()
    ws = _FakeWebSocket()
    service._websocket = ws
    await service.create_audio_context("ctx-1")
    service._contexts_pending_close.add("ctx-1")

    await service.on_audio_context_interrupted("ctx-1")

    assert "ctx-1" not in service._contexts_pending_close
    assert "ctx-1" not in service._contexts_with_first_audio


@pytest.mark.asyncio
async def test_disconnect_clears_deferred_close_state():
    """Disconnecting drops any lingering per-context tracking so it can't leak."""
    service = _make_service()
    service._websocket = None
    service._contexts_pending_close.add("ctx-1")
    service._contexts_with_first_audio.add("ctx-1")

    await service._disconnect_websocket()

    assert service._contexts_pending_close == set()
    assert service._contexts_with_first_audio == set()


# ---------------------------------------------------------------------------
# Local stop_frame_timeout_s idle timeout vs. deferred close
#
# A context's local audio queue can also reach completion by timing out
# (TTSService._handle_audio_context, e.g. slow synthesis under load) rather
# than via a server isFinal ack. If a close was deferred at turn end and no
# audio ever arrived to trigger it, the timeout path must still send it, or
# the server-side context leaks indefinitely.
# ---------------------------------------------------------------------------


async def _run_local_timeout(service: ElevenLabsTTSService, context_id: str):
    """Drive the real base-class idle-timeout path for a context.

    Mirrors the relevant part of TTSService._audio_context_task_handler: run
    the real _handle_audio_context (which times out on an empty queue after
    stop_frame_timeout_s) and then the real on_audio_context_completed, in
    the same order the base class uses.
    """
    await service._handle_audio_context(context_id)
    del service._audio_contexts[context_id]
    await service.on_audio_context_completed(context_id=context_id)


@pytest.mark.asyncio
async def test_local_timeout_sends_deferred_close_exactly_once():
    """No audio within stop_frame_timeout_s still gets its close sent, via completion."""
    service = _make_service(stop_frame_timeout_s=0.05)
    ws = _FakeWebSocket()
    service._websocket = ws
    await service.create_audio_context("ctx-1")
    service._turn_context_id = "ctx-1"

    await service.on_turn_context_completed()
    assert "ctx-1" in service._contexts_pending_close
    assert _close_messages(ws) == []

    await _run_local_timeout(service, "ctx-1")

    assert _close_messages(ws) == [{"context_id": "ctx-1", "close_context": True}]
    assert "ctx-1" not in service._contexts_pending_close
    assert "ctx-1" not in service._contexts_with_first_audio


@pytest.mark.asyncio
async def test_completion_after_first_audio_close_sends_no_second_close():
    """A context closed via the first-audio path doesn't get a second close on completion."""
    service = _make_service(stop_frame_timeout_s=0.05)
    ws = _FakeWebSocket([_audio_message("ctx-1")])
    service._websocket = ws
    await service.create_audio_context("ctx-1")
    service._turn_context_id = "ctx-1"

    await service.on_turn_context_completed()
    await service._receive_messages()
    assert _close_messages(ws) == [{"context_id": "ctx-1", "close_context": True}]

    # The queue reaches completion normally (e.g. the server's isFinal ack
    # drained it) rather than via the idle timeout -- either way,
    # on_audio_context_completed must not re-close an already-closed context.
    del service._audio_contexts["ctx-1"]
    await service.on_audio_context_completed(context_id="ctx-1")

    assert _close_messages(ws) == [{"context_id": "ctx-1", "close_context": True}]


@pytest.mark.asyncio
async def test_late_audio_after_timeout_close_is_dropped_not_resurrected():
    """Audio arriving after the timeout-triggered close is dropped, not recreated.

    _turn_context_id is already None by the time a context reaches
    _contexts_pending_close (on_turn_context_completed resets it before
    deferring), so append_to_audio_context's recreate-on-append branch can
    never match a pending-close context: late audio for it is silently
    dropped rather than resurrecting a context we've already told the server
    to close.
    """
    service = _make_service(stop_frame_timeout_s=0.05)
    ws = _FakeWebSocket()
    service._websocket = ws
    await service.create_audio_context("ctx-1")
    service._turn_context_id = "ctx-1"

    await service.on_turn_context_completed()
    assert service._turn_context_id is None

    await _run_local_timeout(service, "ctx-1")
    assert _close_messages(ws) == [{"context_id": "ctx-1", "close_context": True}]

    # Late audio for the now-closed context arrives on the wire.
    service._websocket = _FakeWebSocket([_audio_message("ctx-1")])
    await service._receive_messages()

    assert "ctx-1" not in service._audio_contexts
    assert _close_messages(service._websocket) == []
