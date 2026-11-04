#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Connection handling of Inworld WebSocket TTS against a local fake Inworld server.

Each test speaks a few ``TTSSpeakFrame`` turns. The fake server can drop a
connection after a turn and delay handshakes, and a send can be made to stall
the way one does once the peer stops reading, which ``ws_send_timeout`` turns
into a dropped connection.
"""

import asyncio
import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import AsyncMock

import pytest
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed, ConnectionClosedError

from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.services.inworld.tts import InworldTTSService
from pipecat.tests.utils import SleepFrame, run_test

TURNS = ["Hello there.", "How are you?", "Goodbye."]


@dataclass
class _FakeInworld:
    # Abort the first connection once this many of its contexts have closed.
    drop_after_turns: int | None = None
    # Delay every handshake after the first.
    handshake_delay: float = 0.0
    handshakes: int = 0
    messages_per_connection: list[list[dict]] = field(default_factory=list)

    async def process_request(self, connection, request):
        self.handshakes += 1
        if self.handshakes > 1:
            await asyncio.sleep(self.handshake_delay)

    async def handle(self, websocket):
        index = len(self.messages_per_connection)
        messages: list[dict] = []
        self.messages_per_connection.append(messages)
        contexts: set[str] = set()
        try:
            await self._serve(websocket, index, messages, contexts)
        except ConnectionClosed:
            pass

    async def _serve(self, websocket, index: int, messages: list[dict], contexts: set[str]):
        closed = 0
        async for raw in websocket:
            message = json.loads(raw)
            messages.append(message)
            context_id = message.get("contextId")
            if "create" in message:
                contexts.add(context_id)
                await self._send(websocket, context_id, contextCreated={})
            elif "send_text" in message:
                if context_id is None:
                    await self._send_status(websocket, None, 3, "context_id is required")
                elif context_id not in contexts:
                    await self._send_status(websocket, context_id, 5, "Context not found")
                elif message["send_text"]["text"]:
                    audio = base64.b64encode(b"\x00\x00" * 480).decode()
                    await self._send(websocket, context_id, audioChunk={"audioContent": audio})
            elif "flush_context" in message:
                await self._send(websocket, context_id, flushCompleted={})
            elif "close_context" in message:
                contexts.discard(context_id)
                await self._send(websocket, context_id, contextClosed={})
                closed += 1
                if index == 0 and closed == self.drop_after_turns:
                    websocket.transport.abort()
                    return

    @staticmethod
    async def _send(websocket, context_id: str | None, **result):
        await websocket.send(json.dumps({"result": {"contextId": context_id, **result}}))

    @staticmethod
    async def _send_status(websocket, context_id: str | None, code: int, message: str):
        await _FakeInworld._send(websocket, context_id, status={"code": code, "message": message})

    def texts_sent(self, index: int) -> list[str]:
        return [
            m["send_text"]["text"].strip()
            for m in self.messages_per_connection[index]
            if "send_text" in m and m["send_text"]["text"]
        ]


def _stall_sends(monkeypatch, should_stall: Callable[[dict], bool]):
    """Make matching sends wait as they do once the peer stops reading."""
    send = ClientConnection.send
    stalled = []

    async def stalling_send(self, message, *args, **kwargs):
        if isinstance(message, str) and not stalled and should_stall(json.loads(message)):
            stalled.append(message)
            await asyncio.Event().wait()
        await send(self, message, *args, **kwargs)

    monkeypatch.setattr(ClientConnection, "send", stalling_send)


def _sends_text(text: str) -> Callable[[dict], bool]:
    return lambda m: m.get("send_text", {}).get("text", "").strip() == text


def _fast_failures(service: InworldTTSService):
    service._ws_send_timeout = 0.1
    service._reconnect_backoff_min_wait = 0.05
    service._reconnect_backoff_max_wait = 0.05


async def _run(server: _FakeInworld, turns: list[str], configure=None):
    async with serve(server.handle, "127.0.0.1", 0, process_request=server.process_request) as s:
        service = InworldTTSService(
            api_key="test-key", url=f"ws://127.0.0.1:{s.sockets[0].getsockname()[1]}"
        )
        if configure:
            configure(service)
        frames = []
        for text in turns:
            frames += [TTSSpeakFrame(text), SleepFrame(sleep=0.5)]
        down, up = await run_test(service, frames_to_send=frames)
    return service, down, up


def _turns_with_audio(down) -> int:
    return len({f.context_id for f in down if isinstance(f, TTSAudioRawFrame)})


@pytest.mark.asyncio
async def test_a_dropped_connection_is_replaced_before_the_next_turn():
    server = _FakeInworld(drop_after_turns=1)

    _, down, _ = await _run(server, TURNS[:2])

    assert server.handshakes == 2
    assert server.texts_sent(1) == [TURNS[1]]
    assert _turns_with_audio(down) == 2


@pytest.mark.asyncio
async def test_a_stalled_send_fails_its_turn_and_the_next_turn_is_spoken(monkeypatch):
    _stall_sends(monkeypatch, _sends_text(TURNS[1]))
    server = _FakeInworld()

    service, down, up = await _run(server, TURNS, configure=_fast_failures)

    assert server.handshakes == 2
    assert server.texts_sent(1) == [TURNS[2]]
    assert _turns_with_audio(down) == 2
    assert any(isinstance(f, ErrorFrame) for f in up)
    assert service.is_usable


@pytest.mark.asyncio
async def test_a_turn_starting_while_reconnecting_does_not_open_a_second_connection(
    monkeypatch,
):
    # The second turn's context is created eagerly, as soon as the turn starts.
    sent_creates = []

    def second_create(message):
        if "create" in message:
            sent_creates.append(message)
        return len(sent_creates) == 2

    _stall_sends(monkeypatch, second_create)
    server = _FakeInworld(handshake_delay=0.2)

    _, down, up = await _run(server, TURNS, configure=_fast_failures)

    assert server.handshakes == 2
    assert server.texts_sent(1) == [TURNS[2]]
    assert _turns_with_audio(down) == 2
    assert any(isinstance(f, ErrorFrame) and "reconnecting" in f.error for f in up)


@pytest.mark.asyncio
async def test_a_context_whose_create_failed_is_created_again():
    service = InworldTTSService(api_key="test-key")
    service._websocket = AsyncMock()
    service._websocket.send.side_effect = [ConnectionClosedError(None, None), None]

    with pytest.raises(ConnectionClosedError):
        await service._send_context("context")
    await service._send_context("context")
    await service._send_context("context")

    assert service._websocket.send.await_count == 2
