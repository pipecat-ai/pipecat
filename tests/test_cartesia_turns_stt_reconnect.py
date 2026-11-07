#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Connection handling of Cartesia Turns STT against a local fake Ink-2 server.

The service runs in a test pipeline fed with real-time 20 ms audio frames. The
fake server can reject handshakes, drop a connection, reject the connection's
settings, or never confirm one with ``connected``.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from http import HTTPStatus

import pytest
from websockets.asyncio.server import serve
from websockets.frames import CloseCode

from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    MetricsFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.metrics.metrics import STTUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.cartesia.turns.stt import CartesiaTurnsSTTService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.errors import ErrorCategory

FRAME_SECONDS = 0.02


@dataclass
class _FakeInk2:
    rejected_handshakes: int = 0
    reject_after: int | None = None
    drop_first_after: int | None = None
    # Open a turn on the first connection just before dropping it.
    drop_mid_turn: bool = False
    # Continue that turn on the second connection: the rest of what the user says.
    resume_turn: bool = False
    # Open a turn on the first connection after this many audio chunks; the
    # server ends it when the service sends close.
    open_turn_after: int | None = None
    # How many connections, from the first, never get connected; None for all.
    unconfirmed: int | None = 0
    # Reject every connection's settings after the handshake.
    reject_settings: bool = False
    handshakes: int = 0
    audio_per_connection: list[int] = field(default_factory=list)

    async def process_request(self, connection, request):
        self.handshakes += 1
        rejected = self.handshakes <= self.rejected_handshakes or (
            self.reject_after is not None and self.handshakes > self.reject_after
        )
        if rejected:
            return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "unavailable\n")
        return None

    async def handle(self, websocket):
        index = len(self.audio_per_connection)
        self.audio_per_connection.append(0)
        if self.reject_settings:
            error = {
                "type": "error",
                "error_code": "model_not_found",
                "message": "Could not resolve model.",
                "status_code": 400,
            }
            await websocket.send(json.dumps(error))
            await websocket.close(CloseCode.POLICY_VIOLATION, "Invalid model")
            return
        if self.unconfirmed is not None and index >= self.unconfirmed:
            await websocket.send(json.dumps({"type": "connected", "request_id": "test"}))
        turn_open = False
        async for message in websocket:
            if not isinstance(message, bytes):
                if json.loads(message).get("type") == "close":
                    if turn_open:
                        await self._send(websocket, "turn.end", "hello there")
                    await websocket.close()
                    return
                continue
            self.audio_per_connection[index] += 1
            count = self.audio_per_connection[index]
            if index == 0 and count == self.open_turn_after:
                await self._send(websocket, "turn.start", "hello")
                await self._send(websocket, "turn.update", "hello there")
                turn_open = True
            if index == 1 and self.resume_turn and count == 3:
                await self._send(websocket, "turn.start", "how are you")
                await self._send(websocket, "turn.end", "how are you")
            if index == 0 and count == self.drop_first_after:
                if self.drop_mid_turn:
                    await self._send(websocket, "turn.start", "hello")
                    await self._send(websocket, "turn.update", "hello there")
                    await asyncio.sleep(0.05)
                websocket.transport.abort()
                return

    @staticmethod
    async def _send(websocket, message_type: str, transcript: str):
        await websocket.send(json.dumps({"type": message_type, "transcript": transcript}))


def _audio_frames(seconds: float) -> list:
    frames = []
    for _ in range(int(seconds / FRAME_SECONDS)):
        frames.append(
            InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
        )
        frames.append(SleepFrame(sleep=FRAME_SECONDS))
    return frames


async def _run(server: _FakeInk2, seconds: float, configure=None):
    async with serve(server.handle, "127.0.0.1", 0, process_request=server.process_request) as s:
        service = CartesiaTurnsSTTService(
            api_key="test-key", url=f"ws://127.0.0.1:{s.sockets[0].getsockname()[1]}"
        )
        if configure:
            configure(service)
        down, up = await run_test(
            service,
            frames_to_send=_audio_frames(seconds),
            pipeline_params=PipelineParams(enable_usage_metrics=True),
        )
        # Give a connection still being served time to record what it received.
        await asyncio.sleep(0.1)
    return service, down, up


def _turn_ends(down) -> tuple[list[tuple[str, bool]], list[int]]:
    """The transcriptions pushed and the positions of proposed turn stops."""
    transcripts = [(f.text, f.finalized) for f in down if isinstance(f, TranscriptionFrame)]
    stops = [i for i, f in enumerate(down) if isinstance(f, ProposedUserStoppedSpeakingFrame)]
    return transcripts, stops


@pytest.mark.asyncio
async def test_a_failed_first_connect_is_retried_in_the_background():
    server = _FakeInk2(rejected_handshakes=1)

    service, _, _ = await _run(server, seconds=0.5)

    assert server.handshakes == 2
    assert server.audio_per_connection[0] > 0
    assert service.is_usable


@pytest.mark.asyncio
async def test_a_dropped_connection_is_replaced_by_one_new_connection():
    server = _FakeInk2(drop_first_after=10)

    await _run(server, seconds=1.0)

    assert len(server.audio_per_connection) == 2
    assert server.audio_per_connection[1] > 0


@pytest.mark.asyncio
async def test_a_turn_cut_short_by_a_drop_ends_with_its_transcript_so_far():
    server = _FakeInk2(drop_first_after=10, drop_mid_turn=True)

    _, down, _ = await _run(server, seconds=1.0)

    transcripts, stops = _turn_ends(down)
    assert transcripts == [("hello there", False)]
    assert len(stops) == 1
    first_transcript = next(i for i, f in enumerate(down) if isinstance(f, TranscriptionFrame))
    assert first_transcript < stops[0]
    # Usage is reported ahead of the transcript, for the STT span it closes.
    assert any(
        isinstance(f, MetricsFrame) and any(isinstance(d, STTUsageMetricsData) for d in f.data)
        for f in down[:first_transcript]
    )
    assert len(server.audio_per_connection) == 2


@pytest.mark.asyncio
async def test_speech_continuing_after_a_drop_becomes_a_new_turn():
    server = _FakeInk2(drop_first_after=10, drop_mid_turn=True, resume_turn=True)

    _, down, _ = await _run(server, seconds=1.0)

    transcripts, stops = _turn_ends(down)
    assert transcripts == [("hello there", False), ("how are you", True)]
    assert len(stops) == 2


@pytest.mark.asyncio
async def test_a_turn_ends_when_reconnecting_gives_up():
    def fast_backoff(service):
        service._reconnect_backoff_min_wait = 0.01
        service._reconnect_backoff_max_wait = 0.01

    server = _FakeInk2(drop_first_after=10, drop_mid_turn=True, reject_after=1)

    service, down, _ = await _run(server, seconds=1.0, configure=fast_backoff)

    transcripts, stops = _turn_ends(down)
    assert transcripts == [("hello there", False)]
    assert len(stops) == 1
    assert not service.is_usable


@pytest.mark.asyncio
async def test_stopping_mid_turn_lets_the_server_end_the_turn():
    server = _FakeInk2(open_turn_after=5)

    _, down, _ = await _run(server, seconds=0.4)

    transcripts, stops = _turn_ends(down)
    assert transcripts == [("hello there", True)]
    assert len(stops) == 1


@pytest.mark.asyncio
async def test_an_unconfirmed_connection_is_retried(monkeypatch):
    monkeypatch.setattr(CartesiaTurnsSTTService, "_CONNECTION_TIMEOUT", 0.1)
    server = _FakeInk2(unconfirmed=1)

    service, _, _ = await _run(server, seconds=0.5)

    assert server.handshakes == 2
    assert server.audio_per_connection[1] > 0
    assert service.is_usable


@pytest.mark.asyncio
async def test_connections_that_never_confirm_are_given_up_after_three(monkeypatch):
    monkeypatch.setattr(CartesiaTurnsSTTService, "_CONNECTION_TIMEOUT", 0.1)
    server = _FakeInk2(unconfirmed=None)

    def stable_before_timeout(service):
        # As in production, the wait for connected outlasts the time after
        # which a connection counts as stable rather than a quick failure.
        service._quick_failure_tracker.min_stable_duration = 0.05

    service, _, _ = await _run(server, seconds=1.0, configure=stable_before_timeout)

    assert server.handshakes == 3
    assert not service.is_usable


@pytest.mark.asyncio
async def test_rejected_settings_are_not_retried():
    server = _FakeInk2(reject_settings=True)

    service, _, up = await _run(server, seconds=0.5)

    assert server.handshakes == 1
    assert not service.is_usable
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert [e.category for e in errors] == [ErrorCategory.INVALID_REQUEST]
    assert "model_not_found" in errors[0].error


@pytest.mark.asyncio
async def test_audio_keeps_flowing_while_the_provider_is_down():
    # The connection drops and every reconnect is rejected. Reconnecting
    # happens in the background, so frames pass through the service on time
    # while each attempt fails and backs off.
    def long_backoff(service):
        service._reconnect_backoff_min_wait = 30
        service._reconnect_backoff_max_wait = 30

    server = _FakeInk2(drop_first_after=5, reject_after=1)

    start = time.monotonic()
    _, down, _ = await asyncio.wait_for(_run(server, seconds=0.5, configure=long_backoff), 10)

    assert time.monotonic() - start < 3
    assert sum(isinstance(f, InputAudioRawFrame) for f in down) == 25


@pytest.mark.asyncio
async def test_stopping_does_not_reconnect():
    server = _FakeInk2()

    await _run(server, seconds=0.3)
    await asyncio.sleep(0.3)

    assert server.handshakes == 1
