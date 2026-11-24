#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Soniox stream reconnect against a local fake Soniox server.

The fake server keeps the stream apart from its connections, as Soniox does: it
numbers the messages the client sends and the responses it sends back over the
whole stream, so a connection can drop and another can carry the stream on.
Every few audio messages it answers with a final transcript naming the audio
message count, so the transcripts show whether audio was lost or duplicated.
"""

import json
from dataclasses import dataclass, field

import pytest
from websockets.asyncio.server import ServerConnection, serve

import pipecat.services.soniox.stt as soniox_stt
from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame
from pipecat.services.soniox.stt import SonioxSTTService
from pipecat.tests.utils import SleepFrame, run_test

TRANSCRIPT_EVERY = 5
FRAME_SECONDS = 0.02


@dataclass
class _FakeSoniox:
    # Abort the first connection on this audio message, which is lost with it,
    # along with the response the message before it produced.
    drop_on_audio: int | None = None
    refuse_reconnect: bool = False
    answer_pings_on_first_connection: bool = True

    port: int = 0
    # The first message of each connection: "config" or "reconnect".
    connections: list[str] = field(default_factory=list)
    configs: list[dict] = field(default_factory=list)
    audio: int = 0
    received: int = 0
    pings: int = 0
    responses: list[str] = field(default_factory=list)
    token: int = 0
    dropped: bool = False

    def _response(self, text: str) -> str:
        tokens = [{"text": text, "is_final": True}, {"text": "<end>", "is_final": True}]
        return json.dumps({"tokens": tokens, "seq": len(self.responses) + 1})

    def _new_token(self) -> str:
        self.token += 1
        return f"token-{self.token}"

    async def handle(self, websocket: ServerConnection):
        first = json.loads(await websocket.recv())
        if first.get("type") == "reconnect":
            self.connections.append("reconnect")
            if self.refuse_reconnect or first["reconnect_token"] != f"token-{self.token}":
                await websocket.send(
                    json.dumps({"tokens": [], "error_code": 400, "error_type": "reconnect_failed"})
                )
                return
            reply = {
                "type": "reconnected",
                "reconnect_token": self._new_token(),
                "last_request": self.received,
            }
            await websocket.send(json.dumps(reply))
            for response in self.responses[first["last_seq"] :]:
                await websocket.send(response)
        else:
            self.connections.append("config")
            self.configs.append(first)
            self.audio = self.received = 0
            self.responses = []
            if first.get("enable_reconnect"):
                url = f"ws://127.0.0.1:{self.port}/transcribe-websocket?reconnect=true&r=x"
                offer = {"type": "reconnect", "reconnect_token": self._new_token()}
                await websocket.send(json.dumps({**offer, "reconnect_url": url}))

        first_connection = len(self.connections) == 1
        async for message in websocket:
            if isinstance(message, str) and message and json.loads(message)["type"] == "ping":
                self.pings += 1
                if self.answer_pings_on_first_connection or not first_connection:
                    pong = {"type": "pong", "id": json.loads(message)["id"]}
                    await websocket.send(json.dumps(pong))
                continue
            if (
                first_connection
                and isinstance(message, bytes)
                and self.audio + 1 == self.drop_on_audio
            ):
                self.dropped = True
                websocket.transport.abort()
                return
            self.received += 1
            if isinstance(message, bytes):
                self.audio += 1
                if self.audio % TRANSCRIPT_EVERY == 0:
                    response = self._response(f"a{self.audio}")
                    self.responses.append(response)
                    # The response before the drop is lost with the connection.
                    if not (first_connection and self.audio + 1 == self.drop_on_audio):
                        await websocket.send(response)


async def _run(server: _FakeSoniox, service_kwargs: dict, seconds: float):
    async with serve(server.handle, "127.0.0.1", 0) as ws_server:
        server.port = ws_server.sockets[0].getsockname()[1]
        service = SonioxSTTService(
            api_key="test-key",
            url=f"ws://127.0.0.1:{server.port}",
            settings=SonioxSTTService.Settings(**service_kwargs),
        )
        frames = []
        for _ in range(int(seconds / FRAME_SECONDS)):
            frames.append(
                InputAudioRawFrame(audio=b"\x00\x00" * 320, sample_rate=16000, num_channels=1)
            )
            frames.append(SleepFrame(sleep=FRAME_SECONDS))
        down, _ = await run_test(service, frames_to_send=frames)
    transcripts = [f.text for f in down if isinstance(f, TranscriptionFrame)]
    return service, transcripts, len(frames) // 2


def _expected(audio_messages: int) -> list[str]:
    return [f"a{n}" for n in range(TRANSCRIPT_EVERY, audio_messages + 1, TRANSCRIPT_EVERY)]


@pytest.mark.asyncio
async def test_a_dropped_connection_resumes_the_stream_without_losing_audio():
    server = _FakeSoniox(drop_on_audio=23)

    _, transcripts, sent = await _run(server, {"enable_reconnect": True}, seconds=1.5)

    assert server.dropped
    assert server.connections == ["config", "reconnect"]
    # The audio lost with the connection was resent, and nothing twice.
    assert server.audio == sent
    # The transcript lost with the connection came back on the new one, once.
    # Allow for a transcript still in flight when the pipeline ends.
    expected = _expected(sent)
    assert transcripts == expected[: len(transcripts)]
    assert len(transcripts) >= len(expected) - 1


@pytest.mark.asyncio
async def test_a_refused_reconnect_starts_a_new_stream():
    server = _FakeSoniox(drop_on_audio=23, refuse_reconnect=True)

    _, transcripts, _ = await _run(server, {"enable_reconnect": True}, seconds=1.5)

    assert server.connections == ["config", "reconnect", "config"]
    assert transcripts


@pytest.mark.asyncio
async def test_a_late_pong_moves_the_stream_to_a_new_connection(monkeypatch):
    monkeypatch.setattr(soniox_stt, "STREAM_PING_INTERVAL_S", 0.1)
    monkeypatch.setattr(soniox_stt, "STREAM_RECONNECT_TIMEOUT_S", 0.1)
    server = _FakeSoniox(answer_pings_on_first_connection=False)

    _, transcripts, sent = await _run(server, {"enable_reconnect": True}, seconds=1.0)

    assert server.connections[:2] == ["config", "reconnect"]
    assert server.audio == sent
    expected = _expected(sent)
    assert transcripts == expected[: len(transcripts)]


@pytest.mark.asyncio
async def test_pongs_release_the_messages_the_server_has(monkeypatch):
    monkeypatch.setattr(soniox_stt, "STREAM_PING_INTERVAL_S", 0.1)
    server = _FakeSoniox()
    held = []

    original_send = SonioxSTTService._send

    async def recording_send(self, message, description="send"):
        await original_send(self, message, description)
        if self._stream:
            held.append(len(self._stream.unconfirmed))

    monkeypatch.setattr(SonioxSTTService, "_send", recording_send)

    _, _, sent = await _run(server, {"enable_reconnect": True}, seconds=1.0)

    assert server.pings > 0
    assert max(held) < sent


@pytest.mark.asyncio
async def test_reconnect_is_off_by_default():
    server = _FakeSoniox(drop_on_audio=23)

    _, _, _ = await _run(server, {}, seconds=1.0)

    assert "enable_reconnect" not in server.configs[0]
    assert server.pings == 0
    # The dropped stream is replaced by a new one.
    assert server.connections == ["config", "config"]
