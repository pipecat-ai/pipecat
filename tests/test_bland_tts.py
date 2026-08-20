#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for BlandTTSService and BlandHttpTTSService."""

import asyncio
import io
import json
import struct
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest
import websockets
from aiohttp import web
from loguru import logger
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.bland.tts import BlandHttpTTSService, BlandTTSService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import SleepFrame, run_test

DEFAULT_VOICE_ID = "2f29fdbb-c55e-4add-9c7c-93437ebf379d"
OTHER_VOICE_ID = "c18a1cd5-91ef-4b06-841a-e58b8b487e8c"

AUDIO_CHUNK_1 = b"\x00\x01" * 512
AUDIO_CHUNK_2 = b"\x02\x03" * 512


def _pcm_bytes(num_samples: int = 4096) -> bytes:
    """Bare little-endian int16 PCM, which is what ``container: raw`` returns."""
    return struct.pack(f"<{num_samples}h", *(((i * 97) % 2000) - 1000 for i in range(num_samples)))


def _audio_of(frames) -> bytes:
    return b"".join(f.audio for f in frames if isinstance(f, TTSAudioRawFrame))


# --- /v2/tts/ws ----------------------------------------------------------------------


def _ws_server_handler(
    captured: dict,
    *,
    init_error: dict | None = None,
    turn_error: dict | None = None,
    end_reason: str = "complete",
    ready_encoding: str = "pcm_s16le",
    ready_sample_rate: int | None = None,
    acknowledge_init: bool = True,
):
    """Build a fake Bland realtime server following the documented turn flow."""

    async def handler(ws):
        captured["auth_header"] = ws.request.headers.get("Authorization")
        captured["sessions"] = captured.get("sessions", 0) + 1

        try:
            async for raw in ws:
                msg = json.loads(raw)
                captured["messages"].append(msg)
                msg_type = msg.get("type")

                if msg_type == "init":
                    if init_error is not None:
                        await ws.send(json.dumps({"type": "error", **init_error}))
                        await ws.close()
                        return
                    if not acknowledge_init:
                        continue
                    requested_rate = msg.get("audio", {}).get("sample_rate", 48000)
                    await ws.send(
                        json.dumps(
                            {
                                "type": "ready",
                                "session_id": "test-session",
                                "encoding": ready_encoding,
                                "sample_rate": (
                                    ready_sample_rate
                                    if ready_sample_rate is not None
                                    else requested_rate
                                ),
                            }
                        )
                    )
                elif msg_type == "speak":
                    context_id = msg["context_id"]
                    if turn_error is not None:
                        await ws.send(
                            json.dumps({"type": "error", "context_id": context_id, **turn_error})
                        )
                        continue
                    if context_id not in captured.setdefault("started_contexts", set()):
                        captured["started_contexts"].add(context_id)
                        await ws.send(
                            json.dumps({"type": "utterance_start", "context_id": context_id})
                        )
                elif msg_type == "end_of_turn":
                    context_id = msg["context_id"]
                    if turn_error is not None:
                        continue
                    await ws.send(AUDIO_CHUNK_1)
                    await ws.send(AUDIO_CHUNK_2)
                    await ws.send(
                        json.dumps(
                            {
                                "type": "utterance_end",
                                "context_id": context_id,
                                "reason": end_reason,
                                "frames": 2,
                                "duration_ms": 100,
                            }
                        )
                    )
                elif msg_type == "close":
                    await ws.send(json.dumps({"type": "done", "session_id": "test-session"}))
                    captured["done_sent"] = True
                    await ws.close()
        except websockets.ConnectionClosed:
            pass

    return handler


def _of_type(captured: dict, type: str) -> list[dict]:
    return [m for m in captured["messages"] if m.get("type") == type]


@pytest.mark.asyncio
async def test_bland_tts_protocol_roundtrip():
    """init/speak/end_of_turn are sent, and the turn's audio is emitted."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = BlandTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/tts/ws",
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello from Bland."), SleepFrame(sleep=0.3)],
        )

    frame_types = [type(frame) for frame in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSAudioRawFrame in frame_types
    assert TTSStoppedFrame in frame_types
    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)

    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert all(frame.sample_rate == 24000 for frame in audio_frames)
    assert all(frame.num_channels == 1 for frame in audio_frames)
    assert _audio_of(down_frames) == AUDIO_CHUNK_1 + AUDIO_CHUNK_2

    assert captured["auth_header"] == "Bearer test-key"
    init = _of_type(captured, "init")[0]
    assert init["voice"] == DEFAULT_VOICE_ID
    assert init["audio"] == {"encoding": "pcm_s16le", "sample_rate": 24000}
    assert "controls" not in init

    speak = _of_type(captured, "speak")[0]
    end_of_turn = _of_type(captured, "end_of_turn")[0]
    assert speak["text"] == "Hello from Bland."
    # the turn is ended under the id its deltas were sent with
    assert end_of_turn["context_id"] == speak["context_id"]


@pytest.mark.asyncio
async def test_bland_tts_token_streaming_sends_tokens_verbatim():
    """In the default TOKEN mode, LLM tokens map 1:1 to speak messages, unaltered."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = BlandTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/tts/ws",
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMTextFrame("Unbelieva"),
                LLMTextFrame("ble"),
                LLMTextFrame(" isn't it?"),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.3),
            ],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    assert any(isinstance(frame, TTSAudioRawFrame) for frame in down_frames)

    speaks = _of_type(captured, "speak")
    # Bland appends each delta verbatim, so an inserted space would split words.
    assert [m["text"] for m in speaks] == ["Unbelieva", "ble", " isn't it?"]
    # every delta of one response belongs to one turn, ended once
    assert len({m["context_id"] for m in speaks}) == 1
    assert len(_of_type(captured, "end_of_turn")) == 1


@pytest.mark.asyncio
async def test_bland_tts_sentence_mode_appends_trailing_space():
    """In SENTENCE mode a trailing space separates consecutive generations."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = BlandTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/tts/ws",
            sample_rate=24000,
            text_aggregation_mode=TextAggregationMode.SENTENCE,
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello from Bland."), SleepFrame(sleep=0.3)],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    assert _of_type(captured, "speak")[0]["text"] == "Hello from Bland. "


@pytest.mark.asyncio
async def test_bland_tts_init_carries_controls():
    """Voice and controls are fixed at init for the life of the session."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = BlandTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/tts/ws",
            sample_rate=24000,
            settings=BlandTTSService.Settings(
                voice=OTHER_VOICE_ID, expressiveness=0.9, stability=0.4
            ),
        )

        await run_test(tts, frames_to_send=[])

    init = _of_type(captured, "init")[0]
    assert init["voice"] == OTHER_VOICE_ID
    assert init["controls"] == {"expressiveness": 0.9, "stability": 0.4}


@pytest.mark.asyncio
async def test_bland_tts_partial_controls():
    """Only controls the caller set are sent, so unset ones keep Bland's defaults."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = BlandTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/tts/ws",
            sample_rate=24000,
            settings=BlandTTSService.Settings(stability=0.4),
        )

        await run_test(tts, frames_to_send=[])

    assert _of_type(captured, "init")[0]["controls"] == {"stability": 0.4}


@pytest.mark.asyncio
async def test_bland_tts_unsupported_pipeline_rate_falls_back():
    """A rate Bland cannot render is replaced by its native 48 kHz."""
    captured: dict = {"messages": []}
    sink = io.StringIO()
    handler_id = logger.add(sink, level="WARNING", format="{message}")

    try:
        async with serve(_ws_server_handler(captured), "127.0.0.1", 0) as server:
            host, port = next(iter(server.sockets)).getsockname()[:2]

            tts = BlandTTSService(
                api_key="test-key",
                url=f"ws://{host}:{port}/v2/tts/ws",
                sample_rate=22050,
            )

            down_frames, _ = await run_test(
                tts, frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.3)]
            )
    finally:
        logger.remove(handler_id)

    # The substitution is never silent: a pipeline running at a rate Bland
    # cannot render pays a resample, and the log says so.
    assert "22050" in sink.getvalue() and "48000" in sink.getvalue(), sink.getvalue()
    assert _of_type(captured, "init")[0]["audio"]["sample_rate"] == 48000
    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    # frames are tagged with the rate Bland actually rendered; the output
    # transport resamples to the pipeline rate
    assert all(frame.sample_rate == 48000 for frame in audio_frames)


@pytest.mark.asyncio
async def test_bland_tts_interruption_cancels_without_reconnecting():
    """Barge-in sends cancel, so the session and its warm voice survive."""
    tts = BlandTTSService(api_key="test-key", sample_rate=24000)

    websocket = AsyncMock()
    tts._websocket = websocket

    await tts.on_audio_context_interrupted("turn-17")

    sent = [json.loads(call.args[0]) for call in websocket.send.call_args_list]
    assert sent == [{"type": "cancel", "context_id": "turn-17"}]
    assert not websocket.close.called


@pytest.mark.asyncio
async def test_bland_tts_interruption_abandons_the_turn_locally():
    """A cancelled turn stops taking deltas without waiting for `utterance_end`."""
    tts = BlandTTSService(api_key="test-key", sample_rate=24000)

    websocket = AsyncMock()
    tts._websocket = websocket
    tts._sent_context_id = "turn-17"

    await tts.on_audio_context_interrupted("turn-17")
    websocket.send.reset_mock()

    async for _ in tts.run_tts("the tail nobody asked for", "turn-17"):
        pass

    # Feeding a cancelled turn has Bland admit and bill it afresh, and leaving it
    # in flight has a dying socket report it as a turn lost mid-sentence.
    assert not websocket.send.called
    assert tts._sent_context_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["insufficient_credits", "rate_limited"])
async def test_bland_tts_turn_error_surfaces(code):
    """A turn-scoped error frame becomes an ErrorFrame carrying code and message."""
    captured: dict = {"messages": []}
    error = {"code": code, "message": "Turn admission refused."}

    async with serve(_ws_server_handler(captured, turn_error=error), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = BlandTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/tts/ws",
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.3)]
        )

    errors = [f for f in down_frames + up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert code in errors[0].error
    assert "Turn admission refused." in errors[0].error
    assert tts.get_audio_contexts() == []


@pytest.mark.asyncio
async def test_bland_tts_failed_turn_surfaces():
    """A turn that ends as `failed` reports rather than hanging on missing audio."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured, end_reason="failed"), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = BlandTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/tts/ws",
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.3)]
        )

    errors = [f for f in down_frames + up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "failed" in errors[0].error
    assert tts.get_audio_contexts() == []


@pytest.mark.asyncio
async def test_bland_tts_server_preemption_releases_audio_context():
    """A server-side terminal cleans up even if Pipecat did not interrupt first."""
    captured: dict = {"messages": []}

    async with serve(
        _ws_server_handler(captured, end_reason="preempted"), "127.0.0.1", 0
    ) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = BlandTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/tts/ws",
            sample_rate=24000,
        )
        await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.3)])

    assert tts.get_audio_contexts() == []


@pytest.mark.asyncio
async def test_bland_tts_rejected_init_surfaces():
    """A session Bland refuses fails at connect, not on the first turn."""
    captured: dict = {"messages": []}
    error = {"code": "voice_not_found", "message": "Voice was not found."}

    async with serve(_ws_server_handler(captured, init_error=error), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = BlandTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/tts/ws",
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(tts, frames_to_send=[])

    errors = [f for f in down_frames + up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "voice_not_found" in errors[0].error


@pytest.mark.asyncio
async def test_bland_tts_init_timeout_closes_provisional_connection():
    """A peer that upgrades but never acknowledges init cannot hang startup."""
    captured: dict = {"messages": []}

    async with serve(
        _ws_server_handler(captured, acknowledge_init=False), "127.0.0.1", 0
    ) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = BlandTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/tts/ws",
            sample_rate=24000,
        )
        with patch("pipecat.services.bland.tts._READY_TIMEOUT_SECONDS", 0.05):
            down_frames, up_frames = await run_test(tts, frames_to_send=[])

    errors = [f for f in down_frames + up_frames if isinstance(f, ErrorFrame)]
    assert errors


@pytest.mark.asyncio
async def test_bland_tts_cancelled_init_closes_provisional_connection():
    """Task cancellation during init must not leak the upgraded socket."""
    tts = BlandTTSService(api_key="test-key")
    # Connecting reads the rate setup() resolves, and this test has no setup.
    tts._bland_sample_rate = 48000
    websocket = AsyncMock()
    websocket.recv.side_effect = asyncio.CancelledError
    tts._websocket_connect = AsyncMock(return_value=websocket)

    with pytest.raises(asyncio.CancelledError):
        await tts._connect_websocket()

    websocket.close.assert_awaited_once()


@pytest.mark.parametrize("service", [BlandTTSService, BlandHttpTTSService])
def test_bland_tts_requires_nonempty_api_key(service):
    with pytest.raises(ValueError, match="API key"):
        service(api_key="")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("encoding", "sample_rate"),
    [("mulaw", 24000), ("pcm_s16le", 16000)],
)
async def test_bland_tts_rejects_mismatched_ready_format(encoding, sample_rate):
    """Audio must never be tagged with a format the server did not acknowledge."""
    captured: dict = {"messages": []}

    async with serve(
        _ws_server_handler(
            captured,
            ready_encoding=encoding,
            ready_sample_rate=sample_rate,
        ),
        "127.0.0.1",
        0,
    ) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = BlandTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/tts/ws",
            sample_rate=24000,
        )
        down_frames, up_frames = await run_test(tts, frames_to_send=[])

    errors = [f for f in down_frames + up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "unexpected audio format" in errors[0].error


@pytest.mark.asyncio
async def test_bland_tts_close_settles_the_session():
    """Shutdown asks Bland to settle usage instead of dropping the socket."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = BlandTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/tts/ws",
            sample_rate=24000,
        )

        await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.3)])

    assert len(_of_type(captured, "close")) == 1
    assert captured["done_sent"] is True
    assert captured["sessions"] == 1


# --- /v2/tts -------------------------------------------------------------------------


async def _serve(handler):
    app = web.Application()
    app.router.add_post("/v2/tts", handler)
    return app


@pytest.mark.asyncio
async def test_run_bland_http_tts_success(aiohttp_client):
    """Sends the documented request and emits PCM frames from the response."""
    requests = []
    payload = _pcm_bytes()

    async def handler(request):
        requests.append((request.headers.get("Authorization"), await request.json()))
        return web.Response(body=payload, content_type="audio/pcm")

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandHttpTTSService(
            api_key="test-key",
            base_url=f"{base_url}/",
            aiohttp_session=session,
            sample_rate=24000,
        )
        down_frames, _ = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello from Bland.")],
        )

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types

    auth, body = requests[0]
    assert auth == "Bearer test-key"
    assert body["text"] == "Hello from Bland."
    assert body["voice"] == DEFAULT_VOICE_ID
    # 24000 is a rate Bland renders directly, so it is requested as-is.
    assert body["audio"] == {
        "encoding": "pcm_s16le",
        "sample_rate": 24000,
        "container": "raw",
    }
    assert "controls" not in body
    # fields the request shape does not define
    assert "language" not in body
    assert "output_format" not in body
    assert "voice_id" not in body

    audio = _audio_of(down_frames)
    assert audio == payload
    assert not audio.startswith(b"RIFF")
    assert {f.sample_rate for f in down_frames if isinstance(f, TTSAudioRawFrame)} == {24000}


@pytest.mark.asyncio
async def test_bland_http_tts_resamples_unsupported_pipeline_rate(aiohttp_client):
    """A pipeline rate Bland cannot emit falls back to 48 kHz and is resampled down."""
    requests = []
    payload = _pcm_bytes(4800)

    async def handler(request):
        requests.append(await request.json())
        return web.Response(body=payload, content_type="audio/pcm")

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandHttpTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=22050,
        )
        down_frames, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    assert requests[0]["audio"]["sample_rate"] == 48000
    audio = _audio_of(down_frames)
    assert audio
    assert audio != payload
    assert {f.sample_rate for f in down_frames if isinstance(f, TTSAudioRawFrame)} == {22050}


@pytest.mark.asyncio
async def test_bland_http_tts_reassembles_audio_split_across_chunks(aiohttp_client):
    """A split at an odd byte lands mid-sample; nothing may be dropped or reordered."""
    payload = _pcm_bytes()
    splits = [1, 3, 1000, 2001, len(payload)]

    async def handler(request):
        response = web.StreamResponse(headers={"content-type": "audio/pcm"})
        await response.prepare(request)
        start = 0
        for end in splits:
            await response.write(payload[start:end])
            start = end
        await response.write_eof()
        return response

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandHttpTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
        )
        down_frames, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    assert _audio_of(down_frames) == payload


@pytest.mark.asyncio
async def test_bland_http_tts_settings_payload(aiohttp_client):
    """Settings map into the request body."""
    requests = []

    async def handler(request):
        requests.append(await request.json())
        return web.Response(body=_pcm_bytes(), content_type="audio/pcm")

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandHttpTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
            settings=BlandHttpTTSService.Settings(
                voice=OTHER_VOICE_ID, expressiveness=0.9, stability=0.4
            ),
        )
        await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    body = requests[0]
    assert body["voice"] == OTHER_VOICE_ID
    assert body["controls"] == {"expressiveness": 0.9, "stability": 0.4}


@pytest.mark.asyncio
async def test_bland_http_tts_partial_controls(aiohttp_client):
    """Only controls the caller set are sent, so unset ones keep Bland's defaults."""
    requests = []

    async def handler(request):
        requests.append(await request.json())
        return web.Response(body=_pcm_bytes(), content_type="audio/pcm")

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandHttpTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
            settings=BlandHttpTTSService.Settings(stability=0.4),
        )
        await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    assert requests[0]["controls"] == {"stability": 0.4}


@pytest.mark.asyncio
async def test_bland_http_tts_error_response(aiohttp_client):
    """A non-200 response yields an ErrorFrame carrying the v2 error code and message."""

    async def handler(request):
        return web.json_response(
            {"error": {"code": "voice_not_found", "message": "Voice was not found."}},
            status=404,
        )

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandHttpTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
        )
        _, up_frames = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "voice_not_found" in errors[0].error
    assert "Voice was not found." in errors[0].error


@pytest.mark.asyncio
async def test_bland_http_tts_non_json_error_response(aiohttp_client):
    """A gateway error with an HTML body still surfaces as an ErrorFrame."""

    async def handler(request):
        return web.Response(body=b"<html>gateway</html>", status=502, content_type="text/html")

    client = await aiohttp_client(await _serve(handler))
    base_url = str(client.make_url("/v2"))

    async with aiohttp.ClientSession() as session:
        tts = BlandHttpTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            sample_rate=24000,
        )
        _, up_frames = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "502" in errors[0].error


# --- turns that cannot finish --------------------------------------------------------


def _refusing_server(captured: dict, *, code: str = "insufficient_credits"):
    """Refuses admission for the turn's context, once, as the server does.

    A refused context is recorded and its later deltas dropped silently, so a
    client that keeps feeding one gets no further reply — which is what makes the
    count of `speak` messages the thing worth asserting.
    """

    async def handler(ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                captured["messages"].append(msg)
                if msg["type"] == "init":
                    await ws.send(
                        json.dumps(
                            {
                                "type": "ready",
                                "session_id": "s1",
                                "encoding": "pcm_s16le",
                                "sample_rate": 24000,
                            }
                        )
                    )
                elif msg["type"] == "speak":
                    context_id = msg["context_id"]
                    if context_id in captured.setdefault("refused", set()):
                        continue
                    captured["refused"].add(context_id)
                    await ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "context_id": context_id,
                                "code": code,
                                "message": "wallet depleted",
                            }
                        )
                    )
                elif msg["type"] == "close":
                    await ws.send(json.dumps({"type": "done", "session_id": "s1"}))
                    return
        except websockets.ConnectionClosed:
            pass

    return handler


@pytest.mark.asyncio
async def test_bland_tts_stops_feeding_a_refused_turn():
    """A refused turn is reported once, not re-asked for every remaining token."""
    captured: dict = {"messages": []}

    async with serve(_refusing_server(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = BlandTTSService(
            api_key="test-key", url=f"ws://{host}:{port}/v2/tts/ws", sample_rate=24000
        )

        down, up = await run_test(
            tts,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMTextFrame("first"),
                SleepFrame(sleep=0.2),
                LLMTextFrame(" second"),
                SleepFrame(sleep=0.2),
                LLMTextFrame(" third"),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.2),
            ],
        )

    speaks = _of_type(captured, "speak")
    assert [m["text"] for m in speaks] == ["first"]
    # The refusal still reaches the pipeline, exactly once.
    errors = [f for f in down + up if isinstance(f, ErrorFrame)]
    assert len(errors) == 1
    assert "insufficient_credits" in errors[0].error


@pytest.mark.asyncio
async def test_bland_tts_drops_a_turn_whose_socket_died_midway():
    """Losing the socket mid-turn reports the loss instead of speaking the tail."""
    sessions: list[list[dict]] = []

    async def handler(ws):
        messages: list[dict] = []
        sessions.append(messages)
        first_session = len(sessions) == 1
        try:
            async for raw in ws:
                msg = json.loads(raw)
                messages.append(msg)
                if msg["type"] == "init":
                    await ws.send(
                        json.dumps(
                            {
                                "type": "ready",
                                "session_id": f"s{len(sessions)}",
                                "encoding": "pcm_s16le",
                                "sample_rate": 24000,
                            }
                        )
                    )
                elif msg["type"] == "speak" and first_session:
                    await ws.close(code=1011, reason="injected failure")
                    return
                elif msg["type"] == "close":
                    await ws.send(json.dumps({"type": "done", "session_id": "s"}))
                    return
        except websockets.ConnectionClosed:
            pass

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = BlandTTSService(
            api_key="test-key", url=f"ws://{host}:{port}/v2/tts/ws", sample_rate=24000
        )

        down, up = await run_test(
            tts,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMTextFrame("The weather is clear"),
                SleepFrame(sleep=0.3),
                LLMTextFrame(" and warm today."),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.3),
            ],
        )

    # The replacement session must not be handed the tail of the lost turn.
    later_speaks = [m for messages in sessions[1:] for m in messages if m["type"] == "speak"]
    assert later_speaks == []
    errors = [f for f in down + up if isinstance(f, ErrorFrame)]
    assert any("mid-turn" in f.error for f in errors)


# --- a turn the server has ended ------------------------------------------------------


def _failing_turn_server(captured: dict, *, send_error_first: bool = True):
    """Ends the first turn as `failed`, the way an oversized pause marker does."""

    async def handler(ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                captured["messages"].append(msg)
                if msg["type"] == "init":
                    await ws.send(
                        json.dumps(
                            {
                                "type": "ready",
                                "session_id": "s1",
                                "encoding": "pcm_s16le",
                                "sample_rate": msg.get("audio", {}).get("sample_rate", 48000),
                            }
                        )
                    )
                elif msg["type"] == "speak":
                    context_id = msg["context_id"]
                    if context_id in captured.setdefault("failed", set()):
                        continue
                    captured["failed"].add(context_id)
                    await ws.send(json.dumps({"type": "utterance_start", "context_id": context_id}))
                    if send_error_first:
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "error",
                                    "context_id": context_id,
                                    "code": "invalid_request",
                                    "message": "Pause marker `<|30|>` exceeds the maximum.",
                                }
                            )
                        )
                    await ws.send(
                        json.dumps(
                            {
                                "type": "utterance_end",
                                "context_id": context_id,
                                "reason": "failed",
                                "frames": 0,
                                "duration_ms": 1,
                            }
                        )
                    )
                elif msg["type"] == "close":
                    await ws.send(json.dumps({"type": "done", "session_id": "s1"}))
                    return
        except websockets.ConnectionClosed:
            pass

    return handler


@pytest.mark.asyncio
async def test_bland_tts_stops_feeding_a_failed_turn():
    """A failed turn must not be fed its remaining deltas.

    The server admits a turn on its first `speak`, so a delta arriving after the
    terminal opens — and bills — a second turn under the same context_id, which
    then speaks the tail of a sentence on its own. The later deltas are driven
    directly here: the test pipeline stops feeding a turn once its audio context
    is gone, so it cannot reach the guard that matters in a live session.
    """
    captured: dict = {"messages": []}

    async with serve(_failing_turn_server(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = BlandTTSService(
            api_key="test-key", url=f"ws://{host}:{port}/v2/tts/ws", sample_rate=24000
        )
        await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hold on <|30|>"), SleepFrame(sleep=0.3)]
        )

        spoken = _of_type(captured, "speak")
        assert len(spoken) == 1, spoken
        async for _ in tts.run_tts(" there.", spoken[0]["context_id"]):
            pass

    assert _of_type(captured, "speak") == spoken


@pytest.mark.asyncio
async def test_bland_tts_reports_a_failed_turn_once():
    """The `error` frame carries the detail; the terminal must not add a vaguer one."""
    captured: dict = {"messages": []}

    async with serve(_failing_turn_server(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = BlandTTSService(
            api_key="test-key", url=f"ws://{host}:{port}/v2/tts/ws", sample_rate=24000
        )
        down, up = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hold on <|30|> there."), SleepFrame(sleep=0.3)],
        )

    errors = [f for f in down + up if isinstance(f, ErrorFrame)]
    assert len(errors) == 1, [f.error for f in errors]
    assert "invalid_request" in errors[0].error


@pytest.mark.asyncio
async def test_bland_tts_reports_a_failed_turn_with_no_error_frame():
    """A bare `failed` terminal still has to surface something."""
    captured: dict = {"messages": []}

    async with serve(
        _failing_turn_server(captured, send_error_first=False), "127.0.0.1", 0
    ) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = BlandTTSService(
            api_key="test-key", url=f"ws://{host}:{port}/v2/tts/ws", sample_rate=24000
        )
        down, up = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.3)]
        )

    errors = [f for f in down + up if isinstance(f, ErrorFrame)]
    assert len(errors) == 1
    assert "failed" in errors[0].error


@pytest.mark.asyncio
async def test_bland_tts_idle_close_is_not_a_pipeline_error():
    """Bland reaps an idle session itself; the replacement session is routine."""
    captured: dict = {"messages": [], "reaped": False}

    async def handler(ws):
        async for raw in ws:
            message = json.loads(raw)
            captured["messages"].append(message)
            if message["type"] == "init":
                await ws.send(
                    json.dumps(
                        {
                            "type": "ready",
                            "session_id": "s1",
                            "encoding": "pcm_s16le",
                            "sample_rate": message.get("audio", {}).get("sample_rate", 48000),
                        }
                    )
                )
                # Reap the first session the way the 60s idle timeout does, and
                # hold the one that replaces it open.
                if not captured["reaped"]:
                    captured["reaped"] = True
                    await ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "code": "idle_timeout",
                                "message": "Session idle for 60s.",
                            }
                        )
                    )
                    await ws.close(code=1011, reason="idle")
                    return

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = BlandTTSService(
            api_key="test-key", url=f"ws://{host}:{port}/v2/tts/ws", sample_rate=24000
        )
        down, up = await run_test(tts, frames_to_send=[SleepFrame(sleep=0.4)])

    # A reaped session is replaced, and nothing about that is the application's
    # problem to hear about.
    inits = [m for m in captured["messages"] if m["type"] == "init"]
    assert len(inits) == 2, captured["messages"]
    errors = [f.error for f in down + up if isinstance(f, ErrorFrame)]
    assert not errors, errors


if __name__ == "__main__":
    unittest.main()
