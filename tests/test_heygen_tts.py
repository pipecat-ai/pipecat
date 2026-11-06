#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for HeyGenTTSService and HeyGenHttpTTSService."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest
import websockets
from aiohttp import web
from websockets.asyncio.server import serve
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.heygen.tts import (
    HEYGEN_DEFAULT_VOICE_ID,
    HEYGEN_KEEPALIVE_CONTEXT_ID,
    HeyGenHttpTTSService,
    HeyGenTTSService,
    language_to_heygen_language,
)
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

STREAM_PATH = "/v3/models/audio/tts/stream"


def _make_app(requests: list, audio: bytes = b"", status: int = 200, error: str = ""):
    async def handler(request):
        requests.append({"headers": dict(request.headers), "body": await request.json()})
        if status != 200:
            return web.json_response({"error": {"message": error}}, status=status)

        response = web.StreamResponse(
            status=200, headers={"Content-Type": "audio/pcm", "X-Request-Id": "req-123"}
        )
        await response.prepare(request)
        # Split mid-sample to check that emitted frames stay sample-aligned.
        await response.write(audio[:2047])
        await asyncio.sleep(0.01)
        await response.write(audio[2047:])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post(STREAM_PATH, handler)
    return app


@pytest.mark.asyncio
async def test_heygen_tts_streams_pcm(aiohttp_client):
    """The service sends the documented request and emits sample-aligned PCM frames."""
    requests = []
    pcm_audio = b"\x00\x01\x02\x03" * 1024
    client = await aiohttp_client(_make_app(requests, pcm_audio))

    async with aiohttp.ClientSession() as session:
        tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=24000,
        )
        down_frames, _ = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hello from HeyGen.")]
        )

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types

    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in audio_frames)
    assert all(len(f.audio) % 2 == 0 for f in audio_frames)
    assert b"".join(f.audio for f in audio_frames) == pcm_audio

    assert len(requests) == 1
    assert requests[0]["headers"]["x-api-key"] == "test-key"
    assert requests[0]["body"] == {
        "text": "Hello from HeyGen.",
        "voice_id": HEYGEN_DEFAULT_VOICE_ID,
        "model": "heygen-voice-1-turbo",
        "output_format": {"encoding": "pcm", "sample_rate_hertz": 24000},
    }


@pytest.mark.asyncio
async def test_heygen_tts_settings_in_body(aiohttp_client):
    """Voice, model and language settings reach the request body."""
    requests = []
    client = await aiohttp_client(_make_app(requests, b"\x00\x00" * 2048))

    async with aiohttp.ClientSession() as session:
        tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=16000,
            settings=HeyGenHttpTTSService.Settings(
                voice="my-voice", model="heygen-voice-1", language=Language.EN_US
            ),
        )
        await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    body = requests[0]["body"]
    assert body["voice_id"] == "my-voice"
    assert body["model"] == "heygen-voice-1"
    assert body["language"] == "en"
    assert body["output_format"]["sample_rate_hertz"] == 16000


@pytest.mark.asyncio
async def test_heygen_tts_unsupported_sample_rate_is_resampled(aiohttp_client):
    """A pipeline rate HeyGen doesn't offer requests 24 kHz and resamples to the pipeline rate."""
    requests = []
    client = await aiohttp_client(_make_app(requests, b"\x00\x00" * 24000))

    async with aiohttp.ClientSession() as session:
        tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=32000,
        )
        down_frames, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    assert requests[0]["body"]["output_format"]["sample_rate_hertz"] == 24000
    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames
    assert all(f.sample_rate == 32000 for f in audio_frames)


@pytest.mark.asyncio
async def test_heygen_tts_http_error(aiohttp_client):
    """A non-200 response produces an ErrorFrame with the status and message."""
    requests = []
    client = await aiohttp_client(_make_app(requests, status=429, error="Request limit reached."))

    async with aiohttp.ClientSession() as session:
        tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=24000,
        )
        _, up_frames = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "429" in errors[0].error
    assert "Request limit reached." in errors[0].error


def test_language_mapping():
    """Regional variants resolve to HeyGen's base ISO 639-1 codes."""
    assert language_to_heygen_language(Language.EN) == "en"
    assert language_to_heygen_language(Language.EN_GB) == "en"
    assert language_to_heygen_language(Language.ZH_HK) == "zh"
    assert language_to_heygen_language(Language.PT_BR) == "pt"


# --- HeyGenTTSService (WebSocket) ---


def _alignment(text: str) -> dict:
    chars = list(text)
    return {
        "chars": chars,
        "charStartTimesMs": [i * 50 for i in range(len(chars))],
        "charDurationsMs": [50 for _ in chars],
    }


async def _serve_heygen_ws(
    captured: dict, reply_with_error: bool = False, final_delay: float = 0.0
):
    """A fake multi-stream-input socket: audio for each text, isFinal on close_context.

    ``final_delay`` holds each isFinal back, as real synthesis takes time; the peak
    number of contexts open at once is recorded in ``captured["peak_open"]``.
    """
    open_contexts: set[str] = set()
    captured["peak_open"] = 0

    async def send_final(ws, ctx):
        await asyncio.sleep(final_delay)
        open_contexts.discard(ctx)
        await ws.send(json.dumps({"isFinal": True, "contextId": ctx}))

    audio = b"\x00\x01" * 2400

    async def handler(ws):
        captured["path"] = ws.request.path
        captured["api_key"] = ws.request.headers.get("x-api-key")
        try:
            async for raw in ws:
                msg = json.loads(raw)
                captured["messages"].append(msg)
                ctx = msg.get("context_id")
                if msg.get("text") == " ":
                    open_contexts.add(ctx)
                    captured["peak_open"] = max(captured["peak_open"], len(open_contexts))
                if reply_with_error and msg.get("text", "").strip():
                    await ws.send(
                        json.dumps(
                            {
                                "message": "Insufficient credits.",
                                "error": {
                                    "code": "usage_limit_reached",
                                    "request_id": "req-123",
                                },
                            }
                        )
                    )
                    await ws.close(1008, "usage_limit_reached")
                    return
                if msg.get("text", "").strip():
                    await ws.send(
                        json.dumps(
                            {
                                "audio": base64.b64encode(audio).decode(),
                                "alignment": _alignment(msg["text"]),
                                "contextId": ctx,
                                "isFinal": None,
                            }
                        )
                    )
                if msg.get("close_context"):
                    asyncio.create_task(send_final(ws, ctx))
                if msg.get("close_socket"):
                    await ws.close()
                    return
        except websockets.ConnectionClosed:
            pass

    return serve(handler, "127.0.0.1", 0), audio


@pytest.mark.asyncio
async def test_heygen_websocket_streams_audio_and_words():
    """The service connects with the v3 parameters and turns alignment into word frames."""
    captured: dict = {"messages": []}
    server_cm, audio = await _serve_heygen_ws(captured)

    async with server_cm as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = HeyGenTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}",
            sample_rate=24000,
            settings=HeyGenTTSService.Settings(language=Language.EN_US),
        )
        down_frames, _ = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello from HeyGen."), SleepFrame(sleep=0.5)],
        )

    assert captured["api_key"] == "test-key"
    parsed = urlparse(captured["path"])
    assert parsed.path == "/v3/models/audio/tts/multi-stream-input"
    assert parse_qs(parsed.query) == {
        "voice_id": [HEYGEN_DEFAULT_VOICE_ID],
        "model": ["heygen-voice-1-turbo"],
        "output_format": ["pcm_24000"],
        "auto_mode": ["true"],
        "language": ["en"],
    }

    sent = captured["messages"]
    context_id = sent[0]["context_id"]
    assert sent[0] == {"text": " ", "context_id": context_id}
    assert {"text": "Hello from HeyGen. ", "context_id": context_id} in sent
    assert {"context_id": context_id, "close_context": True} in sent

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types
    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert b"".join(f.audio for f in audio_frames) == audio
    words = [f.text for f in down_frames if isinstance(f, TTSTextFrame)]
    assert words == ["Hello", "from", "HeyGen."]


@pytest.mark.asyncio
async def test_heygen_websocket_error_message_becomes_error_frame():
    """HeyGen's error message before a close surfaces as an ErrorFrame with its text."""
    captured: dict = {"messages": []}
    server_cm, _ = await _serve_heygen_ws(captured, reply_with_error=True)

    async with server_cm as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = HeyGenTTSService(api_key="test-key", url=f"ws://{host}:{port}", sample_rate=24000)
        _, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello."), SleepFrame(sleep=0.5)],
        )

    errors = [f.error for f in up_frames if isinstance(f, ErrorFrame)]
    assert any("Insufficient credits." in e and "req-123" in e for e in errors)


@pytest.mark.asyncio
async def test_heygen_websocket_interruption_closes_context():
    """On barge-in the service closes the HeyGen context and keeps the socket open."""
    tts = HeyGenTTSService(api_key="test-key", sample_rate=24000)
    websocket = AsyncMock()
    websocket.state = State.OPEN
    tts._websocket = websocket

    await tts.on_audio_context_interrupted("ctx-1")

    sent = [json.loads(call.args[0]) for call in websocket.send.call_args_list]
    assert {"context_id": "ctx-1", "close_context": True} in sent


@pytest.mark.asyncio
async def test_heygen_websocket_keepalive_never_names_a_closed_context():
    """Keepalives name only an open context; HeyGen opens a context for any other."""
    tts = HeyGenTTSService(api_key="test-key", sample_rate=24000)
    websocket = AsyncMock()
    websocket.state = State.OPEN
    tts._websocket = websocket

    def last_sent():
        return json.loads(websocket.send.call_args_list[-1].args[0])

    idle = {"context_id": HEYGEN_KEEPALIVE_CONTEXT_ID, "close_context": True}

    await tts._send_keepalive()
    assert last_sent() == idle

    tts._turn_context_id = "ctx-1"
    await tts._send_context_init("ctx-1")
    await tts._send_keepalive()
    assert last_sent() == {"text": "", "context_id": "ctx-1"}

    # Closed but still the active context while its last audio arrives.
    await tts._close_context("ctx-1")
    await tts._send_keepalive()
    assert last_sent() == idle


def test_heygen_websocket_token_mode_disables_auto_mode():
    """Token streaming leaves generation to HeyGen's chunk schedule."""
    tts = HeyGenTTSService(api_key="k", text_aggregation_mode=TextAggregationMode.TOKEN)
    assert tts._auto_mode is False
    assert HeyGenTTSService(api_key="k")._auto_mode is True


@pytest.mark.asyncio
async def test_heygen_websocket_unsupported_sample_rate_is_resampled():
    """At a rate HeyGen doesn't offer, the socket asks for 24 kHz and frames carry the pipeline rate."""
    captured: dict = {"messages": []}
    server_cm, audio = await _serve_heygen_ws(captured)

    async with server_cm as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = HeyGenTTSService(api_key="test-key", url=f"ws://{host}:{port}", sample_rate=11025)
        down_frames, _ = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hello."), SleepFrame(sleep=0.5)]
        )

    assert parse_qs(urlparse(captured["path"]).query)["output_format"] == ["pcm_24000"]
    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames
    assert all(f.sample_rate == 11025 for f in audio_frames)
    resampled = sum(len(f.audio) for f in audio_frames)
    assert resampled < len(audio) * 0.6  # 24 kHz -> 11.025 kHz is about 0.46x the bytes


@pytest.mark.asyncio
async def test_heygen_websocket_turn_end_flushes_before_close():
    """At the end of a turn the context is flushed once, then closed, so no text is dropped."""
    tts = HeyGenTTSService(api_key="test-key", sample_rate=24000)
    websocket = AsyncMock()
    websocket.state = State.OPEN
    tts._websocket = websocket
    tts._turn_context_id = "ctx-1"
    tts._context_init_sent.add("ctx-1")
    tts.audio_context_available = lambda context_id: context_id == "ctx-1"

    await tts.on_turn_context_completed()

    sent = [json.loads(call.args[0]) for call in websocket.send.call_args_list]
    assert sent == [
        {"context_id": "ctx-1", "flush": True},
        {"context_id": "ctx-1", "close_context": True},
    ]


@pytest.mark.asyncio
async def test_heygen_websocket_turn_without_text_sends_nothing():
    """A turn that produced no speech, such as a tool call alone, opens no HeyGen context."""
    tts = HeyGenTTSService(api_key="test-key", sample_rate=24000)
    websocket = AsyncMock()
    websocket.state = State.OPEN
    tts._websocket = websocket
    tts._turn_context_id = "ctx-1"

    await tts.on_turn_context_completed()

    websocket.send.assert_not_called()


def test_heygen_websocket_url_options():
    """Connection options and settings reach the URL only when given."""
    plain = parse_qs(urlparse(HeyGenTTSService(api_key="k")._build_websocket_url()).query)
    for name in (
        "enable_logging",
        "enable_ssml_parsing",
        "end_user_id",
        "apply_text_normalization",
        "expressiveness_boost",
    ):
        assert name not in plain

    tts = HeyGenTTSService(
        api_key="k",
        enable_logging=False,
        enable_ssml_parsing=True,
        end_user_id="user-42",
        settings=HeyGenTTSService.Settings(
            apply_text_normalization="off", expressiveness_boost=0.5
        ),
    )
    query = parse_qs(urlparse(tts._build_websocket_url()).query)
    assert query["enable_logging"] == ["false"]
    assert query["enable_ssml_parsing"] == ["true"]
    assert query["end_user_id"] == ["user-42"]
    assert query["apply_text_normalization"] == ["off"]
    assert query["expressiveness_boost"] == ["0.5"]


def test_heygen_websocket_holds_frames_until_turn_audio_plays():
    """By default turns run one at a time, keeping under HeyGen's 5 contexts per socket."""
    assert HeyGenTTSService(api_key="k")._pause_frame_processing is True


@pytest.mark.asyncio
async def test_heygen_unsupported_language_is_not_sent(aiohttp_client):
    """A language HeyGen doesn't speak is left out, so HeyGen detects the language."""
    requests = []
    client = await aiohttp_client(_make_app(requests, b"\x00\x00" * 2048))
    async with aiohttp.ClientSession() as session:
        http_tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=24000,
            settings=HeyGenHttpTTSService.Settings(language=Language.NO),
        )
        await run_test(http_tts, frames_to_send=[TTSSpeakFrame(text="Hei.")])
    assert "language" not in requests[0]["body"]

    ws_tts = HeyGenTTSService(
        api_key="k", sample_rate=24000, settings=HeyGenTTSService.Settings(language=Language.NO)
    )
    assert "language=" not in ws_tts._build_websocket_url()


@pytest.mark.asyncio
async def test_heygen_websocket_handshake_refusal_reports_reason():
    """HeyGen's reason for refusing the handshake reaches the ErrorFrame."""
    body = '{"error":{"code":"invalid_parameter","message":"Voice not found"}}'

    def refuse(connection, request):
        return connection.respond(400, body)

    async with serve(lambda ws: None, "127.0.0.1", 0, process_request=refuse) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = HeyGenTTSService(api_key="test-key", url=f"ws://{host}:{port}", sample_rate=24000)
        _, up_frames = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hello."), SleepFrame(sleep=0.3)]
        )

    errors = [f.error for f in up_frames if isinstance(f, ErrorFrame)]
    assert any("Voice not found" in e for e in errors)


@pytest.mark.asyncio
async def test_heygen_websocket_back_to_back_speech_stays_under_context_limit():
    """Seven queued sentences never hold more than HeyGen's 5 contexts open at once."""
    captured: dict = {"messages": []}
    server_cm, _ = await _serve_heygen_ws(captured, final_delay=0.2)

    async with server_cm as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = HeyGenTTSService(api_key="test-key", url=f"ws://{host}:{port}", sample_rate=24000)
        await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text=f"Sentence {i}.") for i in range(7)]
            + [SleepFrame(sleep=2.5)],
        )

    opened = {m["context_id"] for m in captured["messages"] if m.get("text") == " "}
    assert len(opened) == 7
    assert captured["peak_open"] <= 5
