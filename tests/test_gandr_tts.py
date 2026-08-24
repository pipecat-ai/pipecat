#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for GandrTTSService."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock

import pytest
import websockets
from pydantic import ValidationError
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.gandr._text import MAX_REQUEST_CHARS, split_for_request
from pipecat.services.gandr.tts import GandrTTSService, _Utterance
from pipecat.tests.utils import SleepFrame, run_test

AUDIO_CHUNK_1 = b"\x00\x01" * 512
AUDIO_CHUNK_2 = b"\x02\x03" * 512

#: A syntactically valid base64 payload; the fake server never decodes it.
FAKE_REFERENCE_WAV = "UklGRmZha2U="


def _audio_of(frames) -> bytes:
    return b"".join(f.audio for f in frames if isinstance(f, TTSAudioRawFrame))


def _texts(captured: dict) -> list:
    return [m["text"] for m in captured["messages"]]


# --- /ws ------------------------------------------------------------------------------


def _ws_server_handler(
    captured: dict,
    *,
    audio_map: dict | None = None,
    busy_times: int = 0,
    error_code: str | None = None,
    require_voice: bool = False,
    forget_voice: bool = False,
    send_completion: bool = True,
):
    """Build a fake Gandr door following the documented utterance flow.

    One JSON in per utterance; binary PCM frames out, then a closing JSON
    carrying the server's own timings. Errors are JSON of the shape
    ``{"error": code}``. ``require_voice`` refuses utterances until one
    arrives carrying ``voice_wav_b64``, the way a connection with a cloned
    voice behaves; ``forget_voice`` refuses every utterance without it, the
    way a server that lost its registration behaves. ``busy_times`` answers
    ``busy`` that many times per distinct text before rendering it.
    """

    async def handler(ws):
        captured["api_key_header"] = ws.request.headers.get("x-api-key")
        captured["sessions"] = captured.get("sessions", 0) + 1
        voice_registered = False

        try:
            async for raw in ws:
                msg = json.loads(raw)
                captured["messages"].append(msg)

                if error_code is not None:
                    await ws.send(json.dumps({"error": error_code}))
                    continue

                if "voice_wav_b64" in msg:
                    voice_registered = True
                elif require_voice or forget_voice:
                    if forget_voice or not voice_registered:
                        await ws.send(json.dumps({"error": "need_voice"}))
                        continue

                if busy_times:
                    counts = captured.setdefault("busy_counts", {})
                    seen = counts.get(msg["text"], 0)
                    if seen < busy_times:
                        counts[msg["text"]] = seen + 1
                        await ws.send(json.dumps({"error": "busy"}))
                        continue

                chunks = (audio_map or {}).get(msg["text"], [AUDIO_CHUNK_1, AUDIO_CHUNK_2])
                for chunk in chunks:
                    await ws.send(chunk)
                if send_completion:
                    await ws.send(json.dumps({"ttfa_ms": 111, "audio_ms": 222}))
        except websockets.ConnectionClosed:
            pass

    return handler


@pytest.mark.asyncio
async def test_gandr_tts_protocol_roundtrip():
    """One utterance in, its audio out, and the message carries only what was set."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = GandrTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/ws",
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello from Gandr."), SleepFrame(sleep=0.3)],
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

    assert captured["api_key_header"] == "test-key"
    assert captured["sessions"] == 1

    message = captured["messages"][0]
    assert message["text"] == "Hello from Gandr."
    assert message["lang"] == "en"
    assert message["voice_id"] == "gandr-mia"
    assert message["output_sample_rate"] == 24000
    # Unset controls stay off the wire, so the door's defaults apply.
    for key in ("speed", "volume", "temperature", "cfg_weight", "seed", "voice_wav_b64"):
        assert key not in message, key


@pytest.mark.asyncio
async def test_gandr_tts_utterances_serialize_in_order_on_one_connection():
    """The connection is reused, and each utterance's audio arrives under its turn."""
    captured: dict = {"messages": []}
    audio_map = {
        "First thing.": [AUDIO_CHUNK_1],
        "Second thing.": [AUDIO_CHUNK_2],
    }

    async with serve(_ws_server_handler(captured, audio_map=audio_map), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = GandrTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/ws",
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[
                TTSSpeakFrame(text="First thing."),
                SleepFrame(sleep=0.3),
                TTSSpeakFrame(text="Second thing."),
                SleepFrame(sleep=0.3),
            ],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    assert captured["sessions"] == 1
    assert _texts(captured) == ["First thing.", "Second thing."]
    # A single in-flight pointer attributes audio to the current utterance, so
    # ordering is the observable proof the sender serialises.
    assert _audio_of(down_frames) == AUDIO_CHUNK_1 + AUDIO_CHUNK_2
    stopped = [f for f in down_frames if isinstance(f, TTSStoppedFrame)]
    assert len(stopped) == 2


@pytest.mark.asyncio
async def test_gandr_tts_expression_controls_sent_when_set():
    """Controls set in InputParams ride every utterance message."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = GandrTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/ws",
            sample_rate=24000,
            params=GandrTTSService.InputParams(
                voice_id="gandr-leo",
                language="es",
                speed=1.2,
                volume=0.8,
                temperature=0.3,
                cfg_weight=0.6,
                seed=7,
            ),
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hola."), SleepFrame(sleep=0.3)],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    message = captured["messages"][0]
    assert message["voice_id"] == "gandr-leo"
    assert message["lang"] == "es"
    assert message["speed"] == 1.2
    assert message["volume"] == 0.8
    assert message["temperature"] == 0.3
    assert message["cfg_weight"] == 0.6
    assert message["seed"] == 7


@pytest.mark.asyncio
async def test_gandr_tts_input_params_sample_rate_beats_constructor():
    """InputParams.sample_rate wins over the constructor's, and tags the audio."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = GandrTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/ws",
            sample_rate=16000,
            params=GandrTTSService.InputParams(sample_rate=22050),
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.3)],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    assert captured["messages"][0]["output_sample_rate"] == 22050
    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert audio_frames
    assert all(frame.sample_rate == 22050 for frame in audio_frames)


@pytest.mark.asyncio
async def test_gandr_tts_clone_reference_registers_once_per_connection():
    """Reference audio rides the first utterance only; the registration persists."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured, require_voice=True), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = GandrTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/ws",
            sample_rate=24000,
            params=GandrTTSService.InputParams(
                voice_id="gnd:abc123",
                voice_wav_b64=FAKE_REFERENCE_WAV,
            ),
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[
                TTSSpeakFrame(text="First thing."),
                SleepFrame(sleep=0.3),
                TTSSpeakFrame(text="Second thing."),
                SleepFrame(sleep=0.3),
            ],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    assert captured["sessions"] == 1
    assert len(captured["messages"]) == 2
    assert captured["messages"][0]["voice_wav_b64"] == FAKE_REFERENCE_WAV
    # Sending it again would re-upload the reference on every turn.
    assert "voice_wav_b64" not in captured["messages"][1]
    assert _audio_of(down_frames) == (AUDIO_CHUNK_1 + AUDIO_CHUNK_2) * 2


@pytest.mark.asyncio
async def test_gandr_tts_need_voice_retries_with_reference():
    """A server that lost the registration gets the reference audio again."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured, forget_voice=True), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = GandrTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/ws",
            sample_rate=24000,
            params=GandrTTSService.InputParams(
                voice_id="gnd:abc123",
                voice_wav_b64=FAKE_REFERENCE_WAV,
            ),
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[
                TTSSpeakFrame(text="First thing."),
                SleepFrame(sleep=0.3),
                TTSSpeakFrame(text="Second thing."),
                SleepFrame(sleep=0.3),
            ],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    # First utterance carried the reference. The second went without it, was
    # refused with need_voice, and was retried with the reference attached.
    assert _texts(captured) == ["First thing.", "Second thing.", "Second thing."]
    assert "voice_wav_b64" in captured["messages"][0]
    assert "voice_wav_b64" not in captured["messages"][1]
    assert "voice_wav_b64" in captured["messages"][2]
    assert _audio_of(down_frames) == (AUDIO_CHUNK_1 + AUDIO_CHUNK_2) * 2


@pytest.mark.asyncio
async def test_gandr_tts_need_voice_without_reference_is_a_clear_error():
    """A cloned voice with no reference audio fails with the fix in the message."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured, require_voice=True), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = GandrTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/ws",
            sample_rate=24000,
            params=GandrTTSService.InputParams(voice_id="gnd:abc123"),
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.3)],
        )

    errors = [f for f in down_frames + up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "voice_wav_b64" in errors[0].error
    assert tts.get_audio_contexts() == []


@pytest.mark.asyncio
async def test_gandr_tts_busy_backpressure_retries():
    """A busy answer is retried after the backoff, and the caller never hears it."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured, busy_times=1), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = GandrTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/ws",
            sample_rate=24000,
            busy_retry_s=0.05,
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.5)],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    assert _texts(captured) == ["Hi.", "Hi."]
    assert _audio_of(down_frames) == AUDIO_CHUNK_1 + AUDIO_CHUNK_2


@pytest.mark.asyncio
async def test_gandr_tts_stays_busy_surfaces_error():
    """Backpressure that never lifts is reported instead of retried forever."""
    captured: dict = {"messages": []}

    async with serve(_ws_server_handler(captured, busy_times=99), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = GandrTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/ws",
            sample_rate=24000,
            busy_retry_s=0.01,
            max_attempts=2,
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.5)],
        )

    errors = [f for f in down_frames + up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "busy" in errors[0].error
    assert tts.get_audio_contexts() == []


@pytest.mark.asyncio
async def test_gandr_tts_error_code_surfaces():
    """An error code from the door becomes an ErrorFrame carrying that code."""
    captured: dict = {"messages": []}

    async with serve(
        _ws_server_handler(captured, error_code="quota_exceeded"), "127.0.0.1", 0
    ) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = GandrTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/ws",
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.3)],
        )

    errors = [f for f in down_frames + up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "quota_exceeded" in errors[0].error
    assert tts.get_audio_contexts() == []


@pytest.mark.asyncio
async def test_gandr_tts_missing_completion_frame_times_out():
    """An utterance whose closing frame never arrives fails instead of hanging."""
    captured: dict = {"messages": []}

    async with serve(
        _ws_server_handler(captured, send_completion=False), "127.0.0.1", 0
    ) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = GandrTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/ws",
            sample_rate=24000,
            utterance_timeout_s=0.2,
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.6)],
        )

    errors = [f for f in down_frames + up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "no completion frame" in errors[0].error
    assert tts.get_audio_contexts() == []


@pytest.mark.asyncio
async def test_gandr_tts_long_text_is_split_untruncated():
    """Text past the request cap goes as several utterances, nothing dropped."""
    captured: dict = {"messages": []}
    text = "All work and no play makes Jack a dull boy. " * 60

    async with serve(_ws_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts = GandrTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/ws",
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text=text), SleepFrame(sleep=0.5)],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    sent = _texts(captured)
    assert len(sent) >= 2
    assert all(len(piece) <= MAX_REQUEST_CHARS for piece in sent)
    assert " ".join(sent) == text.strip()
    # One logical turn: many wire messages, one stop.
    stopped = [f for f in down_frames if isinstance(f, TTSStoppedFrame)]
    assert len(stopped) == 1
    assert _audio_of(down_frames) == (AUDIO_CHUNK_1 + AUDIO_CHUNK_2) * len(sent)


# --- interruption ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gandr_tts_interruption_drops_queued_and_abandons_inflight():
    """Barge-in empties the outbox and abandons the render on the wire."""
    tts = GandrTTSService(
        api_key="test-key", sample_rate=24000, reconnect_on_interruption=False
    )
    tts._reconnect = AsyncMock()
    websocket = AsyncMock()
    tts._websocket = websocket

    queued_1 = _Utterance(text="queued one", context_id="turn-17", is_final=False)
    queued_2 = _Utterance(text="queued two", context_id="turn-17", is_final=True)
    tts._outbox.put_nowait(queued_1)
    tts._outbox.put_nowait(queued_2)
    inflight = _Utterance(text="rendering now", context_id="turn-17", is_final=False)
    tts._inflight = inflight

    await tts.on_audio_context_interrupted("turn-17")

    assert tts._outbox.qsize() == 0
    assert inflight.abandoned
    assert inflight.done.is_set()
    # With reconnect_on_interruption off, the connection survives the barge-in.
    assert not tts._reconnect.called
    assert not websocket.close.called


@pytest.mark.asyncio
async def test_gandr_tts_interruption_reconnects_by_default():
    """By default barge-in reopens the connection so the next turn starts clean."""
    tts = GandrTTSService(api_key="test-key", sample_rate=24000)
    tts._reconnect = AsyncMock()
    tts._websocket = AsyncMock()
    tts._inflight = _Utterance(text="rendering now", context_id="turn-17", is_final=False)

    await tts.on_audio_context_interrupted("turn-17")

    tts._reconnect.assert_awaited_once()


# --- construction ---------------------------------------------------------------------


@pytest.mark.parametrize("api_key", ["", "   "])
def test_gandr_tts_requires_nonempty_api_key(api_key):
    with pytest.raises(ValueError, match="API key"):
        GandrTTSService(api_key=api_key)


@pytest.mark.parametrize(
    "params",
    [
        {"voice_id": "   "},
        {"language": "   "},
        {"sample_rate": 44100},
        {"speed": 0.5},
        {"speed": 1.6},
        {"volume": 0.4},
        {"volume": 2.5},
    ],
)
def test_gandr_tts_input_params_rejects_out_of_range(params):
    with pytest.raises(ValidationError):
        GandrTTSService.InputParams(**params)


def test_gandr_tts_input_params_accepts_the_boundaries():
    params = GandrTTSService.InputParams(
        voice_id="gandr-ava",
        language="fr",
        sample_rate=8000,
        speed=0.6,
        volume=2.0,
    )
    assert params.sample_rate == 8000
    assert params.speed == 0.6
    assert params.volume == 2.0


# --- text splitting -------------------------------------------------------------------


def test_split_for_request_empty_input_yields_nothing():
    assert split_for_request("") == []
    assert split_for_request("   \n  ") == []


def test_split_for_request_short_text_passes_through_stripped():
    assert split_for_request("  Hello there.  ") == ["Hello there."]


def test_split_for_request_prefers_sentence_boundaries():
    text = "One sentence here. " * 300
    pieces = split_for_request(text)
    assert len(pieces) >= 2
    assert all(len(piece) <= MAX_REQUEST_CHARS for piece in pieces)
    assert all(piece.endswith(".") for piece in pieces)
    assert " ".join(pieces) == text.strip()


def test_split_for_request_hard_cuts_an_unbroken_token():
    text = "x" * (MAX_REQUEST_CHARS * 2 + 500)
    pieces = split_for_request(text)
    assert [len(piece) for piece in pieces] == [MAX_REQUEST_CHARS, MAX_REQUEST_CHARS, 500]
    assert "".join(pieces) == text


def test_split_for_request_rejects_a_nonpositive_limit():
    with pytest.raises(ValueError):
        split_for_request("hello", limit=0)


if __name__ == "__main__":
    unittest.main()
