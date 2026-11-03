#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AssemblyAITTSService against a fake Streaming TTS server."""

import base64
import json
import time
from urllib.parse import parse_qs, urlparse

import pytest
import websockets
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    ErrorFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.assemblyai import tts as assemblyai_tts
from pipecat.services.assemblyai.tts import AssemblyAITTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

# 80 ms of 24 kHz 16-bit mono audio per Audio frame, two frames per request.
AUDIO_CHUNK = b"\x01\x02" * 1920
CHUNK_MS = 80
WORD_MS = 50


def _server(
    captured: dict,
    *,
    word_boundaries: bool = True,
    stall_first_flush: bool = False,
    error_after_begin: dict | None = None,
    error_before_begin: dict | None = None,
    error_once: bool = False,
    configuration: dict | None = None,
    expires_in: float = 3600,
):
    """Build a fake server following the documented Streaming TTS protocol.

    Each Flush is answered with two Audio frames, a FlushDone and (unless
    ``word_boundaries`` is False) a WordBoundaries frame placing the request's
    words on the session timeline. With ``stall_first_flush`` the first request
    sends one Audio frame and then waits, so it is still in flight when the
    client cancels it. ``error_after_begin`` ends every session (or only the
    first, with ``error_once``) with that Error; ``error_before_begin`` refuses
    every session with it in place of Begin. ``configuration`` adds keys to
    Begin's configuration, and ``expires_in`` sets the first session's
    expires_at; later sessions last an hour.
    """

    async def handler(ws):
        captured["connections"] = captured.get("connections", 0) + 1
        captured.setdefault("connect_times", []).append(time.monotonic())
        captured["path"] = ws.request.path
        captured["auth"] = ws.request.headers.get("Authorization")
        if error_before_begin:
            await ws.send(json.dumps({"type": "Error", **error_before_begin}))
            await ws.close(error_before_begin["error_code"], "See Error message for details")
            return
        await ws.send(
            json.dumps(
                {
                    "type": "Begin",
                    "id": f"session-{captured['connections']}",
                    "expires_at": time.time()
                    + (expires_in if captured["connections"] == 1 else 3600),
                    "configuration": {
                        "voice": "jane",
                        "word_boundaries": word_boundaries,
                        **(configuration or {}),
                    },
                }
            )
        )
        if error_after_begin and not (error_once and captured["connections"] > 1):
            await ws.send(json.dumps({"type": "Error", **error_after_begin}))
            await ws.close(error_after_begin["error_code"], "See Error message for details")
            return

        flush_id = 0
        session_ms = 0
        buffer = ""
        stalled = stall_first_flush
        try:
            async for raw in ws:
                msg = json.loads(raw)
                captured["messages"].append(msg)
                if msg["type"] == "Generate":
                    buffer += msg["text"]
                elif msg["type"] == "Flush":
                    audio = base64.b64encode(AUDIO_CHUNK).decode()
                    frame = {"type": "Audio", "audio": audio, "flush_id": flush_id}
                    if stalled:
                        stalled = False
                        await ws.send(json.dumps(frame))
                        # Cancelled audio still holds its place on the timeline.
                        session_ms += CHUNK_MS
                        flush_id += 1
                        buffer = ""
                        continue
                    await ws.send(json.dumps(frame))
                    await ws.send(json.dumps(frame))
                    duration = 2 * CHUNK_MS
                    await ws.send(
                        json.dumps(
                            {
                                "type": "FlushDone",
                                "flush_id": flush_id,
                                "audio_duration_ms": duration,
                            }
                        )
                    )
                    if word_boundaries:
                        words = [
                            {
                                "word": word,
                                "start": session_ms + i * WORD_MS,
                                "end": session_ms + (i + 1) * WORD_MS,
                                "confidence": 0.9,
                            }
                            for i, word in enumerate(buffer.split())
                        ]
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "WordBoundaries",
                                    "flush_id": flush_id,
                                    "audio_start_ms": session_ms,
                                    "words": words,
                                }
                            )
                        )
                    session_ms += duration
                    flush_id += 1
                    buffer = ""
                elif msg["type"] == "Cancel":
                    await ws.send(json.dumps({"type": "Cancelled", "audio_duration_seconds": 0.08}))
                    flush_id += 1
                    buffer = ""
                elif msg["type"] == "Terminate":
                    await ws.send(
                        json.dumps(
                            {
                                "type": "Termination",
                                "session_duration_seconds": 0,
                                "total_input_char_length": 0,
                                "audio_duration_seconds": 0.0,
                            }
                        )
                    )
                    await ws.close(1000, "Session ended")
                    return
        except websockets.ConnectionClosed:
            pass

    return handler


def _service(server, **kwargs) -> AssemblyAITTSService:
    port = next(iter(server.sockets)).getsockname()[1]
    return AssemblyAITTSService(
        api_key="test-key", url=f"ws://127.0.0.1:{port}/v1/ws", sample_rate=24000, **kwargs
    )


def _sent(captured: dict, msg_type: str) -> list[dict]:
    return [m for m in captured["messages"] if m["type"] == msg_type]


@pytest.mark.asyncio
async def test_speak_streams_audio_and_words():
    """A TTSSpeakFrame is sent as one flushed request and its audio and words come back."""
    captured: dict = {"messages": []}
    async with serve(_server(captured), "127.0.0.1", 0) as server:
        down_frames, up_frames = await run_test(
            _service(server),
            frames_to_send=[
                TTSSpeakFrame(text="Hello from AssemblyAI."),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert not any(isinstance(f, ErrorFrame) for f in [*down_frames, *up_frames])
    assert sum(isinstance(f, TTSStartedFrame) for f in down_frames) == 1
    assert sum(isinstance(f, TTSStoppedFrame) for f in down_frames) == 1
    audio = b"".join(f.audio for f in down_frames if isinstance(f, TTSAudioRawFrame))
    assert audio == AUDIO_CHUNK * 2
    words = [f.text for f in down_frames if isinstance(f, TTSTextFrame)]
    assert words == ["Hello", "from", "AssemblyAI."]

    # The key goes in the header raw, and the session is configured in the URL.
    assert captured["auth"] == "test-key"
    query = parse_qs(urlparse(captured["path"]).query)
    assert query["sample_rate"] == ["24000"]
    assert query["encoding"] == ["pcm_s16le"]
    assert [m["type"] for m in captured["messages"]] == ["Generate", "Flush"]
    assert captured["connections"] == 1


@pytest.mark.asyncio
async def test_llm_turn_flushes_each_sentence_into_one_context():
    """Each sentence is its own request, and the turn's words stay in order."""
    captured: dict = {"messages": []}
    tokens = "Hi there! Thanks for calling. How can I help?".split(" ")
    async with serve(_server(captured), "127.0.0.1", 0) as server:
        down_frames, _ = await run_test(
            _service(server),
            frames_to_send=[
                LLMFullResponseStartFrame(),
                *[LLMTextFrame(text=token + " ") for token in tokens],
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    generates = [m["text"] for m in _sent(captured, "Generate")]
    assert generates == ["Hi there!", "Thanks for calling.", "How can I help?"]
    assert len(_sent(captured, "Flush")) == 3

    assert sum(isinstance(f, TTSStartedFrame) for f in down_frames) == 1
    assert sum(isinstance(f, TTSStoppedFrame) for f in down_frames) == 1
    word_frames = [f for f in down_frames if isinstance(f, TTSTextFrame)]
    assert [f.text for f in word_frames] == tokens
    # Word times are on the session's timeline, measured from the turn's first audio.
    pts = [f.pts for f in word_frames]
    assert pts == sorted(pts)
    assert pts[2] - pts[0] == CHUNK_MS * 2 * 1_000_000


@pytest.mark.asyncio
async def test_missing_word_boundaries_still_close_the_context(monkeypatch):
    """Without WordBoundaries the context closes after the grace period and keeps its text."""
    monkeypatch.setattr(assemblyai_tts, "_WORD_BOUNDARIES_GRACE_S", 0.05)
    captured: dict = {"messages": []}
    async with serve(_server(captured, word_boundaries=False), "127.0.0.1", 0) as server:
        down_frames, _ = await run_test(
            _service(server),
            frames_to_send=[
                TTSSpeakFrame(text="Hello from AssemblyAI."),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert sum(isinstance(f, TTSStoppedFrame) for f in down_frames) == 1
    text = " ".join(f.text for f in down_frames if isinstance(f, TTSTextFrame))
    assert text == "Hello from AssemblyAI."


@pytest.mark.asyncio
async def test_interruption_cancels_without_reconnecting():
    """An interruption sends Cancel, drops the cancelled audio and reuses the session."""
    captured: dict = {"messages": []}
    async with serve(_server(captured, stall_first_flush=True), "127.0.0.1", 0) as server:
        service = _service(server)
        down_frames, _ = await run_test(
            service,
            frames_to_send=[
                TTSSpeakFrame(text="This reply gets cut off."),
                SleepFrame(sleep=0.2),
                InterruptionFrame(),
                SleepFrame(sleep=0.1),
                TTSSpeakFrame(text="Go ahead."),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert len(_sent(captured, "Cancel")) == 1
    assert captured["connections"] == 1
    words = [f.text for f in down_frames if isinstance(f, TTSTextFrame)]
    assert words[-2:] == ["Go", "ahead."]
    assert not service.get_audio_contexts()


@pytest.mark.asyncio
async def test_interruption_after_synthesis_finished_sends_no_cancel():
    """Once every request has been answered there is nothing to cancel on the server."""
    captured: dict = {"messages": []}
    async with serve(_server(captured), "127.0.0.1", 0) as server:
        await run_test(
            _service(server),
            frames_to_send=[
                TTSSpeakFrame(text="Hello from AssemblyAI."),
                SleepFrame(sleep=0.3),
                InterruptionFrame(),
                SleepFrame(sleep=0.1),
            ],
        )

    assert not _sent(captured, "Cancel")


def _error(code: int, error_type: str | None, text: str, **fields) -> dict:
    error = {"error_code": code, "error": text}
    if error_type:
        error.update(error_type=error_type, retryable=False, param=None)
    error.update(fields)
    return error


@pytest.mark.asyncio
async def test_missing_voice_makes_the_service_unusable():
    """A voice that can't be found fails the same way on every retry."""
    captured: dict = {"messages": []}
    error = _error(
        3012,
        "voice_not_found",
        "The requested voice was not found.",
        param="voice_id",
    )
    async with serve(_server(captured, error_before_begin=error), "127.0.0.1", 0) as server:
        service = _service(server, settings=AssemblyAITTSService.Settings(voice="nobody"))
        _, up_frames = await run_test(service, frames_to_send=[SleepFrame(sleep=0.3)])

    assert not service.is_usable
    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert any("3012 voice_not_found" in f.error for f in errors)
    assert captured["connections"] == 1


@pytest.mark.asyncio
async def test_unspeakable_text_is_not_permanent():
    """A text the server can't split fails that text only; the service reconnects."""
    captured: dict = {"messages": []}
    error = _error(
        3006,
        "invalid_request",
        "Text contains a 900-character segment too long to synthesize. Add spaces or punctuation.",
    )
    server_handler = _server(captured, error_after_begin=error, error_once=True)
    async with serve(server_handler, "127.0.0.1", 0) as server:
        service = _service(server, reconnect_backoff_min_wait=0, reconnect_backoff_max_wait=0)
        _, up_frames = await run_test(service, frames_to_send=[SleepFrame(sleep=0.3)])

    assert service.is_usable
    assert any(isinstance(f, ErrorFrame) for f in up_frames)
    assert captured["connections"] == 2


@pytest.mark.asyncio
async def test_rejected_key_makes_the_service_unusable():
    """A credential refused before Begin fails the same way on every retry."""
    captured: dict = {"messages": []}
    error = _error(1008, "unauthorized", "Unauthorized: Invalid API key")
    async with serve(_server(captured, error_before_begin=error), "127.0.0.1", 0) as server:
        service = _service(server)
        _, up_frames = await run_test(service, frames_to_send=[SleepFrame(sleep=0.3)])

    assert not service.is_usable
    assert any("1008 unauthorized" in f.error for f in up_frames if isinstance(f, ErrorFrame))
    assert captured["connections"] == 1


@pytest.mark.asyncio
async def test_error_without_error_type_after_begin_is_not_permanent():
    """An error after Begin is reported but left to the reconnect to settle."""
    captured: dict = {"messages": []}
    error = _error(3006, None, "Unsupported preset voice 'nobody'.")
    server_handler = _server(captured, error_after_begin=error, error_once=True)
    async with serve(server_handler, "127.0.0.1", 0) as server:
        service = _service(server)
        _, up_frames = await run_test(service, frames_to_send=[SleepFrame(sleep=0.3)])

    assert service.is_usable
    assert any("error 3006: Unsupported" in f.error for f in up_frames if isinstance(f, ErrorFrame))
    assert captured["connections"] == 2


@pytest.mark.asyncio
async def test_routine_session_end_is_not_an_error():
    """An expired session reconnects without reporting an error."""
    captured: dict = {"messages": []}
    error = _error(
        3008,
        "session_expired",
        "Session reached its maximum duration; open a new connection to continue",
        retryable=True,
    )
    server_handler = _server(captured, error_after_begin=error, error_once=True)
    async with serve(server_handler, "127.0.0.1", 0) as server:
        _, up_frames = await run_test(_service(server), frames_to_send=[SleepFrame(sleep=0.3)])

    assert not any(isinstance(f, ErrorFrame) for f in up_frames)
    assert captured["connections"] == 2


@pytest.mark.asyncio
async def test_reconnect_waits_for_retry_after_seconds():
    """A retryable refusal is retried no sooner than the server asks."""
    captured: dict = {"messages": []}
    error = _error(
        3005,
        "at_capacity",
        "The TTS service is at capacity; please retry in 1 seconds.",
        retryable=True,
        retry_after_seconds=0.4,
    )
    server_handler = _server(captured, error_after_begin=error, error_once=True)
    async with serve(server_handler, "127.0.0.1", 0) as server:
        service = _service(server)
        await run_test(service, frames_to_send=[SleepFrame(sleep=1.5)])

    assert service.is_usable
    assert len(captured["connect_times"]) == 2
    first, second = captured["connect_times"]
    assert second - first >= 0.4


@pytest.mark.asyncio
async def test_begin_limits_size_generate_messages():
    """Generate texts are split at the limit Begin reports and joined as sent."""
    captured: dict = {"messages": []}
    limits = {"limits": {"max_generate_text_length": 8}}
    async with serve(_server(captured, configuration=limits), "127.0.0.1", 0) as server:
        await run_test(
            _service(server),
            frames_to_send=[
                SleepFrame(sleep=0.1),
                TTSSpeakFrame(text="Hello from AssemblyAI."),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    generates = [m["text"] for m in _sent(captured, "Generate")]
    assert all(len(text) <= 8 for text in generates)
    assert "".join(generates) == "Hello from AssemblyAI."


@pytest.mark.asyncio
async def test_idle_session_is_kept_alive(monkeypatch):
    """An idle session sends KeepAlive, since the server closes sessions that go quiet."""
    monkeypatch.setattr(assemblyai_tts, "_KEEPALIVE_INTERVAL_S", 0.1)
    captured: dict = {"messages": []}
    async with serve(_server(captured), "127.0.0.1", 0) as server:
        await run_test(_service(server), frames_to_send=[SleepFrame(sleep=0.35)])

    assert len(_sent(captured, "KeepAlive")) >= 2


@pytest.mark.asyncio
async def test_expiring_session_is_renewed_between_requests():
    """A session near expires_at is replaced before the next request, keeping the turn."""
    captured: dict = {"messages": []}
    async with serve(_server(captured, expires_in=1.0), "127.0.0.1", 0) as server:
        down_frames, _ = await run_test(
            _service(server),
            frames_to_send=[
                TTSSpeakFrame(text="First reply."),
                SleepFrame(sleep=0.8),
                BotStoppedSpeakingFrame(),
                TTSSpeakFrame(text="Second reply."),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert captured["connections"] == 2
    words = [f.text for f in down_frames if isinstance(f, TTSTextFrame)]
    assert words == ["First", "reply.", "Second", "reply."]


@pytest.mark.asyncio
async def test_session_renewed_mid_turn_keeps_word_times():
    """A turn that spans a renewal keeps its word times measured from its first audio."""
    captured: dict = {"messages": []}
    tokens = ["Hi ", "there! ", "Thanks ", "for ", "calling. ", "Bye."]
    async with serve(_server(captured, expires_in=1.0), "127.0.0.1", 0) as server:
        down_frames, _ = await run_test(
            _service(server),
            frames_to_send=[
                LLMFullResponseStartFrame(),
                *[LLMTextFrame(text=token) for token in tokens[:3]],
                # The first sentence is spoken and the session nears expires_at.
                SleepFrame(sleep=0.8),
                *[LLMTextFrame(text=token) for token in tokens[3:]],
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert captured["connections"] == 2
    word_frames = [f for f in down_frames if isinstance(f, TTSTextFrame)]
    assert [f.text for f in word_frames] == [token.strip() for token in tokens]
    # Each sentence's words follow the audio of the sentences before it.
    start = word_frames[0].pts
    offsets_ms = [(f.pts - start) // 1_000_000 for f in word_frames]
    assert offsets_ms == [
        0,
        WORD_MS,
        2 * CHUNK_MS,
        2 * CHUNK_MS + WORD_MS,
        2 * CHUNK_MS + 2 * WORD_MS,
        4 * CHUNK_MS,
    ]


@pytest.mark.asyncio
async def test_context_recreated_after_quiet_timeout_restarts_word_times():
    """Words after a context times out mid-turn are timed from the recreated context's audio."""
    captured: dict = {"messages": []}
    async with serve(_server(captured), "127.0.0.1", 0) as server:
        service = _service(server, stop_frame_timeout_s=0.3)
        word_times = []
        add_word_timestamps = service.add_word_timestamps

        async def record_word_times(times, context_id=None, **kwargs):
            word_times.extend(times)
            await add_word_timestamps(times, context_id, **kwargs)

        service.add_word_timestamps = record_word_times
        down_frames, _ = await run_test(
            service,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMTextFrame(text="Hi there! "),
                LLMTextFrame(text="Thanks "),
                # The turn goes quiet for longer than stop_frame_timeout_s.
                SleepFrame(sleep=0.8),
                LLMTextFrame(text="for calling. "),
                LLMTextFrame(text="Bye."),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert sum(isinstance(f, TTSStartedFrame) for f in down_frames) == 2
    assert [word for word, _ in word_times] == ["Hi", "there!", "Thanks", "for", "calling.", "Bye."]
    # The recreated context's words start again from zero.
    offsets_ms = [round(time * 1000) for _, time in word_times]
    assert offsets_ms == [0, WORD_MS, 0, WORD_MS, 2 * WORD_MS, 2 * CHUNK_MS]


def test_language_is_sent_when_set():
    service = AssemblyAITTSService(
        api_key="test-key",
        settings=AssemblyAITTSService.Settings(voice="aai-jane", language=Language.ES),
    )
    service._sample_rate = 16000
    query = parse_qs(urlparse(service._build_websocket_url()).query)
    assert query["voice_id"] == ["aai-jane"]
    assert query["language"] == ["spanish"]
    assert query["sample_rate"] == ["16000"]


def test_language_enum_maps_to_language_name():
    service = AssemblyAITTSService(
        api_key="test-key",
        settings=AssemblyAITTSService.Settings(language=Language.DE_DE),
    )
    assert service._settings.language == "german"


def test_default_voice_is_sent_and_language_is_left_to_the_server():
    service = AssemblyAITTSService(api_key="test-key")
    service._sample_rate = 24000
    query = parse_qs(urlparse(service._build_websocket_url()).query)
    assert query["voice_id"] == [assemblyai_tts._DEFAULT_VOICE_ID]
    assert "language" not in query


def test_voice_none_leaves_the_voice_to_the_server():
    service = AssemblyAITTSService(
        api_key="test-key", settings=AssemblyAITTSService.Settings(voice=None)
    )
    service._sample_rate = 24000
    query = parse_qs(urlparse(service._build_websocket_url()).query)
    assert "voice_id" not in query


def test_cloned_voice_and_model_are_sent():
    service = AssemblyAITTSService(
        api_key="test-key",
        settings=AssemblyAITTSService.Settings(
            voice="01a142a4-66f3-7315-b15f-668b6edfb595",
            model="assemblyai-tts-1",
        ),
    )
    service._sample_rate = 24000
    query = parse_qs(urlparse(service._build_websocket_url()).query)
    assert query["voice_id"] == ["01a142a4-66f3-7315-b15f-668b6edfb595"]
    assert query["model_id"] == ["assemblyai-tts-1"]
    assert "voice" not in query
    assert "language" not in query


def test_voice_id_key_sets_the_voice():
    """A settings update keyed ``voice_id``, as other services take it, sets the voice."""
    delta = AssemblyAITTSService.Settings.from_mapping({"voice_id": "aai-mary"})
    assert delta.voice == "aai-mary"
