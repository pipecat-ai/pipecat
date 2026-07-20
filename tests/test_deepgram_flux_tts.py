#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for DeepgramFluxTTSService."""

import json
import unittest
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
import websockets
from websockets.asyncio.server import serve
from websockets.protocol import State

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
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.deepgram.flux.tts import DeepgramFluxTTSService
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.tests.utils import SleepFrame, run_test

AUDIO_CHUNK_1 = b"\x00\x01" * 512
AUDIO_CHUNK_2 = b"\x02\x03" * 512


def _flux_server_handler(captured: dict, *, warning_first: bool = False):
    """Build a fake Flux TTS server handler following the documented turn flow."""

    async def handler(ws):
        captured["request_path"] = ws.request.path
        captured["auth_header"] = ws.request.headers.get("Authorization")

        try:
            async for raw in ws:
                msg = json.loads(raw)
                captured["messages"].append(msg)
                if msg.get("type") == "Speak" and not captured.get("speech_started"):
                    captured["speech_started"] = True
                    await ws.send(
                        json.dumps(
                            {
                                "type": "Connected",
                                "request_id": "test-request",
                                "model_name": "flux-alexis-en",
                            }
                        )
                    )
                    if warning_first:
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "Warning",
                                    "code": "NO_ACTIVE_SPEECH",
                                    "description": "There is no active turn.",
                                }
                            )
                        )
                    await ws.send(json.dumps({"type": "SpeechStarted", "speech_id": "dg_sp_test"}))
                elif msg.get("type") == "Flush":
                    # Flux sends the flush ack before the turn's remaining
                    # audio; SpeechMetadata arrives only after all audio.
                    await ws.send(AUDIO_CHUNK_1)
                    await ws.send(json.dumps({"type": "Flushed", "speech_id": "dg_sp_test"}))
                    await ws.send(AUDIO_CHUNK_2)
                    await ws.send(
                        json.dumps(
                            {
                                "type": "SpeechMetadata",
                                "speech_id": "dg_sp_test",
                                "audio_duration_ms": 100,
                                "input_character_count": 17,
                                "billable_character_count": 17,
                            }
                        )
                    )
        except websockets.ConnectionClosed:
            pass

    return handler


@pytest.mark.asyncio
async def test_flux_tts_protocol_roundtrip():
    """Speak/Flush are sent, and audio (including post-Flushed audio) is emitted."""
    captured: dict = {"messages": []}

    async with serve(_flux_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts_service = DeepgramFluxTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/speak",
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            tts_service,
            frames_to_send=[
                TTSSpeakFrame(text="Hello from Flux."),
                SleepFrame(sleep=0.3),
                # With pause_frame_processing=True the transport's
                # BotStoppedSpeakingFrame resumes frame processing; there is no
                # transport in run_test, so send it explicitly (it is a system
                # frame, so it bypasses the paused queue).
                BotStoppedSpeakingFrame(),
            ],
        )

    frame_types = [type(frame) for frame in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSAudioRawFrame in frame_types
    assert TTSStoppedFrame in frame_types
    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)

    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert all(frame.sample_rate == 24000 for frame in audio_frames)
    assert all(frame.num_channels == 1 for frame in audio_frames)
    # Audio sent after Flushed but before SpeechMetadata must not be dropped.
    assert b"".join(frame.audio for frame in audio_frames) == AUDIO_CHUNK_1 + AUDIO_CHUNK_2

    assert captured["auth_header"] == "Token test-key"
    query = parse_qs(urlparse(captured["request_path"]).query)
    assert query["model"] == ["flux-alexis-en"]
    assert query["encoding"] == ["linear16"]
    assert query["sample_rate"] == ["24000"]

    types_sent = [m.get("type") for m in captured["messages"]]
    assert "Flush" in types_sent
    speak_msg = next(m for m in captured["messages"] if m.get("type") == "Speak")
    # In the default token streaming mode, text is sent verbatim.
    assert speak_msg["text"] == "Hello from Flux."


@pytest.mark.asyncio
async def test_flux_tts_token_streaming_sends_tokens_verbatim():
    """In the default TOKEN mode, LLM tokens map 1:1 to Speak messages, unaltered."""
    captured: dict = {"messages": []}

    async with serve(_flux_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts_service = DeepgramFluxTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/speak",
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            tts_service,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                LLMTextFrame("Unbelieva"),
                LLMTextFrame("ble"),
                LLMTextFrame(" isn't it?"),
                LLMFullResponseEndFrame(),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    assert any(isinstance(frame, TTSAudioRawFrame) for frame in down_frames)

    speak_texts = [m["text"] for m in captured["messages"] if m.get("type") == "Speak"]
    # No spaces may be inserted between tokens: Flux never strips whitespace
    # between Speak messages, so an added space would split words.
    assert speak_texts == ["Unbelieva", "ble", " isn't it?"]
    assert [m.get("type") for m in captured["messages"]].count("Flush") == 1


@pytest.mark.asyncio
async def test_flux_tts_sentence_mode_appends_trailing_space():
    """In SENTENCE mode a trailing space separates consecutive generations."""
    captured: dict = {"messages": []}

    async with serve(_flux_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts_service = DeepgramFluxTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/speak",
            sample_rate=24000,
            text_aggregation_mode=TextAggregationMode.SENTENCE,
        )

        down_frames, up_frames = await run_test(
            tts_service,
            frames_to_send=[
                TTSSpeakFrame(text="Hello from Flux."),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    speak_msg = next(m for m in captured["messages"] if m.get("type") == "Speak")
    assert speak_msg["text"] == "Hello from Flux. "


@pytest.mark.asyncio
async def test_flux_tts_interruption_reconnects_and_sends_no_clear():
    """On barge-in the service reconnects the websocket and sends no message."""
    tts_service = DeepgramFluxTTSService(api_key="test-key", sample_rate=24000)

    websocket = AsyncMock()
    websocket.state = State.OPEN
    tts_service._websocket = websocket
    tts_service._bot_speaking = True

    with (
        patch.object(TTSService, "_handle_interruption", new=AsyncMock()),
        patch.object(tts_service, "_connect", new=AsyncMock()) as connect_spy,
        patch.object(tts_service, "_disconnect", new=AsyncMock()) as disconnect_spy,
    ):
        await tts_service._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    assert disconnect_spy.called
    assert connect_spy.called
    # Flux has no Clear/Interrupt message; nothing must be sent on the socket.
    assert not websocket.send.called


@pytest.mark.asyncio
async def test_flux_tts_warning_is_not_fatal():
    """A server Warning is logged but does not produce an ErrorFrame."""
    captured: dict = {"messages": []}

    async with serve(_flux_server_handler(captured, warning_first=True), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts_service = DeepgramFluxTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/speak",
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            tts_service,
            frames_to_send=[
                TTSSpeakFrame(text="Hello from Flux."),
                SleepFrame(sleep=0.3),
                BotStoppedSpeakingFrame(),
            ],
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in down_frames + up_frames)
    assert any(isinstance(frame, TTSAudioRawFrame) for frame in down_frames)


@pytest.mark.asyncio
async def test_flux_tts_query_params():
    """Connection config and settings appear as query parameters."""
    captured: dict = {"messages": []}

    async with serve(_flux_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        tts_service = DeepgramFluxTTSService(
            api_key="test-key",
            url=f"ws://{host}:{port}/v2/speak",
            sample_rate=8000,
            mip_opt_out=True,
            tag=["tag-a", "tag-b"],
            settings=DeepgramFluxTTSService.Settings(voice="flux-thalia-en"),
        )

        await run_test(tts_service, frames_to_send=[])

    query = parse_qs(urlparse(captured["request_path"]).query)
    assert query["model"] == ["flux-thalia-en"]
    assert tts_service._settings.model == "flux-thalia-en"
    assert query["encoding"] == ["linear16"]
    assert query["sample_rate"] == ["8000"]
    assert query["mip_opt_out"] == ["true"]
    assert query["tag"] == ["tag-a", "tag-b"]


def test_flux_tts_sample_rate_validation():
    """Unsupported sample rates are rejected."""
    with pytest.raises(ValueError):
        DeepgramFluxTTSService(api_key="test-key", sample_rate=22050)


if __name__ == "__main__":
    unittest.main()
