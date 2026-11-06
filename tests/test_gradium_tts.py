#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for GradiumTTSService."""

import base64
import json

import pytest
import websockets
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    ErrorFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
)
from pipecat.services.gradium.tts import GradiumTTSService
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import SleepFrame, run_test

AUDIO_CHUNK = b"\x00\x01" * 512


def _ws_server_handler(captured: dict):
    """Build a fake Gradium TTS server: audio is returned once a stream ends."""

    async def handler(ws):
        try:
            async for raw in ws:
                msg = json.loads(raw)
                captured["messages"].append(msg)
                req_id = msg.get("client_req_id")
                if msg["type"] == "setup":
                    await ws.send(json.dumps({"type": "ready", "client_req_id": req_id}))
                elif msg["type"] == "end_of_stream":
                    audio = base64.b64encode(AUDIO_CHUNK).decode()
                    await ws.send(
                        json.dumps({"type": "audio", "audio": audio, "client_req_id": req_id})
                    )
                    await ws.send(json.dumps({"type": "end_of_stream", "client_req_id": req_id}))
        except websockets.ConnectionClosed:
            pass

    return handler


def _of_type(captured: dict, type: str) -> list[dict]:
    return [m for m in captured["messages"] if m["type"] == type]


async def _run(frames, **kwargs) -> tuple[dict, list, list]:
    captured: dict = {"messages": []}
    async with serve(_ws_server_handler(captured), "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        tts = GradiumTTSService(api_key="test-key", url=f"ws://{host}:{port}", **kwargs)
        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[*frames, SleepFrame(sleep=0.3), BotStoppedSpeakingFrame()],
        )
    assert not any(isinstance(f, ErrorFrame) for f in down_frames + up_frames)
    assert any(isinstance(f, TTSAudioRawFrame) for f in down_frames)
    return captured, down_frames, up_frames


@pytest.mark.asyncio
async def test_gradium_tts_streams_tokens_as_whole_words():
    """By default tokens are streamed, and a word split across tokens is sent whole."""
    captured, _, _ = await _run(
        [
            LLMFullResponseStartFrame(),
            LLMTextFrame("Unbelieva"),
            LLMTextFrame("ble"),
            LLMTextFrame(" isn't"),
            LLMTextFrame(" it?"),
            LLMFullResponseEndFrame(),
        ]
    )

    texts = [m["text"] for m in _of_type(captured, "text")]
    # Gradium ends a word at the end of each message, so none may end mid-word.
    assert texts == ["Unbelievable ", "isn't ", "it?"]
    # The held-back last word goes out before the stream is ended.
    assert [m["type"] for m in captured["messages"]][-2:] == ["text", "end_of_stream"]
    # One response is one Gradium stream.
    assert len(_of_type(captured, "setup")) == 1
    assert len({m["client_req_id"] for m in captured["messages"]}) == 1


@pytest.mark.asyncio
async def test_gradium_tts_sentence_mode_sends_sentences():
    """In SENTENCE mode each sentence goes out as one message, unchanged."""
    captured, _, _ = await _run(
        [TTSSpeakFrame(text="Hello from Gradium.")],
        text_aggregation_mode=TextAggregationMode.SENTENCE,
    )

    assert [m["text"] for m in _of_type(captured, "text")] == ["Hello from Gradium."]
