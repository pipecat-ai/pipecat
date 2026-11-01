#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Offline tests with fake Yoob servers, in the style of Pipecat's own service tests (no network, no key)."""

import asyncio
import json
import struct

import pytest
from aiohttp import web
from websockets.asyncio.server import serve

from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.yoob.stt import YoobSTTService
from pipecat.services.yoob.tts import YoobTTSService
from pipecat.tests.utils import SleepFrame, run_test


def _pcm48(seconds: float = 0.5) -> bytes:
    n = int(48000 * seconds)
    return struct.pack(f"<{n}h", *(((i * 97) % 2000) - 1000 for i in range(n)))


@pytest.fixture
async def fake_speech_api(aiohttp_unused_port=None):
    captured: dict = {}

    async def speech(request: web.Request):
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = await request.json()
        if captured["body"]["voice"] == "bad":
            return web.json_response(
                {"error": {"message": "Unknown voice 'bad'.", "code": "invalid_voice"}}, status=400
            )
        resp = web.StreamResponse(headers={"x-request-id": "req_1", "x-yoob-sample-rate": "48000"})
        await resp.prepare(request)
        audio = _pcm48()
        for i in range(0, len(audio), 4801):  # odd sizes: the service must keep samples aligned
            await resp.write(audio[i : i + 4801])
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/v1", captured
    await runner.cleanup()


async def test_tts_request_and_resampled_audio(fake_speech_api):
    base_url, captured = fake_speech_api
    tts = YoobTTSService(
        api_key="yoob_test_key",
        base_url=base_url,
        sample_rate=24000,
        settings=YoobTTSService.Settings(voice="m0770", instructions="warm voice", quality="fast"),
    )
    down, up = await run_test(
        tts, frames_to_send=[TTSSpeakFrame(text="[happy] Hello there."), SleepFrame(sleep=0.2)]
    )
    types = [type(f) for f in down]
    assert TTSStartedFrame in types and TTSAudioRawFrame in types and TTSStoppedFrame in types
    assert not any(isinstance(f, ErrorFrame) for f in list(down) + list(up))
    assert captured["auth"] == "Bearer yoob_test_key"
    assert captured["body"] == {
        "model": "yoob-tts-1",
        "voice": "m0770",
        "input": "[happy] Hello there.",
        "response_format": "pcm",
        "stream": True,
        "quality": "fast",
        "instructions": "warm voice",
    }
    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert all(f.sample_rate == 24000 and len(f.audio) % 2 == 0 for f in audio)
    total = sum(len(f.audio) for f in audio) // 2
    # 0.5 s at 24 kHz, less the streaming resampler's ~26 ms tail (Pipecat's helper does not flush it)
    assert 11000 <= total <= 12000


async def test_tts_error_becomes_error_frame(fake_speech_api):
    base_url, _ = fake_speech_api
    tts = YoobTTSService(
        api_key="k", base_url=base_url, settings=YoobTTSService.Settings(voice="bad")
    )
    down, up = await run_test(
        tts, frames_to_send=[TTSSpeakFrame(text="Hi."), SleepFrame(sleep=0.2)]
    )
    errors = [f for f in list(down) + list(up) if isinstance(f, ErrorFrame)]
    assert errors and "400" in errors[0].error and "Unknown voice" in errors[0].error


async def test_stt_protocol():
    captured: dict = {"text": [], "audio_bytes": 0}

    async def handler(ws):
        captured["auth"] = ws.request.headers.get("Authorization")
        await ws.send(
            json.dumps({"type": "ready", "id": "s1", "sample_rate": 16000, "encoder": "x"})
        )
        sent_partial = False
        async for msg in ws:
            if isinstance(msg, bytes):
                captured["audio_bytes"] += len(msg)
                if not sent_partial:
                    sent_partial = True
                    await ws.send(
                        json.dumps(
                            {"type": "partial", "utt": 1, "text": "hello", "start": 0.0, "end": 0.3}
                        )
                    )
                continue
            m = json.loads(msg)
            captured["text"].append(m)
            if m["type"] == "end":
                await ws.send(
                    json.dumps(
                        {
                            "type": "final",
                            "utt": 1,
                            "text": "hello there",
                            "start": 0.0,
                            "end": 0.6,
                            "emotion": "joy",
                            "caption": "[emo:joy] hello there",
                            "timing": {"x": 1},
                        }
                    )
                )

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        stt = YoobSTTService(
            api_key="yoob_test_key", base_url=f"http://{host}:{port}/v1", sample_rate=16000
        )
        frames = [
            InputAudioRawFrame(audio=b"\x01\x00" * 320, sample_rate=16000, num_channels=1)
            for _ in range(10)
        ]
        frames += [SleepFrame(sleep=0.2), VADUserStoppedSpeakingFrame(), SleepFrame(sleep=0.3)]
        down, up = await run_test(stt, frames_to_send=frames)

    assert captured["auth"] == "Bearer yoob_test_key"
    assert captured["audio_bytes"] == 6400
    assert {"type": "end"} in captured["text"]
    interims = [f for f in down if isinstance(f, InterimTranscriptionFrame)]
    finals = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert [f.text for f in interims] == ["hello"] and [f.text for f in finals] == ["hello there"]
    assert finals[0].result["emotion"] == "joy" and "timing" not in finals[0].result
    assert not any(isinstance(f, ErrorFrame) for f in list(down) + list(up))


def _audio_frames(n: int = 10):
    return [
        InputAudioRawFrame(audio=b"\x01\x00" * 320, sample_rate=16000, num_channels=1)
        for _ in range(n)
    ]


async def test_stt_end_waits_for_last_results():
    async def handler(ws):
        await ws.send(json.dumps({"type": "ready", "id": "s1", "sample_rate": 16000}))
        async for msg in ws:
            if isinstance(msg, bytes):
                continue
            if json.loads(msg)["type"] == "stop":
                await asyncio.sleep(
                    0.2
                )  # the last final comes after the client has stopped sending
                await ws.send(json.dumps({"type": "final", "utt": 1, "text": "last words"}))
                await ws.send(json.dumps({"type": "done"}))
                await ws.close()

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        stt = YoobSTTService(api_key="k", base_url=f"http://{host}:{port}/v1", sample_rate=16000)
        down, up = await run_test(stt, frames_to_send=_audio_frames() + [SleepFrame(sleep=0.2)])

    finals = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert [f.text for f in finals] == ["last words"]
    assert not any(isinstance(f, ErrorFrame) for f in list(down) + list(up))


async def test_stt_keeps_last_partial_when_the_socket_closes():
    connections = []

    async def handler(ws):
        connections.append(ws)
        await ws.send(json.dumps({"type": "ready", "id": "s1", "sample_rate": 16000}))
        if len(connections) == 1:
            await ws.recv()
            await ws.send(json.dumps({"type": "partial", "utt": 1, "text": "cut off"}))
            await ws.close()
            return
        async for msg in ws:
            if not isinstance(msg, bytes) and json.loads(msg)["type"] == "stop":
                await ws.send(json.dumps({"type": "done"}))
                await ws.close()

    async with serve(handler, "127.0.0.1", 0) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]
        stt = YoobSTTService(api_key="k", base_url=f"http://{host}:{port}/v1", sample_rate=16000)
        down, up = await run_test(stt, frames_to_send=_audio_frames(3) + [SleepFrame(sleep=1.5)])

    finals = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert [f.text for f in finals] == ["cut off"]
