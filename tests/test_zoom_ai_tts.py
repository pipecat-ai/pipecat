#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the Zoom TTS service, against a local HTTP server."""

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import ErrorFrame, InterruptionFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.services.zoom_ai.tts import MAX_INPUT_CHARS, ZoomAIHttpTTSService, _split_text
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.errors import ErrorCategory

PCM = b"\x01\x02\x03\x04" * 2048


class FakeZoomTTS:
    """A local stand-in for the Zoom TTS endpoint that records each request."""

    def __init__(self, reply=None):
        self.requests: list[dict] = []
        self._reply = reply or self._stream_pcm

    async def handler(self, request: web.Request) -> web.StreamResponse:
        self.requests.append({"headers": dict(request.headers), "raw": await request.read()})
        return await self._reply(request)

    @staticmethod
    async def _stream_pcm(request):
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm;rate=44100"})
        await response.prepare(request)
        # Split mid-sample, as network reads can.
        await response.write(PCM[:4095])
        await asyncio.sleep(0.01)
        await response.write(PCM[4095:])
        await response.write_eof()
        return response

    def body(self, n: int = 0) -> dict:
        return json.loads(self.requests[n]["raw"])


async def _run(aiohttp_client, fake, text="Hello from Zoom.", **kwargs):
    app = web.Application()
    app.router.add_post("/v2/aiservices/tts/synthesize", fake.handler)
    client = await aiohttp_client(app)
    async with aiohttp.ClientSession() as session:
        tts = ZoomAIHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/v2/aiservices/tts/synthesize")),
            sample_rate=44100,
            **kwargs,
        )
        down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame(text=text)])
    return tts, down, up


@pytest.mark.asyncio
async def test_streams_sample_aligned_audio_at_44100(aiohttp_client):
    fake = FakeZoomTTS()
    _, down, up = await _run(aiohttp_client, fake)

    audio = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert audio
    assert all(len(f.audio) % 2 == 0 for f in audio)
    assert all(f.sample_rate == 44100 for f in audio)
    assert b"".join(f.audio for f in audio) == PCM
    assert not [f for f in up if isinstance(f, ErrorFrame)]


@pytest.mark.asyncio
async def test_request_carries_text_voice_and_auth(aiohttp_client):
    fake = FakeZoomTTS()
    await _run(aiohttp_client, fake)

    assert fake.body() == {
        "input": "Hello from Zoom.",
        "response_format": "pcm_44100",
        "voice": "jenna",
    }
    headers = fake.requests[0]["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["x-zm-trackingid"].startswith("zoom_ai_community_pipecat_")


@pytest.mark.asyncio
async def test_settings_map_to_request_fields(aiohttp_client):
    fake = FakeZoomTTS()
    await _run(
        aiohttp_client,
        fake,
        settings=ZoomAIHttpTTSService.Settings(voice="ava", speed=1.2, volume=0.8),
    )

    body = fake.body()
    assert (body["voice"], body["speed"], body["volume"]) == ("ava", 1.2, 0.8)


@pytest.mark.asyncio
async def test_non_ascii_text_is_sent_as_utf8(aiohttp_client):
    """Unescaped UTF-8 keeps the longest input within the API's body limit."""
    fake = FakeZoomTTS()
    await _run(aiohttp_client, fake, text="Café crème.")

    raw = fake.requests[0]["raw"]
    assert "Café crème.".encode() in raw
    assert b"\\u" not in raw


@pytest.mark.asyncio
async def test_long_text_is_split_across_requests(aiohttp_client):
    fake = FakeZoomTTS()
    text = ("word " * (MAX_INPUT_CHARS // 4)).strip()
    await _run(aiohttp_client, fake, text=text)

    inputs = [fake.body(n)["input"] for n in range(len(fake.requests))]
    assert len(inputs) == 2
    assert all(len(i) <= MAX_INPUT_CHARS for i in inputs)
    assert " ".join(inputs) == text


def test_split_text_cuts_at_whitespace():
    assert _split_text("aaa bbb ccc", limit=7) == ["aaa bbb", "ccc"]
    assert _split_text("abcdefghij", limit=4) == ["abcd", "efgh", "ij"]
    assert _split_text("short") == ["short"]


@pytest.mark.asyncio
async def test_unsupported_voice_reports_zoom_reason_and_makes_the_service_unusable(
    aiohttp_client,
):
    async def reject(request):
        return web.json_response(
            {"code": 400, "reason": "UNSUPPORTED_VOICE", "message": "Unsupported voice"},
            status=400,
        )

    tts, down, up = await _run(
        aiohttp_client, FakeZoomTTS(reject), settings=ZoomAIHttpTTSService.Settings(voice="nobody")
    )

    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert len(errors) == 1
    assert "HTTP 400 UNSUPPORTED_VOICE Unsupported voice" in errors[0].error
    assert "x-zm-trackingid zoom_ai_community_pipecat_" in errors[0].error
    assert errors[0].category == ErrorCategory.INVALID_REQUEST
    assert not [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert not tts.is_usable


@pytest.mark.asyncio
async def test_unavailable_service_leaves_the_service_usable(aiohttp_client):
    async def unavailable(request):
        return web.json_response(
            {"code": 503, "reason": "SERVICE_UNAVAILABLE", "message": "Try again later"},
            status=503,
            headers={"Retry-After": "10"},
        )

    tts, _, up = await _run(aiohttp_client, FakeZoomTTS(unavailable))

    # The base service also reports the sentence that produced no audio.
    errors = [f for f in up if isinstance(f, ErrorFrame) and "Zoom TTS" in f.error]
    assert len(errors) == 1
    assert "HTTP 503 SERVICE_UNAVAILABLE" in errors[0].error
    assert errors[0].category == ErrorCategory.SERVER
    assert tts.is_usable


@pytest.mark.asyncio
async def test_stream_that_ends_early_is_reported(aiohttp_client):
    """A synthesis failure after the 200 ends the chunked response without its last chunk."""

    async def cut_off(request):
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(PCM[:4096])
        await asyncio.sleep(0.01)
        request.transport.close()
        return response

    _, down, up = await _run(aiohttp_client, FakeZoomTTS(cut_off))

    assert [f for f in down if isinstance(f, TTSAudioRawFrame)]
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert len(errors) == 1
    assert "stream ended early" in errors[0].error


@pytest.mark.asyncio
async def test_interruption_closes_the_request(aiohttp_client):
    """Closing the request is how Zoom TTS is told to stop synthesizing."""
    closed = asyncio.Event()

    async def slow(request):
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        try:
            for _ in range(100):
                await response.write(PCM[:4096])
                await asyncio.sleep(0.05)
        except (asyncio.CancelledError, ConnectionError):
            # The client went away: aiohttp cancels the handler, or the write fails.
            closed.set()
            raise
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v2/aiservices/tts/synthesize", FakeZoomTTS(slow).handler)
    client = await aiohttp_client(app)
    async with aiohttp.ClientSession() as session:
        tts = ZoomAIHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/v2/aiservices/tts/synthesize")),
            sample_rate=44100,
        )
        await run_test(
            tts,
            frames_to_send=[
                TTSSpeakFrame(text="A long reply the user talks over."),
                SleepFrame(sleep=0.3),
                InterruptionFrame(),
                SleepFrame(sleep=0.3),
            ],
        )
        # Checked while the session is open, since closing it closes every request.
        await asyncio.wait_for(closed.wait(), 1)
