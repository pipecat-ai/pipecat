#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Airy's HTTP streaming TTS service."""

import asyncio
import struct

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InterruptionFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData, TTSUsageMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.airy.tts import AiryHttpTTSService
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory


@pytest.mark.asyncio
async def test_streams_pcm_across_sample_boundaries(aiohttp_server):
    requests = []

    async def synthesize(request):
        requests.append((request.headers.get("Authorization"), await request.json()))
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        for chunk in (b"\x01", b"\x02\x03", b"\x04\x05\x06"):
            await response.write(chunk)
            await asyncio.sleep(0.01)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/audio/speech/stream", synthesize)
    server = await aiohttp_server(app)

    async with aiohttp.ClientSession() as session:
        tts = AiryHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(server.make_url("/")),
            sample_rate=24000,
            settings=AiryHttpTTSService.Settings(language=Language.KO),
        )
        down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame("안녕하세요.")])
        assert not session.closed

    assert requests == [
        (
            "Bearer test-key",
            {
                "input": "안녕하세요.",
                "model": "airy-tts-v1",
                "voice": "a597bb7a98fc9ec1",
                "language": "ko",
                "style": "normal",
            },
        )
    ]
    assert not up
    audio = [frame for frame in down if isinstance(frame, TTSAudioRawFrame)]
    assert b"".join(frame.audio for frame in audio) == b"\x01\x02\x03\x04\x05\x06"
    assert all(frame.sample_rate == 24000 and frame.num_channels == 1 for frame in audio)
    assert all(len(frame.audio) % 2 == 0 for frame in audio)
    started = next(frame for frame in down if isinstance(frame, TTSStartedFrame))
    stopped = next(frame for frame in down if isinstance(frame, TTSStoppedFrame))
    assert all(frame.context_id == started.context_id for frame in audio)
    assert stopped.context_id == started.context_id
    assert down.index(started) < down.index(audio[0]) < down.index(stopped)


@pytest_asyncio.fixture
async def make_service(aiohttp_server):
    async with aiohttp.ClientSession() as session:

        async def create(handler, **kwargs):
            app = web.Application()
            app.router.add_post("/v1/audio/speech/stream", handler)
            server = await aiohttp_server(app)
            return AiryHttpTTSService(
                api_key="test-key",
                aiohttp_session=session,
                base_url=str(server.make_url("/")),
                stop_frame_timeout_s=kwargs.pop("stop_frame_timeout_s", 0.1),
                **kwargs,
            )

        yield create


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate,expected_bytes", [(8000, 160), (16000, 320), (48000, 960)])
async def test_resampling_preserves_short_utterance_duration(
    make_service, sample_rate, expected_bytes
):
    async def synthesize(request):
        return web.Response(body=b"\x00\x10" * 240, content_type="audio/pcm")

    tts = await make_service(synthesize)
    down, up = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame("Hello.")],
        pipeline_params=PipelineParams(audio_out_sample_rate=sample_rate),
    )
    audio = [frame for frame in down if isinstance(frame, TTSAudioRawFrame)]
    assert not up
    assert sum(len(frame.audio) for frame in audio) == expected_bytes
    assert all(frame.sample_rate == sample_rate and frame.context_id for frame in audio)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code",
    [
        (402, "insufficient_credits"),
        (429, "inference_busy"),
        (500, "internal_error"),
    ],
)
async def test_reports_api_error_and_accepts_next_utterance(make_service, status, code):
    requests = []

    async def synthesize(request):
        requests.append(await request.json())
        if len(requests) == 1:
            return web.json_response(
                {
                    "error": {
                        "type": "api_error",
                        "code": code,
                        "message": "Request failed",
                        "param": None,
                    },
                    "request_id": "req_test",
                },
                status=status,
                headers={"Retry-After": "1"},
            )
        return web.Response(body=b"\x01\x02", content_type="audio/pcm")

    tts = await make_service(synthesize)
    down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame("First."), TTSSpeakFrame("Next.")])
    errors = [frame for frame in up if isinstance(frame, ErrorFrame)]
    assert any(
        str(status) in frame.error and code in frame.error and "req_test" in frame.error
        for frame in errors
    )
    assert all(not frame.fatal for frame in errors)
    assert all("test-key" not in frame.error for frame in errors)
    assert len(requests) == 2
    assert (
        b"".join(frame.audio for frame in down if isinstance(frame, TTSAudioRawFrame))
        == b"\x01\x02"
    )


@pytest.mark.asyncio
async def test_authentication_error_marks_service_unusable(make_service):
    async def synthesize(request):
        return web.json_response(
            {
                "error": {"code": "invalid_api_key", "message": "Rejected test-key"},
                "request_id": "req_auth",
            },
            status=401,
        )

    tts = await make_service(synthesize)
    _, up = await run_test(tts, frames_to_send=[TTSSpeakFrame("Hello.")])
    errors = [frame for frame in up if isinstance(frame, ErrorFrame)]
    assert any(
        frame.category == ErrorCategory.AUTHENTICATION
        and "401" in frame.error
        and "invalid_api_key" in frame.error
        and "req_auth" in frame.error
        for frame in errors
    )
    assert all(not frame.fatal and "test-key" not in frame.error for frame in errors)
    assert not tts.is_usable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,content_type,body",
    [
        (503, "text/html", b"<html>Unavailable</html>"),
        (200, "application/json", b'{"error":"failed"}'),
        (200, "audio/pcm", b"\x01\x02\x03"),
    ],
)
async def test_rejects_non_audio_or_incomplete_pcm(make_service, status, content_type, body):
    async def synthesize(request):
        return web.Response(status=status, content_type=content_type, body=body)

    tts = await make_service(synthesize)
    down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame("Hello.")])
    errors = [frame for frame in up if isinstance(frame, ErrorFrame)]
    assert errors
    assert all(not frame.fatal for frame in errors)
    audio = b"".join(frame.audio for frame in down if isinstance(frame, TTSAudioRawFrame))
    assert audio == (b"\x01\x02" if content_type == "audio/pcm" else b"")
    assert any(isinstance(frame, TTSStoppedFrame) for frame in down)


@pytest.mark.asyncio
async def test_runtime_settings_apply_to_next_request(make_service):
    requests = []

    async def synthesize(request):
        requests.append(await request.json())
        return web.Response(body=b"\x01\x02", content_type="audio/pcm")

    tts = await make_service(
        synthesize, settings=AiryHttpTTSService.Settings(language=Language.EN_US)
    )
    _, up = await run_test(
        tts,
        frames_to_send=[
            TTSSpeakFrame("Hello."),
            TTSUpdateSettingsFrame(
                delta=AiryHttpTTSService.Settings(
                    voice="bdb7de5e2cdd3324", language=Language.KO_KR, style="calm"
                )
            ),
            TTSSpeakFrame("안녕하세요."),
        ],
    )
    assert not up
    assert requests == [
        {
            "input": "Hello.",
            "model": "airy-tts-v1",
            "voice": "a597bb7a98fc9ec1",
            "language": "en",
            "style": "normal",
        },
        {
            "input": "안녕하세요.",
            "model": "airy-tts-v1",
            "voice": "bdb7de5e2cdd3324",
            "language": "ko",
            "style": "calm",
        },
    ]


@pytest.mark.asyncio
async def test_rejects_oversized_text_before_sending_request(make_service):
    requests = []

    async def synthesize(request):
        requests.append(await request.json())
        return web.Response(body=b"\x01\x02", content_type="audio/pcm")

    tts = await make_service(synthesize)
    _, up = await run_test(tts, frames_to_send=[TTSSpeakFrame("a" * 1281)])
    assert not requests
    assert any(
        isinstance(frame, ErrorFrame)
        and frame.category == ErrorCategory.INVALID_REQUEST
        and "1280" in frame.error
        for frame in up
    )


@pytest.mark.asyncio
async def test_timeout_releases_http_request(make_service):
    async def synthesize(request):
        await asyncio.sleep(0.2)
        return web.Response(body=b"\x01\x02", content_type="audio/pcm")

    tts = await make_service(synthesize, request_timeout=0.05)
    down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame("Hello.")])
    assert any(isinstance(frame, ErrorFrame) and "timed out" in frame.error for frame in up)
    assert not any(isinstance(frame, TTSAudioRawFrame) for frame in down)


@pytest.mark.asyncio
async def test_audio_and_metrics_arrive_before_response_finishes(make_service):
    first_audio = asyncio.Event()

    class ObserveAudio(BaseObserver):
        async def on_push_frame(self, data: FramePushed):
            if data.source is tts and isinstance(data.frame, TTSAudioRawFrame):
                first_audio.set()

    async def synthesize(request):
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(b"\x01\x02")
        await asyncio.wait_for(first_audio.wait(), timeout=1)
        await response.write(b"\x03\x04")
        await response.write_eof()
        return response

    tts = await make_service(synthesize)
    down, up = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame("Hello.")],
        observers=[ObserveAudio()],
        pipeline_params=PipelineParams(
            enable_metrics=True, enable_usage_metrics=True, send_initial_empty_metrics=False
        ),
    )
    assert not up
    assert (
        b"".join(frame.audio for frame in down if isinstance(frame, TTSAudioRawFrame))
        == b"\x01\x02\x03\x04"
    )
    metrics = [metric for frame in down if isinstance(frame, MetricsFrame) for metric in frame.data]
    assert len([metric for metric in metrics if isinstance(metric, TTFBMetricsData)]) == 1
    assert [metric.value for metric in metrics if isinstance(metric, TTSUsageMetricsData)] == [6]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True], ids=["interruption", "cancel"])
async def test_interruption_closes_response_and_discards_resampler_tail(make_service, cancel):
    first_request_transport = None
    release_first = asyncio.Event()
    requests = []
    interrupted_context = None
    next_context = None
    second_audio = []
    first_closed_before_next = False

    class InterruptAfterAudio(BaseObserver):
        async def on_push_frame(self, data: FramePushed):
            nonlocal interrupted_context, next_context
            if data.source is not tts:
                return
            frame = data.frame
            if isinstance(frame, TTSAudioRawFrame):
                if interrupted_context is None:
                    interrupted_context = frame.context_id
                    await tts.pipeline_worker.queue_frame(
                        CancelFrame() if cancel else InterruptionFrame()
                    )
                elif frame.context_id != interrupted_context:
                    next_context = frame.context_id
                    second_audio.append(frame.audio)
            elif isinstance(frame, InterruptionFrame):
                await tts.pipeline_worker.queue_frames([TTSSpeakFrame("Next."), EndFrame()])

    async def synthesize(request):
        nonlocal first_request_transport, first_closed_before_next
        requests.append(await request.json())
        if len(requests) > 1:
            first_closed_before_next = first_request_transport.is_closing()
            release_first.set()
            return web.Response(body=b"\x00\x00" * 240, content_type="audio/pcm")
        first_request_transport = request.transport
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(b"\x00\x20" * 24000)
        await release_first.wait()
        return response

    tts = await make_service(synthesize, sample_rate=16000)
    try:
        _, up = await asyncio.wait_for(
            run_test(
                tts,
                frames_to_send=[TTSSpeakFrame("First.")],
                observers=[InterruptAfterAudio()],
                send_end_frame=False,
            ),
            timeout=5,
        )
        assert not any(isinstance(frame, ErrorFrame) for frame in up)
        assert first_request_transport.is_closing()
        assert not tts._session.closed
        if not cancel:
            assert first_closed_before_next
            assert [request["input"] for request in requests] == ["First.", "Next."]
            assert next_context and next_context != interrupted_context
            audio = b"".join(second_audio)
            assert len(audio) == 320
            # SoXR's int16 output dithers silence by at most one sample unit.
            assert max(abs(sample[0]) for sample in struct.iter_unpack("<h", audio)) <= 1
    finally:
        release_first.set()


@pytest.mark.asyncio
async def test_empty_audio_returns_error_and_next_request_still_works(make_service):
    requests = []

    async def synthesize(request):
        requests.append(await request.json())
        return web.Response(
            body=b"" if len(requests) == 1 else b"\x01\x02", content_type="audio/pcm"
        )

    tts = await make_service(synthesize)
    down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame("First."), TTSSpeakFrame("Next.")])
    assert any(isinstance(frame, ErrorFrame) and "Empty audio" in frame.error for frame in up)
    assert (
        b"".join(frame.audio for frame in down if isinstance(frame, TTSAudioRawFrame))
        == b"\x01\x02"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate,expected_bytes", [(8000, 320), (16000, 640), (48000, 1920)])
async def test_network_gap_preserves_buffered_audio(make_service, sample_rate, expected_bytes):
    async def synthesize(request):
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(b"\x00\x10" * 240)
        await asyncio.sleep(0.25)
        await response.write(b"\x00\x10" * 240)
        await response.write_eof()
        return response

    tts = await make_service(synthesize, sample_rate=sample_rate, stop_frame_timeout_s=1)
    down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame("Hello.")])
    assert not up
    assert (
        sum(len(frame.audio) for frame in down if isinstance(frame, TTSAudioRawFrame))
        == expected_bytes
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("delay_before_headers", [False, True])
async def test_pending_request_keeps_audio_context_open(make_service, delay_before_headers):
    async def synthesize(request):
        if delay_before_headers:
            await asyncio.sleep(0.25)
        response = web.StreamResponse(headers={"Content-Type": "audio/pcm"})
        await response.prepare(request)
        await response.write(b"\x01\x02")
        if not delay_before_headers:
            await asyncio.sleep(0.25)
        await response.write(b"\x03\x04")
        await response.write_eof()
        return response

    tts = await make_service(synthesize, stop_frame_timeout_s=0.1, request_timeout=1)
    down, up = await run_test(tts, frames_to_send=[TTSSpeakFrame("Hello.")])
    assert not up
    audio = [frame for frame in down if isinstance(frame, TTSAudioRawFrame)]
    started = [frame for frame in down if isinstance(frame, TTSStartedFrame)]
    stopped = [frame for frame in down if isinstance(frame, TTSStoppedFrame)]
    assert len(started) == len(stopped) == 1
    assert b"".join(frame.audio for frame in audio) == b"\x01\x02\x03\x04"
    assert down.index(started[0]) < down.index(audio[0])
    assert down.index(audio[-1]) < down.index(stopped[0])
