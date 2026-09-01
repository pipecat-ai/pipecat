#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the AssemblyAI Sync STT service."""

import json

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import (
    ErrorFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.services.assemblyai.stt import AssemblyAISyncSTTService
from pipecat.services.stt_latency import ASSEMBLYAI_SYNC_TTFS_P99
from pipecat.transcriptions.language import Language

WAV = b"RIFF....WAVEfmt "


def _transcribe_app(captured: dict, *, status: int = 200, body: dict | None = None):
    """An app serving /v1/transcribe, recording what each request carried."""

    async def handler(request):
        captured["headers"] = dict(request.headers)
        captured["parts"] = {}
        reader = await request.multipart()
        async for part in reader:
            if part.name == "audio":
                captured["parts"]["audio"] = await part.read()
                captured["audio_type"] = part.headers.get("Content-Type")
                captured["audio_filename"] = part.filename
            else:
                captured["parts"][part.name] = await part.text()
        return web.json_response(
            body if body is not None else {"text": "Hello there", "words": []},
            status=status,
        )

    app = web.Application()
    app.router.add_post("/v1/transcribe", handler)
    return app


async def _service(aiohttp_client, app, session, **kwargs) -> AssemblyAISyncSTTService:
    """Build a service pointed at a test server running ``app``."""
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")
    return AssemblyAISyncSTTService(
        api_key="test-key",
        aiohttp_session=session,
        base_url=base_url,
        **kwargs,
    )


def _config(captured: dict) -> dict:
    return json.loads(captured["parts"]["config"])


#
# Settings
#


def test_defaults_use_the_sync_model_and_english():
    service = AssemblyAISyncSTTService(api_key="k", aiohttp_session=object())

    assert service._settings.model == "universal-3-5-pro"
    # The base class converts the Language enum to AssemblyAI's code at init.
    assert service._settings.language == "en"


def test_ttfs_latency_defaults_to_the_service_constant():
    service = AssemblyAISyncSTTService(api_key="k", aiohttp_session=object())

    assert service._ttfs_p99_latency == ASSEMBLYAI_SYNC_TTFS_P99


def test_language_converts_to_the_assemblyai_code():
    service = AssemblyAISyncSTTService(api_key="k", aiohttp_session=object())

    assert service.language_to_service_language(Language.ES_US) == "es"


#
# Request construction
#


@pytest.mark.asyncio
async def test_transcribe_posts_the_audio_and_config_parts(aiohttp_client):
    captured = {}
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, _transcribe_app(captured), session)

        result = await service._transcribe(WAV)

    assert result["text"] == "Hello there"
    assert captured["headers"]["Authorization"] == "test-key"
    assert captured["headers"]["X-AAI-Model"] == "universal-3-5-pro"
    assert captured["parts"]["audio"] == WAV
    assert captured["audio_type"] == "audio/wav"
    assert captured["audio_filename"] == "audio.wav"
    assert _config(captured) == {"language_codes": ["en"]}


@pytest.mark.asyncio
async def test_config_carries_prompt_and_keyterms(aiohttp_client):
    captured = {}
    async with aiohttp.ClientSession() as session:
        service = await _service(
            aiohttp_client,
            _transcribe_app(captured),
            session,
            settings=AssemblyAISyncSTTService.Settings(
                prompt="Transcribe this call.",
                keyterms_prompt=["Pipecat", "AssemblyAI"],
            ),
        )

        await service._transcribe(WAV)

    config = _config(captured)
    assert config["prompt"] == "Transcribe this call."
    assert config["keyterms_prompt"] == ["Pipecat", "AssemblyAI"]


@pytest.mark.asyncio
async def test_config_part_is_omitted_when_nothing_applies(aiohttp_client):
    captured = {}
    async with aiohttp.ClientSession() as session:
        service = await _service(
            aiohttp_client,
            _transcribe_app(captured),
            session,
            settings=AssemblyAISyncSTTService.Settings(language=None),
        )

        await service._transcribe(WAV)

    assert "config" not in captured["parts"]


#
# Transcription
#


@pytest.mark.asyncio
async def test_run_stt_yields_a_transcription_frame(aiohttp_client):
    captured = {}
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, _transcribe_app(captured), session)

        frames = [frame async for frame in service.run_stt(WAV)]

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "Hello there"
    assert frames[0].language == "en"
    assert frames[0].result == {"text": "Hello there", "words": []}


@pytest.mark.asyncio
async def test_run_stt_yields_nothing_for_an_empty_transcript(aiohttp_client):
    captured = {}
    app = _transcribe_app(captured, body={"text": "   ", "words": []})
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, app, session)

        frames = [frame async for frame in service.run_stt(WAV)]

    assert frames == []


@pytest.mark.asyncio
async def test_run_stt_yields_an_error_frame_on_a_problem_details_body(aiohttp_client):
    captured = {}
    app = _transcribe_app(
        captured,
        status=400,
        body={"status": 400, "title": "Bad Request", "detail": "invalid config part"},
    )
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, app, session)

        frames = [frame async for frame in service.run_stt(WAV)]

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert "Bad Request - invalid config part" in frames[0].error


@pytest.mark.asyncio
async def test_run_stt_surfaces_the_message_of_an_error_code_body(aiohttp_client):
    captured = {}
    app = _transcribe_app(
        captured,
        status=413,
        body={"error_code": "audio_too_large", "message": "audio exceeds 120 seconds"},
    )
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, app, session)

        frames = [frame async for frame in service.run_stt(WAV)]

    assert isinstance(frames[0], ErrorFrame)
    assert "audio exceeds 120 seconds" in frames[0].error


#
# Conversation context
#


@pytest.mark.asyncio
async def test_a_turn_is_absent_from_its_own_request_and_present_in_the_next(aiohttp_client):
    captured = {}
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, _transcribe_app(captured), session)

        [frame async for frame in service.run_stt(WAV)]
        assert "conversation_context" not in _config(captured)

        [frame async for frame in service.run_stt(WAV)]

    assert _config(captured)["conversation_context"] == ["Hello there"]


@pytest.mark.asyncio
async def test_agent_replies_share_the_buffer_in_the_order_spoken(aiohttp_client):
    captured = {}
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, _transcribe_app(captured), session)

        [frame async for frame in service.run_stt(WAV)]
        await service._process_assistant_turn("How can I help?")
        [frame async for frame in service.run_stt(WAV)]

    assert _config(captured)["conversation_context"] == ["Hello there", "How can I help?"]


def test_context_evicts_the_oldest_turn_past_the_turn_cap():
    service = AssemblyAISyncSTTService(api_key="k", aiohttp_session=object(), max_context_turns=2)

    for turn in ("one", "two", "three"):
        service._append_context_turn(turn)

    assert service._context_turns == ["two", "three"]


def test_context_evicts_the_oldest_turn_past_the_char_cap():
    service = AssemblyAISyncSTTService(api_key="k", aiohttp_session=object(), max_context_chars=10)

    service._append_context_turn("aaaaa")
    service._append_context_turn("bbbbb")
    service._append_context_turn("ccccc")

    # Eviction stops as soon as the buffer is back within budget.
    assert service._context_turns == ["bbbbb", "ccccc"]


def test_a_turn_longer_than_the_char_cap_is_kept_alone():
    service = AssemblyAISyncSTTService(api_key="k", aiohttp_session=object(), max_context_chars=10)

    service._append_context_turn("aaaaa")
    service._append_context_turn("b" * 40)

    assert service._context_turns == ["b" * 40]


def test_blank_turns_are_not_buffered():
    service = AssemblyAISyncSTTService(api_key="k", aiohttp_session=object())

    service._append_context_turn("   ")

    assert service._context_turns == []


@pytest.mark.asyncio
async def test_zero_max_context_turns_disables_the_buffer(aiohttp_client):
    captured = {}
    async with aiohttp.ClientSession() as session:
        service = await _service(
            aiohttp_client, _transcribe_app(captured), session, max_context_turns=0
        )

        [frame async for frame in service.run_stt(WAV)]
        [frame async for frame in service.run_stt(WAV)]

    assert service._context_turns == []
    assert "conversation_context" not in _config(captured)


@pytest.mark.asyncio
async def test_an_explicit_context_is_sent_as_is_and_stops_buffering(aiohttp_client):
    captured = {}
    async with aiohttp.ClientSession() as session:
        service = await _service(
            aiohttp_client,
            _transcribe_app(captured),
            session,
            settings=AssemblyAISyncSTTService.Settings(
                conversation_context=["Booking a flight to Lisbon."]
            ),
        )

        [frame async for frame in service.run_stt(WAV)]
        [frame async for frame in service.run_stt(WAV)]

    assert _config(captured)["conversation_context"] == ["Booking a flight to Lisbon."]
    assert service._context_turns == []


#
# Pre-warming
#


@pytest.mark.asyncio
async def test_warm_gets_the_warm_path_with_the_model_and_no_auth(aiohttp_client):
    captured = {}

    async def handler(request):
        captured["headers"] = dict(request.headers)
        return web.json_response({"warm": "toasty"})

    app = web.Application()
    app.router.add_get("/v1/warm", handler)

    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, app, session)

        await service.warm()

    assert captured["headers"]["X-AAI-Model"] == "universal-3-5-pro"
    assert "Authorization" not in captured["headers"]


@pytest.mark.asyncio
async def test_a_failed_warm_is_swallowed(aiohttp_client):
    async def handler(request):
        return web.Response(status=500)

    app = web.Application()
    app.router.add_get("/v1/warm", handler)

    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, app, session)

        # A failed warm only forfeits the latency saving; it must not raise.
        await service.warm()


@pytest.mark.asyncio
async def test_speech_start_schedules_a_warm():
    service = AssemblyAISyncSTTService(api_key="k", aiohttp_session=object())
    scheduled = []

    def create_task(coro, *args, **kwargs):
        scheduled.append(coro)
        coro.close()
        return None

    service.create_task = create_task

    await service._handle_user_started_speaking(VADUserStartedSpeakingFrame())

    assert len(scheduled) == 1


@pytest.mark.asyncio
async def test_speech_start_schedules_no_warm_when_pre_warming_is_off():
    service = AssemblyAISyncSTTService(
        api_key="k", aiohttp_session=object(), enable_prewarming=False
    )
    scheduled = []

    def create_task(coro, *args, **kwargs):
        scheduled.append(coro)
        coro.close()
        return None

    service.create_task = create_task

    await service._handle_user_started_speaking(VADUserStartedSpeakingFrame())

    assert scheduled == []


@pytest.mark.asyncio
async def test_cleanup_cancels_a_pending_warm():
    service = AssemblyAISyncSTTService(api_key="k", aiohttp_session=object())
    cancelled = []

    class PendingTask:
        def done(self):
            return False

    async def cancel_task(task, *args, **kwargs):
        cancelled.append(task)

    service._warm_task = PendingTask()
    service.cancel_task = cancel_task

    await service.cleanup()

    assert len(cancelled) == 1
    assert service._warm_task is None
