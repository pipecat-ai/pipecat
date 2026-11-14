#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Deepgram segmented (pre-recorded API) STT service."""

import asyncio

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TranscriptionSpeaker,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.deepgram.stt import DeepgramSegmentedSTTService
from pipecat.services.stt_latency import DEEPGRAM_SEGMENTED_TTFS_P99
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory

WAV = b"RIFF....WAVEfmt "


def _result(transcript: str = "Hello there.", **channel) -> dict:
    """A pre-recorded response body with one channel and one alternative."""
    return {
        "metadata": {"request_id": "req-1", "duration": 1.0, "channels": 1},
        "results": {
            "channels": [
                {"alternatives": [{"transcript": transcript, "confidence": 0.99}], **channel}
            ]
        },
    }


def _listen_app(captured: dict, *, status: int = 200, body: dict | None = None):
    """An app serving /v1/listen, recording what each request carried."""

    async def handler(request):
        captured["headers"] = dict(request.headers)
        captured["query"] = list(request.query.items())
        captured["body"] = await request.read()
        return web.json_response(body if body is not None else _result(), status=status)

    app = web.Application()
    app.router.add_post("/v1/listen", handler)
    return app


async def _service(aiohttp_client, app, session, **kwargs) -> DeepgramSegmentedSTTService:
    """Build a service pointed at a test server running ``app``."""
    client = await aiohttp_client(app)
    return DeepgramSegmentedSTTService(
        api_key="test-key",
        aiohttp_session=session,
        base_url=str(client.make_url("/")),
        **kwargs,
    )


async def _run(aiohttp_client, app, **kwargs) -> list:
    """Transcribe one segment against ``app`` and return the frames yielded."""
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, app, session, **kwargs)
        return [frame async for frame in service.run_stt(WAV)]


async def _run_and_report(service: DeepgramSegmentedSTTService) -> ErrorFrame:
    """Run a segment and return the reported error.

    Errors are classified as the base class pushes them, not as ``run_stt``
    yields them, so a test reading ``category`` or ``is_usable`` has to go
    through ``process_generator`` the way the pipeline does.
    """
    pushed = []

    async def capture(frame, direction=None):
        pushed.append(frame)

    service.push_frame = capture
    await service.process_generator(service.run_stt(WAV))
    return pushed[-1]


#
# Settings
#


def test_defaults_use_nova_3_and_english():
    service = DeepgramSegmentedSTTService(api_key="k", aiohttp_session=object())

    assert service._settings.model == "nova-3-general"
    assert service._settings.language == "en"
    assert service._settings.punctuate is True


def test_ttfs_latency_defaults_to_the_service_constant():
    service = DeepgramSegmentedSTTService(api_key="k", aiohttp_session=object())

    assert service._ttfs_p99_latency == DEEPGRAM_SEGMENTED_TTFS_P99


#
# Request construction
#


@pytest.mark.asyncio
async def test_transcribe_posts_the_wav_segment_with_the_key(aiohttp_client):
    captured = {}
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, _listen_app(captured), session)

        result = await service._transcribe(WAV)

    assert result == _result()
    assert captured["headers"]["Authorization"] == "Token test-key"
    assert captured["headers"]["Content-Type"] == "audio/wav"
    assert captured["body"] == WAV
    assert captured["query"] == [
        ("model", "nova-3-general"),
        ("language", "en"),
        ("detect_entities", "false"),
        ("dictation", "false"),
        ("numerals", "false"),
        ("punctuate", "true"),
        ("smart_format", "false"),
        ("detect_language", "false"),
        ("filler_words", "false"),
    ]


@pytest.mark.asyncio
async def test_settings_are_sent_as_query_parameters(aiohttp_client):
    captured = {}
    await _run(
        aiohttp_client,
        _listen_app(captured),
        settings=DeepgramSegmentedSTTService.Settings(
            model="nova-3",
            language=Language.ES,
            punctuate=False,
            smart_format=True,
            keyterm=["Pipecat", "Daily"],
            version="2025-07-31.0",
        ),
    )

    query = captured["query"]
    assert ("model", "nova-3") in query
    assert ("language", "es") in query
    assert ("punctuate", "false") in query
    assert ("smart_format", "true") in query
    assert ("version", "2025-07-31.0") in query
    # A list is sent as a repeated parameter.
    assert [value for name, value in query if name == "keyterm"] == ["Pipecat", "Daily"]


@pytest.mark.asyncio
async def test_settings_set_to_none_are_left_out(aiohttp_client):
    captured = {}
    await _run(
        aiohttp_client,
        _listen_app(captured),
        settings=DeepgramSegmentedSTTService.Settings(language=None, keyterm=None, version=None),
    )

    names = {name for name, _ in captured["query"]}
    assert not names & {"language", "keyterm", "version"}


@pytest.mark.asyncio
async def test_speaker_format_is_not_sent(aiohttp_client):
    captured = {}
    await _run(
        aiohttp_client,
        _listen_app(captured),
        settings=DeepgramSegmentedSTTService.Settings(speaker_format="{speaker}: {text}"),
    )

    assert "speaker_format" not in {name for name, _ in captured["query"]}


@pytest.mark.asyncio
async def test_extra_and_init_options_are_sent(aiohttp_client):
    captured = {}
    await _run(
        aiohttp_client,
        _listen_app(captured),
        tag=["support", "eu"],
        mip_opt_out=True,
        settings=DeepgramSegmentedSTTService.Settings(extra={"utt_split": 0.5, "paragraphs": True}),
    )

    query = captured["query"]
    assert [value for name, value in query if name == "tag"] == ["support", "eu"]
    assert ("mip_opt_out", "true") in query
    assert ("utt_split", "0.5") in query
    assert ("paragraphs", "true") in query


@pytest.mark.asyncio
async def test_an_extra_key_naming_a_setting_sets_it(aiohttp_client):
    captured = {}
    async with aiohttp.ClientSession() as session:
        service = await _service(
            aiohttp_client,
            _listen_app(captured),
            session,
            settings=DeepgramSegmentedSTTService.Settings(extra={"smart_format": True}),
        )

        await service._transcribe(WAV)

    assert service._settings.smart_format is True
    assert "smart_format" not in service._settings.extra
    assert [value for name, value in captured["query"] if name == "smart_format"] == ["true"]


@pytest.mark.asyncio
async def test_a_settings_update_applies_to_the_next_request(aiohttp_client):
    captured = {}
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, _listen_app(captured), session)

        await service._update_settings(DeepgramSegmentedSTTService.Settings(smart_format=True))
        await service._transcribe(WAV)

    assert ("smart_format", "true") in captured["query"]


@pytest.mark.asyncio
async def test_an_extra_key_naming_a_setting_updates_it_at_runtime(aiohttp_client):
    captured = {}
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, _listen_app(captured), session)

        await service._update_settings(
            DeepgramSegmentedSTTService.Settings(extra={"smart_format": True})
        )
        await service._transcribe(WAV)

    assert service._settings.smart_format is True
    assert "smart_format" not in service._settings.extra
    assert [value for name, value in captured["query"] if name == "smart_format"] == ["true"]


#
# Transcription
#


@pytest.mark.asyncio
async def test_run_stt_yields_a_transcription_frame(aiohttp_client):
    frames = await _run(aiohttp_client, _listen_app({}))

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "Hello there."
    assert frames[0].language == "en"
    assert frames[0].result == _result()


@pytest.mark.asyncio
async def test_run_stt_yields_nothing_for_an_empty_transcript(aiohttp_client):
    frames = await _run(aiohttp_client, _listen_app({}, body=_result("")))

    assert frames == []


@pytest.mark.asyncio
async def test_run_stt_yields_nothing_for_a_result_without_channels(aiohttp_client):
    # Deepgram's answer to a WAV with no samples.
    body = {"metadata": {"duration": 0.0, "channels": 0}, "results": {"channels": []}}

    frames = await _run(aiohttp_client, _listen_app({}, body=body))

    assert frames == []


@pytest.mark.asyncio
async def test_a_detected_language_is_reported(aiohttp_client):
    body = _result("Hola.", detected_language="es")

    frames = await _run(aiohttp_client, _listen_app({}, body=body))

    assert frames[0].language == Language.ES


@pytest.mark.asyncio
async def test_a_multilingual_result_reports_its_first_language(aiohttp_client):
    body = _result("Hola.")
    body["results"]["channels"][0]["alternatives"][0]["languages"] = ["es", "en"]

    frames = await _run(aiohttp_client, _listen_app({}, body=body))

    assert frames[0].language == Language.ES


@pytest.mark.asyncio
async def test_a_detected_language_pipecat_does_not_know_is_reported_as_is(aiohttp_client):
    body = _result("Hello.", detected_language="xx")

    frames = await _run(aiohttp_client, _listen_app({}, body=body))

    assert frames[0].language == "xx"


#
# Diarization
#


def _diarized_result(transcript: str, words: list[dict]) -> dict:
    body = _result(transcript)
    body["results"]["channels"][0]["alternatives"][0]["words"] = words
    return body


TWO_SPEAKERS = _diarized_result(
    "Hi there. Hello.",
    [
        {"word": "hi", "punctuated_word": "Hi", "speaker": 0},
        {"word": "there", "punctuated_word": "there.", "speaker": 0},
        {"word": "hello", "punctuated_word": "Hello.", "speaker": 1},
    ],
)


def _names(captured: dict) -> list[str]:
    return [name for name, _ in captured["query"]]


@pytest.mark.asyncio
async def test_diarization_is_requested_by_model_alone(aiohttp_client):
    captured = {}
    await _run(
        aiohttp_client,
        _listen_app(captured),
        settings=DeepgramSegmentedSTTService.Settings(diarize=True),
    )

    # Deepgram refuses diarize alongside diarize_model.
    assert ("diarize_model", "latest") in captured["query"]
    assert "diarize" not in _names(captured)


@pytest.mark.asyncio
async def test_no_diarization_parameter_is_sent_when_diarization_is_off(aiohttp_client):
    captured = {}
    await _run(
        aiohttp_client,
        _listen_app(captured),
        settings=DeepgramSegmentedSTTService.Settings(diarize_model="v2"),
    )

    assert not {"diarize", "diarize_model"} & set(_names(captured))


@pytest.mark.asyncio
async def test_a_pinned_diarize_model_is_sent(aiohttp_client):
    captured = {}
    await _run(
        aiohttp_client,
        _listen_app(captured),
        settings=DeepgramSegmentedSTTService.Settings(diarize=True, diarize_model="v2"),
    )

    assert [value for name, value in captured["query"] if name == "diarize_model"] == ["v2"]


@pytest.mark.asyncio
async def test_a_diarize_model_in_extra_is_the_one_sent(aiohttp_client):
    captured = {}
    await _run(
        aiohttp_client,
        _listen_app(captured),
        settings=DeepgramSegmentedSTTService.Settings(diarize=True, extra={"diarize_model": "v1"}),
    )

    assert [value for name, value in captured["query"] if name == "diarize_model"] == ["v1"]


@pytest.mark.asyncio
async def test_a_diarized_result_yields_one_frame_per_speaker_run(aiohttp_client):
    frames = await _run(
        aiohttp_client,
        _listen_app({}, body=TWO_SPEAKERS),
        settings=DeepgramSegmentedSTTService.Settings(
            diarize=True, speaker_format="{speaker}: {text}"
        ),
    )

    assert [(f.speaker, f.text) for f in frames] == [
        (TranscriptionSpeaker("0", 1), "0: Hi there."),
        (TranscriptionSpeaker("1", 2), "1: Hello."),
    ]
    assert all(f.language == "en" for f in frames)


@pytest.mark.asyncio
async def test_unpunctuated_words_are_split_by_their_plain_text(aiohttp_client):
    body = _diarized_result(
        "hi there hello",
        [
            {"word": "hi", "speaker": 0},
            {"word": "there", "speaker": 0},
            {"word": "hello", "speaker": 1},
        ],
    )

    frames = await _run(
        aiohttp_client,
        _listen_app({}, body=body),
        settings=DeepgramSegmentedSTTService.Settings(diarize=True, punctuate=False),
    )

    assert [(f.speaker.label, f.text) for f in frames] == [("0", "hi there"), ("1", "hello")]


@pytest.mark.asyncio
async def test_words_without_speakers_yield_one_unlabelled_frame(aiohttp_client):
    body = _diarized_result("Hi there.", [{"word": "hi"}, {"word": "there"}])

    frames = await _run(
        aiohttp_client,
        _listen_app({}, body=body),
        settings=DeepgramSegmentedSTTService.Settings(diarize=True),
    )

    assert [(f.speaker, f.text) for f in frames] == [(None, "Hi there.")]


@pytest.mark.asyncio
async def test_speakers_are_ignored_when_diarization_is_off(aiohttp_client):
    frames = await _run(aiohttp_client, _listen_app({}, body=TWO_SPEAKERS))

    assert [(f.speaker, f.text) for f in frames] == [(None, "Hi there. Hello.")]


@pytest.mark.asyncio
async def test_only_the_last_speaker_run_of_a_segment_is_finalized(aiohttp_client):
    async with aiohttp.ClientSession() as session:
        service = await _service(
            aiohttp_client,
            _listen_app({}, body=TWO_SPEAKERS),
            session,
            sample_rate=16000,
            settings=DeepgramSegmentedSTTService.Settings(diarize=True),
        )

        received_down, _ = await run_test(
            service,
            frames_to_send=[
                VADUserStartedSpeakingFrame(),
                InputAudioRawFrame(audio=bytes(3200), sample_rate=16000, num_channels=1),
                VADUserStoppedSpeakingFrame(),
            ],
        )

    transcripts = [f for f in received_down if isinstance(f, TranscriptionFrame)]
    # A finalized transcript ends the user's turn, so it waits for every run.
    assert [(f.text, f.finalized) for f in transcripts] == [
        ("Hi there.", False),
        ("Hello.", True),
    ]


#
# Errors
#


@pytest.mark.asyncio
async def test_run_stt_names_the_error_code_message_and_request(aiohttp_client):
    body = {
        "err_code": "Bad Request",
        "err_msg": "No such model/language/tier combination found.",
        "request_id": "req-42",
    }

    frames = await _run(aiohttp_client, _listen_app({}, status=400, body=body))

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert "400 Bad Request - No such model/language/tier combination found." in frames[0].error
    assert "req-42" in frames[0].error


@pytest.mark.asyncio
async def test_an_error_does_not_carry_the_request_settings(aiohttp_client):
    body = {"err_code": "Bad Request", "err_msg": "Invalid keyterm.", "request_id": "r"}

    frames = await _run(
        aiohttp_client,
        _listen_app({}, status=400, body=body),
        settings=DeepgramSegmentedSTTService.Settings(keyterm=["Jane Doe"], replace=["foo:bar"]),
    )

    # The error text reaches the client and the logs; the query it was sent
    # with stays on the exception's request_info.
    assert "Jane" not in frames[0].error
    assert "foo:bar" not in frames[0].error
    assert "Jane+Doe" in str(frames[0].exception.request_info.url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, category",
    [
        (400, ErrorCategory.INVALID_REQUEST),
        (401, ErrorCategory.AUTHENTICATION),
        (403, ErrorCategory.AUTHORIZATION),
    ],
)
async def test_a_rejected_request_leaves_the_service_unusable(aiohttp_client, status, category):
    body = {"err_code": "REJECTED", "err_msg": "Rejected.", "request_id": "r"}
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, _listen_app({}, status=status, body=body), session)

        error = await _run_and_report(service)

    assert error.category is category
    # A rejected key or setting stays rejected, so the base class stops
    # handing the service work.
    assert service.is_usable is False


@pytest.mark.asyncio
async def test_a_settings_update_makes_a_rejected_service_usable_again(aiohttp_client):
    body = {"err_code": "INSUFFICIENT_PERMISSIONS", "err_msg": "No access.", "request_id": "r"}
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, _listen_app({}, status=403, body=body), session)
        await _run_and_report(service)

        await service._update_settings(DeepgramSegmentedSTTService.Settings(model="nova-2"))

    assert service.is_usable is True


@pytest.mark.asyncio
async def test_a_rate_limit_leaves_the_service_usable(aiohttp_client):
    body = {"err_code": "TOO_MANY_REQUESTS", "err_msg": "Too many requests.", "request_id": "r"}
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, _listen_app({}, status=429, body=body), session)

        error = await _run_and_report(service)

    assert error.category is ErrorCategory.RATE_LIMIT
    assert service.is_usable is True


@pytest.mark.asyncio
async def test_a_server_error_leaves_the_service_usable(aiohttp_client):
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, _listen_app({}, status=503, body={}), session)

        error = await _run_and_report(service)

    assert error.category is ErrorCategory.SERVER
    assert service.is_usable is True


@pytest.mark.asyncio
async def test_a_cancelled_request_leaves_the_session_usable(aiohttp_client):
    received = asyncio.Event()
    release = asyncio.Event()
    requests = 0

    async def handler(request):
        nonlocal requests
        requests += 1
        await request.read()
        if requests == 1:
            received.set()
            await release.wait()
        return web.json_response(_result())

    app = web.Application()
    app.router.add_post("/v1/listen", handler)
    async with aiohttp.ClientSession() as session:
        service = await _service(aiohttp_client, app, session)

        request = asyncio.create_task(service._transcribe(WAV))
        await received.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        release.set()

        result = await service._transcribe(WAV)

    assert result == _result()
    assert requests == 2
