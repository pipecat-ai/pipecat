#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime STT segment streaming, ordering, errors and metrics against a fake SDK."""

import asyncio

import pytest

pytest.importorskip("rimelabs_sdk")

from rimelabs_sdk import (
    RimeAuthenticationError,
    RimeInputError,
    RimePermissionError,
    RimeResourceLimitError,
    RimeStreamError,
    RimeTimeoutError,
    RimeUnavailableError,
    TranscriptionDelta,
    TranscriptionPartial,
)

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    MetricsFrame,
    STTMuteFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsageMetricsData, TTFBMetricsData
from pipecat.services.rime import stt as rime
from pipecat.services.rime.stt import RimeSTTService
from pipecat.utils.errors import ErrorCategory
from tests.rime_stt_helpers import FakeRime, audio, harness, next_stream, send, wait


@pytest.fixture
def client(monkeypatch):
    client = FakeRime()
    monkeypatch.setattr(rime, "Rime", lambda **kwargs: client)
    return client


async def speak(service, *frames):
    await send(service, VADUserStartedSpeakingFrame(), *frames, VADUserStoppedSpeakingFrame())


@pytest.mark.asyncio
async def test_stream_starts_with_audio_from_before_speech(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, _):
        before, speech = b"\x01\0" * 160, b"\x02\0" * 160
        await send(service, audio(before), VADUserStartedSpeakingFrame())
        stream = await next_stream(client)
        await send(service, audio(speech))
        await send(service, VADUserStoppedSpeakingFrame())
        await wait(stream.exhausted)
        assert b"".join(stream.chunks) == before + speech
        await stream.finish("hello")
        assert (await out.wait_for(TranscriptionFrame))[0].finalized
        assert len(client.streams) == 1


@pytest.mark.asyncio
async def test_audio_without_vad_is_not_streamed(client):
    service = RimeSTTService(api_key="test")
    async with harness(service):
        await send(service, *[audio() for _ in range(10)])
        assert not client.streams


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contract, updates, expected",
    [
        (
            "revised_hypotheses",
            [TranscriptionPartial(text="a red"), TranscriptionPartial(text="a blue")],
            ["a red", "a blue"],
        ),
        (
            "append_fragments",
            [TranscriptionDelta(text="hello "), TranscriptionDelta(text="world\nnext")],
            ["hello ", "hello world\nnext"],
        ),
    ],
)
async def test_interims_are_cumulative_for_both_contracts(client, contract, updates, expected):
    service = RimeSTTService(
        api_key="test", settings=RimeSTTService.Settings(output_contract=contract)
    )
    async with harness(service) as (out, _):
        await send(service, VADUserStartedSpeakingFrame(), audio())
        stream = await next_stream(client)
        assert stream.options["output_contract"].value == contract
        for index, update in enumerate(updates):
            await stream.updates.put(update)
            await out.wait_for(InterimTranscriptionFrame, index + 1)
        await send(service, VADUserStoppedSpeakingFrame())
        await wait(stream.exhausted)
        await stream.finish(expected[-1])
        assert [f.text for f in await out.wait_for(TranscriptionFrame)] == [expected[-1]]
        assert [f.text for f in out.frames if isinstance(f, InterimTranscriptionFrame)] == expected


@pytest.mark.asyncio
async def test_finals_keep_speech_order_and_each_segment_keeps_its_settings(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, _):
        await speak(service, audio())
        first = await next_stream(client)
        await send(service, STTUpdateSettingsFrame(delta=service.Settings(language="es-MX")))
        await speak(service, audio())
        second = await next_stream(client)
        assert first.options["language"] == "en"
        assert second.options["language"] == "es-MX"
        await wait(second.exhausted)
        await second.finish("segundo", "es")
        await wait(second.closed)
        await asyncio.sleep(0.01)
        assert not out.transcripts
        await first.finish("first")
        finals = await out.wait_for(TranscriptionFrame, 2)
        assert [f.text for f in finals] == ["first", "segundo"]
        assert [f.finalized for f in finals] == [False, True]


@pytest.mark.asyncio
async def test_final_is_not_finalized_once_the_user_speaks_again(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, _):
        await speak(service, audio())
        first = await next_stream(client)
        await send(service, VADUserStartedSpeakingFrame(), audio())
        await next_stream(client)
        await wait(first.exhausted)
        await first.finish("first")
        assert not (await out.wait_for(TranscriptionFrame))[0].finalized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error, category, usable",
    [
        (RimeAuthenticationError, ErrorCategory.AUTHENTICATION, False),
        (RimePermissionError, ErrorCategory.AUTHORIZATION, False),
        (RimeInputError, ErrorCategory.INVALID_REQUEST, False),
        (RimeTimeoutError, ErrorCategory.CONNECTIVITY, True),
        (RimeUnavailableError, ErrorCategory.CONNECTIVITY, True),
        (RimeResourceLimitError, ErrorCategory.UNKNOWN, True),
        (RimeStreamError, ErrorCategory.UNKNOWN, True),
    ],
)
async def test_stream_error_category_and_usability(client, error, category, usable):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, upstream):
        await send(service, VADUserStartedSpeakingFrame(), audio())
        stream = await next_stream(client)
        await stream.updates.put(TranscriptionPartial(text="unfinished"))
        await out.wait_for(InterimTranscriptionFrame)
        await stream.updates.put(error("echoed recognition hints", request_id="request-7"))
        errors = await upstream.wait_for(ErrorFrame)
        assert len(errors) == 1 and errors[0].category is category
        assert isinstance(errors[0].exception, error)
        assert "request-7" in errors[0].error
        assert "recognition hints" not in errors[0].error
        assert not out.transcripts
        assert service.is_usable is usable


@pytest.mark.asyncio
async def test_failed_segment_does_not_block_the_next(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, upstream):
        await speak(service, audio())
        first = await next_stream(client)
        await speak(service, audio())
        second = await next_stream(client)
        await wait(second.exhausted)
        await second.finish("second")
        await first.updates.put(RimeUnavailableError("down"))
        await upstream.wait_for(ErrorFrame)
        finals = await out.wait_for(TranscriptionFrame)
        assert [f.text for f in finals] == ["second"] and finals[0].finalized


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RimeAuthenticationError, RimePermissionError])
async def test_credential_failure_recovers_when_marked_usable(client, error):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, upstream):
        await send(service, VADUserStartedSpeakingFrame(), audio())
        first = await next_stream(client)
        await first.updates.put(error("rejected"))
        await upstream.wait_for(ErrorFrame)
        await wait(first.closed)
        await send(service, VADUserStoppedSpeakingFrame())
        assert not service.is_usable
        await speak(service, audio())
        assert len(client.streams) == 1

        await service.set_usable(True)
        await speak(service, audio())
        second = await next_stream(client)
        await wait(second.exhausted)
        await second.finish("hello again")
        assert [f.text for f in await out.wait_for(TranscriptionFrame)] == ["hello again"]


@pytest.mark.asyncio
async def test_startup_credential_failure_recovers_when_marked_usable(monkeypatch):
    client = FakeRime()
    attempts = 0

    def create(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RimeAuthenticationError("missing credential")
        return client

    monkeypatch.setattr(rime, "Rime", create)
    service = RimeSTTService()
    async with harness(service, require_client=False) as (out, upstream):
        errors = await upstream.wait_for(ErrorFrame)
        assert errors[0].category is ErrorCategory.AUTHENTICATION
        assert not service.is_usable
        await speak(service, audio())
        assert not client.streams and attempts == 1

        await service.set_usable(True)
        await speak(service, audio())
        stream = await next_stream(client)
        await wait(stream.exhausted)
        await stream.finish("hello")
        assert [f.text for f in await out.wait_for(TranscriptionFrame)] == ["hello"]
        assert attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
async def test_supported_rate_is_sent_as_is(client, rate):
    service = RimeSTTService(api_key="test", sample_rate=rate)
    async with harness(service):
        speech = bytes(rate * 2 // 10)
        await speak(service, audio(speech, rate=rate))
        stream = await next_stream(client)
        await wait(stream.exhausted)
        assert stream.options["input_format"].sample_rate == rate
        assert stream.options["input_format"].channels == 1
        assert b"".join(stream.chunks) == speech


@pytest.mark.asyncio
async def test_unsupported_rate_is_resampled(client):
    service = RimeSTTService(api_key="test", sample_rate=44100)
    async with harness(service):
        await speak(service, *[audio(b"\x01\0" * 4410, rate=44100) for _ in range(3)])
        stream = await next_stream(client)
        await wait(stream.exhausted)
        assert stream.options["input_format"].sample_rate == 16000
        assert 0 < len(b"".join(stream.chunks)) <= 3 * 3200


@pytest.mark.asyncio
async def test_usage_and_ttfb_are_reported_with_the_final(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, _):
        await send(service, audio(bytes(3200)), VADUserStartedSpeakingFrame())
        await send(service, audio(bytes(3200)), VADUserStoppedSpeakingFrame(stop_secs=0.1))
        stream = await next_stream(client)
        await wait(stream.exhausted)
        await stream.finish("done")
        await out.wait_for(TranscriptionFrame)
        metrics = [d for f in out.frames if isinstance(f, MetricsFrame) for d in f.data]
        usage = [d for d in metrics if isinstance(d, STTUsageMetricsData)]
        assert sum(d.value.audio_seconds for d in usage) == pytest.approx(0.2)
        assert len([d for d in metrics if isinstance(d, TTFBMetricsData)]) == 1


@pytest.mark.asyncio
async def test_mute_stops_audio_and_new_segments(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, _):
        speech = b"\x01\0" * 160
        await send(service, VADUserStartedSpeakingFrame(), audio(speech))
        stream = await next_stream(client)
        await send(service, STTMuteFrame(mute=True), audio(b"\x02\0" * 160))
        await send(service, VADUserStoppedSpeakingFrame())
        await wait(stream.exhausted)
        assert b"".join(stream.chunks) == speech
        await stream.finish("before mute")
        await out.wait_for(TranscriptionFrame)
        await speak(service, audio())
        assert len(client.streams) == 1


@pytest.mark.asyncio
async def test_interruption_keeps_the_segment_streaming(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, _):
        await send(service, VADUserStartedSpeakingFrame(), audio())
        stream = await next_stream(client)
        await send(service, InterruptionFrame(), audio(), VADUserStoppedSpeakingFrame())
        await wait(stream.exhausted)
        await stream.finish("barge in")
        assert [f.text for f in await out.wait_for(TranscriptionFrame)] == ["barge in"]


@pytest.mark.asyncio
async def test_end_frame_finishes_the_segment_in_progress(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, _):
        await send(service, VADUserStartedSpeakingFrame(), audio())
        stream = await next_stream(client)
        ending = asyncio.create_task(send(service, EndFrame()))
        await wait(stream.exhausted)
        await stream.finish("last words")
        await asyncio.wait_for(ending, 2)
        assert [f.text for f in out.transcripts] == ["last words"]
        assert client.closed


@pytest.mark.asyncio
async def test_cancel_drops_segments_without_errors(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, upstream):
        await send(service, VADUserStartedSpeakingFrame(), audio())
        stream = await next_stream(client)
        await asyncio.wait_for(send(service, CancelFrame()), 0.5)
        assert client.closed and stream.closed.is_set()
        assert not out.transcripts
        assert not any(isinstance(f, ErrorFrame) for f in upstream.frames)


@pytest.mark.asyncio
@pytest.mark.parametrize("second_text", ["second", ""])
async def test_user_turn_waits_for_the_latest_final(client, second_text):
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMUserAggregator,
        LLMUserAggregatorParams,
    )
    from pipecat.turns.user_start import VADUserTurnStartStrategy
    from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    context = LLMContext()
    aggregator = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.01)],
            )
        ),
        enable_direct_mode=True,
    )
    turns = []
    completed = asyncio.Event()

    @aggregator.event_handler("on_user_turn_stopped")
    async def on_stopped(aggregator, strategy, message):
        turns.append(message.content)
        completed.set()

    service = RimeSTTService(api_key="test", ttfs_p99_latency=60)
    async with harness(service, downstream=aggregator) as (out, _):
        await speak(service, audio())
        first = await next_stream(client)
        await speak(service, audio())
        second = await next_stream(client)
        await asyncio.sleep(0.03)
        await first.finish("first")
        await out.wait_for(TranscriptionFrame)
        assert not completed.is_set()
        await wait(second.exhausted)
        await second.finish(second_text)
        await wait(completed)
        assert turns == ["first second" if second_text else "first"]


@pytest.mark.asyncio
async def test_empty_final_adds_no_user_message(client):
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator

    context = LLMContext()
    aggregator = LLMUserAggregator(context, enable_direct_mode=True)
    service = RimeSTTService(api_key="test")
    async with harness(service, downstream=aggregator) as (out, _):
        await speak(service, audio())
        stream = await next_stream(client)
        await wait(stream.exhausted)
        await stream.finish("")
        assert (await out.wait_for(TranscriptionFrame))[0].finalized
        assert context.messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["upstream", "aggregator"])
async def test_vad_from_either_placement_drives_one_stream(client, placement):
    from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMUserAggregator,
        LLMUserAggregatorParams,
    )
    from pipecat.processors.audio.vad_processor import VADProcessor
    from pipecat.tests.utils import SleepFrame, run_test

    class TestVAD(VADAnalyzer):
        def num_frames_required(self):
            return 160

        def voice_confidence(self, buffer):
            return 1.0 if any(buffer) else 0.0

    client.automatic = True
    service = RimeSTTService(api_key="test")
    vad = TestVAD(params=VADParams(start_secs=0.01, stop_secs=0.01, min_volume=0))
    if placement == "upstream":
        processors = [VADProcessor(vad_analyzer=vad), service]
    else:
        processors = [
            service,
            LLMUserAggregator(LLMContext(), params=LLMUserAggregatorParams(vad_analyzer=vad)),
        ]
    speech = b"\x10\x10" * 160
    silence = bytes(320)
    frames = []
    for pcm in [speech] * 5 + [silence] * 5:
        frames.extend([audio(pcm), SleepFrame(sleep=0.015)])
    await run_test(Pipeline(processors), frames_to_send=frames)
    assert len(client.streams) == 1
    stream = client.streams[0]
    assert stream.exhausted.is_set()
    received = b"".join(stream.chunks)
    assert received.startswith(speech * 5)
    assert received.count(speech) == 5
