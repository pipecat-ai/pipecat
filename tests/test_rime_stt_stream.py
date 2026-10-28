#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Ordered streaming transcription, bounded buffering and SDK teardown."""

import asyncio
import time

import pytest
from rimelabs_sdk import (
    RimeAuthenticationError,
    RimeCancelledError,
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
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.rime import stt as rime
from pipecat.services.rime.stt import RimeSTTService
from pipecat.utils.errors import ErrorCategory
from tests.rime_stt_helpers import FakeRime, audio, harness, next_stream, send, wait


@pytest.fixture
def client(monkeypatch):
    client = FakeRime()
    monkeypatch.setattr(rime, "Rime", lambda **kwargs: client)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", list(FrameDirection))
async def test_prefix_uploaded_before_stop_without_duplicates(client, direction):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, _):
        prefix, speech = b"\x01\0" * 160, b"\x02\0" * 160
        await send(service, audio(prefix))
        await send(service, VADUserStartedSpeakingFrame(), direction=direction)
        stream = await next_stream(client)
        await wait(stream.uploaded)
        assert b"".join(stream.chunks) == prefix
        await send(service, VADUserStartedSpeakingFrame(), direction=direction)
        await send(service, audio(speech))
        await send(
            service,
            VADUserStoppedSpeakingFrame(),
            VADUserStoppedSpeakingFrame(),
            direction=direction,
        )
        await wait(stream.exhausted)
        assert b"".join(stream.chunks) == prefix + speech
        await stream.finish("hello")
        assert (await out.wait_for(TranscriptionFrame))[0].finalized
        assert len(client.streams) == 1


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
async def test_transcript_contract(client, contract, updates, expected):
    service = RimeSTTService(
        api_key="test", settings=RimeSTTService.Settings(output_contract=contract)
    )
    async with harness(service) as (out, _):
        await send(service, VADUserStartedSpeakingFrame(), audio())
        stream = await next_stream(client)
        for index, update in enumerate(updates):
            await stream.updates.put(update)
            await out.wait_for(InterimTranscriptionFrame, index + 1)
        await send(service, VADUserStoppedSpeakingFrame())
        await wait(stream.exhausted)
        await stream.finish(expected[-1])
        final = await out.wait_for(TranscriptionFrame)
        assert [f.text for f in final] == [expected[-1]]
        assert [f.text for f in out.frames if isinstance(f, InterimTranscriptionFrame)] == expected


@pytest.mark.asyncio
async def test_slow_first_fast_second_order_and_settings(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, _):
        await send(service, VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame())
        first = await next_stream(client)
        await send(
            service,
            STTUpdateSettingsFrame(delta=service.Settings(language="es-MX")),
            VADUserStartedSpeakingFrame(),
            audio(),
        )
        second = await next_stream(client)
        await wait(second.uploaded)
        assert first.options["language"] == "en"
        assert second.options["language"] == "es-MX"
        await second.updates.put(TranscriptionPartial(text="segundo"))
        await send(service, VADUserStoppedSpeakingFrame())
        await wait(second.exhausted)
        await second.finish("segundo", "es")
        await wait(second.closed)
        assert not out.transcripts
        await first.finish("first")
        finals = await out.wait_for(TranscriptionFrame, 2)
        assert [f.text for f in finals] == ["first", "segundo"]
        assert [f.finalized for f in finals] == [False, True]


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
async def test_failure_never_commits_partial_and_preserves_identity(
    client, error, category, usable
):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, upstream):
        await send(service, VADUserStartedSpeakingFrame(), audio())
        stream = await next_stream(client)
        await stream.updates.put(TranscriptionPartial(text="unfinished"))
        await out.wait_for(InterimTranscriptionFrame)
        await stream.updates.put(error("must not echo recognition hints", request_id="safe-id"))
        errors = await upstream.wait_for(ErrorFrame)
        await wait(stream.closed)
        assert len(errors) == 1 and errors[0].category is category
        assert "safe-id" in errors[0].error
        assert "recognition hints" not in errors[0].error
        assert not out.transcripts
        assert service.is_usable is usable
        if not usable:
            await service._update_settings(service.Settings(language="es"))
            assert service.is_usable is (category is ErrorCategory.INVALID_REQUEST)


@pytest.mark.asyncio
async def test_failed_first_unblocks_second(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, upstream):
        for _ in range(2):
            await send(
                service, VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame()
            )
        first, second = await next_stream(client), await next_stream(client)
        await wait(second.exhausted)
        await second.finish("second")
        await first.updates.put(None)
        await upstream.wait_for(ErrorFrame)
        assert [f.text for f in await out.wait_for(TranscriptionFrame)] == ["second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
@pytest.mark.parametrize("channels", [1, 2])
async def test_usage_uses_real_audio_format(client, rate, channels):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, _):
        await send(
            service,
            VADUserStartedSpeakingFrame(),
            audio(bytes(rate * channels * 2 // 10), rate, channels),
            VADUserStoppedSpeakingFrame(stop_secs=0.1),
        )
        stream = await next_stream(client)
        await wait(stream.exhausted)
        await stream.finish("done")
        await wait(stream.closed)
        await out.wait_for(TranscriptionFrame)
        # Metrics follow the final and are emitted once by the reader's finally block.
        task = next(iter(service._tasks), None)
        if task:
            await task
        metrics = [d for f in out.frames if isinstance(f, MetricsFrame) for d in f.data]
        usage = [d for d in metrics if isinstance(d, STTUsageMetricsData)]
        latency = [d for d in metrics if isinstance(d, TTFBMetricsData)]
        assert sum(d.value.audio_seconds for d in usage) == pytest.approx(0.1)
        assert len(latency) == 1
        assert stream.options["input_format"].sample_rate == rate
        assert stream.options["input_format"].channels == channels


@pytest.mark.asyncio
async def test_queue_overflow_is_bounded_and_recovers(client):
    client.consume = False
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, upstream):
        await send(service, VADUserStartedSpeakingFrame(), audio(bytes(160000)))
        stream = await next_stream(client)
        await send(service, audio(), audio(), VADUserStoppedSpeakingFrame())
        assert len(await upstream.wait_for(ErrorFrame)) == 1
        await wait(stream.closed)
        assert not out.transcripts
        client.consume = True
        await send(service, VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame())
        second = await next_stream(client)
        await wait(second.exhausted)
        await second.finish("recovered")
        await out.wait_for(TranscriptionFrame)


@pytest.mark.asyncio
async def test_pending_results_are_bounded(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, upstream):
        for _ in range(5):
            await send(
                service, VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame()
            )
        assert len(await upstream.wait_for(ErrorFrame)) == 1
        assert len(service._recognition.pending) == 4
        await send(service, CancelFrame())
        assert not out.transcripts
        assert not service._tasks


@pytest.mark.asyncio
async def test_interruption_preserves_recognition_and_mute_restarts_fresh(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, _):
        await send(service, VADUserStartedSpeakingFrame(), audio())
        first = await next_stream(client)
        await send(service, InterruptionFrame(), audio())
        assert not first.closed.is_set()
        await send(
            service,
            STTMuteFrame(mute=True),
            audio(),
            STTMuteFrame(mute=False),
            audio(),
            VADUserStoppedSpeakingFrame(),
        )
        second = await next_stream(client)
        await wait(second.exhausted)
        await second.finish("fresh")
        assert [f.text for f in await out.wait_for(TranscriptionFrame)] == ["fresh"]
        assert first.closed.is_set()


@pytest.mark.asyncio
async def test_request_and_shutdown_deadlines(client):
    service = RimeSTTService(api_key="test", request_timeout=0.02, shutdown_timeout=0.02)
    async with harness(service) as (out, upstream):
        await send(service, VADUserStartedSpeakingFrame(), audio())
        stream = await next_stream(client)
        await upstream.wait_for(ErrorFrame)
        await wait(stream.closed)
        await send(service, EndFrame())
        assert not out.transcripts and client.closed


@pytest.mark.asyncio
async def test_cancel_is_prompt_and_cleanup_is_repeatable(client):
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, upstream):
        await send(service, VADUserStartedSpeakingFrame(), audio())
        stream = await next_stream(client)
        start = time.monotonic()
        await send(service, CancelFrame())
        assert time.monotonic() - start < 0.5
        await stream.updates.put(RimeCancelledError("closed"))
        await service.cleanup()
        assert client.closed and stream.closed.is_set()
        assert not out.transcripts
        assert not any(isinstance(f, ErrorFrame) for f in upstream.frames)


@pytest.mark.asyncio
@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("second_text", ["second", ""])
async def test_real_aggregator_keeps_delayed_segments_in_one_turn(client, segmented, second_text):
    from rimelabs_sdk import Transcription

    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMUserAggregator,
        LLMUserAggregatorParams,
    )
    from pipecat.turns.user_start import VADUserTurnStartStrategy
    from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    context = LLMContext()
    stop = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.01)
    aggregator = LLMUserAggregator(
        context,
        params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[stop],
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

    cls = rime.RimeSegmentedSTTService if segmented else RimeSTTService
    service = cls(api_key="test", ttfs_p99_latency=60)
    async with harness(service, downstream=aggregator) as (out, _):
        await send(service, VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame())
        if segmented:
            async with asyncio.timeout(2):
                first_call = await client.new_call.get()
        else:
            first = await next_stream(client)
        await send(service, VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame())
        # Both VAD stops must reach the real turn controller before A finishes.
        await asyncio.sleep(0.03)
        if segmented:
            first_call[2].set_result(Transcription(text="first", language="en"))
            async with asyncio.timeout(2):
                second_call = await client.new_call.get()
        else:
            second = await next_stream(client)
            await first.finish("first")
        await out.wait_for(TranscriptionFrame)
        assert not completed.is_set()
        assert not out.transcripts[0].finalized
        if segmented:
            second_call[2].set_result(Transcription(text=second_text, language="en"))
        else:
            await wait(second.exhausted)
            await second.finish(second_text)
        await wait(completed)
        assert turns == ["first second" if second_text else "first"]
        assert out.transcripts[-1].finalized


@pytest.mark.asyncio
@pytest.mark.parametrize("segmented", [False, True])
async def test_empty_success_does_not_start_an_agent_reply(client, segmented):
    from rimelabs_sdk import Transcription

    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator

    context = LLMContext()
    aggregator = LLMUserAggregator(context, enable_direct_mode=True)
    cls = rime.RimeSegmentedSTTService if segmented else RimeSTTService
    service = cls(api_key="test")
    async with harness(service, downstream=aggregator) as (out, _):
        await send(service, VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame())
        if segmented:
            async with asyncio.timeout(2):
                call = await client.new_call.get()
            call[2].set_result(Transcription(text="", language="en"))
        else:
            stream = await next_stream(client)
            await wait(stream.exhausted)
            await stream.finish("")
        await out.wait_for(TranscriptionFrame)
        assert context.messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize("segmented", [False, True])
@pytest.mark.parametrize("bad_frame", [audio(rate=44100), audio(channels=3), audio(rate=24000)])
async def test_invalid_or_changing_format_is_request_scoped(client, segmented, bad_frame):
    cls = rime.RimeSegmentedSTTService if segmented else RimeSTTService
    service = cls(api_key="test")
    async with harness(service) as (out, upstream):
        await send(
            service,
            VADUserStartedSpeakingFrame(),
            audio(),
            bad_frame,
            VADUserStoppedSpeakingFrame(),
        )
        errors = await upstream.wait_for(ErrorFrame)
        assert len(errors) == 1 and errors[0].category is ErrorCategory.APPLICATION
        assert service.is_usable
        assert not out.transcripts


@pytest.mark.asyncio
async def test_full_audio_source_can_close_without_a_sentinel_slot(client):
    client.consume = False
    service = RimeSTTService(api_key="test")
    async with harness(service) as (out, _):
        await send(
            service,
            VADUserStartedSpeakingFrame(),
            audio(bytes(160000)),
            VADUserStoppedSpeakingFrame(),
        )
        stream = await next_stream(client)
        stream.allow_upload.set()
        await wait(stream.exhausted)
        await stream.finish("done")
        await out.wait_for(TranscriptionFrame)


@pytest.mark.asyncio
async def test_pending_partial_snapshots_coalesce_after_all_deltas(client):
    service = RimeSTTService(
        api_key="test", settings=RimeSTTService.Settings(output_contract="append_fragments")
    )
    async with harness(service) as (out, _):
        for _ in range(2):
            await send(
                service, VADUserStartedSpeakingFrame(), audio(), VADUserStoppedSpeakingFrame()
            )
        first, second = await next_stream(client), await next_stream(client)
        for text in ["one", " two", "\nthree"]:
            await second.updates.put(TranscriptionDelta(text=text))
        await wait(second.exhausted)
        await second.finish("one two\nthree")
        await wait(second.closed)
        await first.finish("first")
        await out.wait_for(TranscriptionFrame, 2)
        assert [f.text for f in out.frames if isinstance(f, InterimTranscriptionFrame)] == [
            "one two\nthree"
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("placement", ["upstream", "aggregator"])
async def test_vad_placement_opens_and_closes_sdk_source(client, placement):
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
