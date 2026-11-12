#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from types import SimpleNamespace

import pytest
from openai.types.audio import Transcription, TranscriptionDiarized

from pipecat.frames.frames import (
    InputAudioRawFrame,
    MetricsFrame,
    TranscriptionFrame,
    TranscriptionSpeaker,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.openai.stt import OpenAIRealtimeSTTService, OpenAISTTService
from pipecat.tests.utils import run_test
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

SAMPLE_RATE = 16000


@pytest.mark.asyncio
async def test_segment_emits_usage_and_transcription(monkeypatch):
    # No trailing padding, so usage equals the buffered audio.
    service = OpenAISTTService(api_key="test-key", trailing_silence_secs=0)

    async def fake_transcribe(audio: bytes) -> Transcription:
        return Transcription(text="hello world")

    monkeypatch.setattr(service, "_transcribe", fake_transcribe)

    pcm = b"\x01\x02" * SAMPLE_RATE  # 1s of 16-bit mono audio
    received_down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=pcm, sample_rate=SAMPLE_RATE, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ],
        pipeline_params=PipelineParams(enable_usage_metrics=True),
    )

    transcripts = [f for f in received_down if isinstance(f, TranscriptionFrame)]
    assert len(transcripts) == 1
    assert transcripts[0].text == "hello world"
    assert transcripts[0].finalized is True

    usage_indexes = [
        i
        for i, f in enumerate(received_down)
        if isinstance(f, MetricsFrame) and any(isinstance(d, STTUsageMetricsData) for d in f.data)
    ]
    assert len(usage_indexes) == 1
    usage_frame = received_down[usage_indexes[0]]
    usage = next(d for d in usage_frame.data if isinstance(d, STTUsageMetricsData))
    assert usage.value.audio_seconds == pytest.approx(len(pcm) / (SAMPLE_RATE * 2))

    # Usage precedes the transcript so tracing attaches it to the span the
    # finalized TranscriptionFrame closes.
    assert usage_indexes[0] < received_down.index(transcripts[0])


def test_openai_realtime_should_interrupt_rides_on_recommended_strategies():
    # should_interrupt configures the strategies the service recommends via its
    # metadata frame; the service never broadcasts the interruption itself.
    for should_interrupt in (True, False):
        service = OpenAIRealtimeSTTService(
            api_key="test-key",
            turn_detection={"type": "server_vad"},
            should_interrupt=should_interrupt,
        )
        strategies = service.service_metadata_frame().user_turn_strategies
        assert isinstance(strategies, ExternalUserTurnStrategies)
        assert strategies.enable_interruptions is should_interrupt


def test_openai_realtime_server_defaults_recommend_strategies():
    """``turn_detection=None`` omits the field, so the session's own default stands.

    That default detects turns, so the recommendation applies just as it does
    for an explicit configuration.
    """
    service = OpenAIRealtimeSTTService(api_key="test-key", turn_detection=None)
    strategies = service.service_metadata_frame().user_turn_strategies
    assert isinstance(strategies, ExternalUserTurnStrategies)


def test_openai_realtime_local_vad_mode_recommends_no_strategies():
    """With turn detection off the server reports no boundaries to propose."""
    service = OpenAIRealtimeSTTService(api_key="test-key")
    assert service.service_metadata_frame().user_turn_strategies is None


@pytest.mark.parametrize(
    "model, expected",
    [
        ("gpt-transcribe", {"response_format": "json", "include": ["logprobs"]}),
        ("whisper-1", {"response_format": "verbose_json"}),
        ("gpt-4o-transcribe-diarize", {"response_format": "diarized_json"}),
    ],
)
@pytest.mark.asyncio
async def test_prob_metrics_request_shape(model, expected):
    """Probability metrics are requested in the form each model family accepts."""
    service = OpenAISTTService(
        api_key="test-key",
        settings=OpenAISTTService.Settings(model=model),
        include_prob_metrics=True,
    )

    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return Transcription(text="hello world")

    service._client.audio.transcriptions.create = fake_create

    await service._transcribe(b"\x01\x02")

    assert captured["model"] == model
    assert {k: captured[k] for k in ("response_format", "include") if k in captured} == expected


@pytest.mark.asyncio
async def test_diarization_model_requests_speaker_segments():
    service = OpenAISTTService(
        api_key="test-key", settings=OpenAISTTService.Settings(model="gpt-4o-transcribe-diarize")
    )
    captured = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return Transcription(text="hello")

    service._client.audio.transcriptions.create = fake_create

    await service._transcribe(b"\x01\x02")

    assert captured["response_format"] == "diarized_json"
    assert captured["chunking_strategy"] == "auto"


def _diarized(text: str, segments: list[tuple[str, str]]) -> TranscriptionDiarized:
    return TranscriptionDiarized.model_validate(
        {
            "duration": 1.0,
            "task": "transcribe",
            "text": text,
            "segments": [
                {
                    "id": str(index),
                    "type": "transcript.text.segment",
                    "start": 0.0,
                    "end": 0.5,
                    "text": segment_text,
                    "speaker": speaker,
                }
                for index, (segment_text, speaker) in enumerate(segments)
            ],
        }
    )


@pytest.mark.asyncio
async def test_diarized_response_emits_one_frame_per_speaker_run():
    service = OpenAISTTService(
        api_key="test-key",
        settings=OpenAISTTService.Settings(
            model="gpt-4o-transcribe-diarize", speaker_format="{speaker}: {text}"
        ),
    )
    service._user_id = "session-user"
    response = _diarized(
        "Hi there. How are you? Fine.",
        [(" Hi there.", "A"), (" How are you?", "A"), (" Fine.", "B")],
    )

    async def fake_transcribe(audio):
        return response

    service._transcribe = fake_transcribe

    frames = [f async for f in service.run_stt(b"audio") if isinstance(f, TranscriptionFrame)]

    assert [(f.speaker, f.text) for f in frames] == [
        (TranscriptionSpeaker("A", 1), "A: Hi there. How are you?"),
        (TranscriptionSpeaker("B", 2), "B: Fine."),
    ]
    assert all(f.user_id == "session-user" for f in frames)


@pytest.mark.asyncio
async def test_segments_without_speakers_are_not_split():
    """Whisper's verbose segments carry no speaker, so the result is one unlabelled frame."""
    service = OpenAISTTService(
        api_key="test-key", settings=OpenAISTTService.Settings(model="whisper-1")
    )
    response = SimpleNamespace(
        text="Hi there. Hello.",
        segments=[SimpleNamespace(text="Hi there."), SimpleNamespace(text="Hello.")],
    )

    async def fake_transcribe(audio):
        return response

    service._transcribe = fake_transcribe

    frames = [f async for f in service.run_stt(b"audio") if isinstance(f, TranscriptionFrame)]

    assert [(f.speaker, f.text) for f in frames] == [(None, "Hi there. Hello.")]
