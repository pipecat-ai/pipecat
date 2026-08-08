#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import io
import wave
from collections.abc import AsyncGenerator

import pytest

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.time import time_now_iso8601

SAMPLE_RATE = 16000
# Distinct, non-zero 16-bit samples so a misread WAV header would be obvious.
PCM = bytes(range(0, 240)) * 4  # 960 bytes, even length


def _make_capturing_service(wants_wav: bool | None = None) -> SegmentedSTTService:
    """Build a SegmentedSTTService that captures the bytes handed to run_stt().

    Defined as a factory (not a module-level class) so this concrete subclass
    isn't picked up by the service-discovery scan in test_service_init.py, which
    would try to construct it and fail on its (intentionally minimal) settings.

    Args:
        wants_wav: If None, inherit the base default; otherwise force the
            ``wants_wav_segments`` contract to this value.
    """

    class _CapturingSegmentedSTTService(SegmentedSTTService):
        def __init__(self, **kwargs):
            super().__init__(sample_rate=SAMPLE_RATE, **kwargs)
            self.captured: list[bytes] = []

        def can_generate_metrics(self) -> bool:
            return True

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            self.captured.append(audio)
            return
            yield  # make this an async generator

    if wants_wav is not None:
        _CapturingSegmentedSTTService.wants_wav_segments = property(lambda self: wants_wav)

    return _CapturingSegmentedSTTService()


async def _drive_one_segment(service: SegmentedSTTService):
    await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ],
    )


@pytest.mark.asyncio
async def test_default_mode_wraps_segment_in_wav():
    service = _make_capturing_service()
    assert service.wants_wav_segments is True

    await _drive_one_segment(service)

    assert len(service.captured) == 1
    audio = service.captured[0]

    # A valid WAV container with the right sample rate and the exact PCM payload.
    with wave.open(io.BytesIO(audio), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getsampwidth() == 2
        assert wav.getnchannels() == 1
        assert wav.readframes(wav.getnframes()) == PCM


@pytest.mark.asyncio
async def test_passthrough_mode_preserves_exact_pcm():
    service = _make_capturing_service(wants_wav=False)
    assert service.wants_wav_segments is False

    await _drive_one_segment(service)

    assert len(service.captured) == 1
    # Raw PCM, byte-for-byte: no WAV header prepended.
    assert service.captured[0] == PCM


@pytest.mark.asyncio
async def test_segment_emits_usage_for_raw_buffer_duration():
    from pipecat.frames.frames import MetricsFrame
    from pipecat.metrics.metrics import STTUsageMetricsData
    from pipecat.pipeline.worker import PipelineParams

    # WAV mode: usage must measure the raw PCM buffer, not the WAV container.
    service = _make_capturing_service()

    received_down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ],
        pipeline_params=PipelineParams(enable_usage_metrics=True),
    )

    usage_data = [
        d
        for f in received_down
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, STTUsageMetricsData)
    ]
    assert len(usage_data) == 1
    assert usage_data[0].value.audio_seconds == pytest.approx(len(PCM) / (SAMPLE_RATE * 2))


def _make_interim_service(
    wants_wav: bool = False,
    *,
    interim_interval: float = 0.05,
    interim_failures: int = 0,
    **kwargs,
) -> SegmentedSTTService:
    """Build a SegmentedSTTService with interim transcriptions enabled.

    Defined as a factory (not a module-level class) so this concrete subclass
    isn't picked up by the service-discovery scan in test_service_init.py.
    The interim stub returns text that changes as the buffer grows, so dedup
    behavior can be asserted; the final stub returns a fixed transcript.
    """

    class _InterimSegmentedSTTService(SegmentedSTTService):
        def __init__(self):
            super().__init__(sample_rate=SAMPLE_RATE, interim_interval=interim_interval, **kwargs)
            self.interim_audio: list[bytes] = []
            self._interim_failures_remaining = interim_failures

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            yield TranscriptionFrame("final transcript", "", time_now_iso8601(), None)

        async def run_interim_stt(self, audio: bytes) -> str | None:
            self.interim_audio.append(audio)
            if self._interim_failures_remaining:
                self._interim_failures_remaining -= 1
                raise RuntimeError("transient interim failure")
            await asyncio.sleep(0.01)
            return f"partial after {len(audio)} bytes"

    _InterimSegmentedSTTService.wants_wav_segments = property(lambda self: wants_wav)

    return _InterimSegmentedSTTService()


@pytest.mark.asyncio
async def test_interims_while_speaking_then_finalized_transcription():
    service = _make_interim_service()

    down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            SleepFrame(sleep=0.15),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    interims = [f for f in down if isinstance(f, InterimTranscriptionFrame)]
    finals = [f for f in down if isinstance(f, TranscriptionFrame)]

    assert len(interims) >= 1
    assert len(finals) == 1
    assert finals[0].text == "final transcript"
    assert finals[0].finalized

    # Interims must precede the final transcription.
    assert down.index(interims[-1]) < down.index(finals[0])


@pytest.mark.asyncio
async def test_identical_consecutive_interims_are_deduplicated():
    service = _make_interim_service()

    # A single audio frame: the buffer stops growing, so every interim pass
    # after the first returns identical text and must not be re-pushed.
    down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            SleepFrame(sleep=0.2),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    interims = [f.text for f in down if isinstance(f, InterimTranscriptionFrame)]
    assert len(service.interim_audio) >= 2
    assert interims == [f"partial after {len(PCM)} bytes"]


@pytest.mark.asyncio
async def test_no_interims_after_user_stops_speaking():
    service = _make_interim_service()

    down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            SleepFrame(sleep=0.12),
            VADUserStoppedSpeakingFrame(),
            # Audio keeps arriving after the turn (e.g. background noise); the
            # interim task must be gone and push nothing for it.
            InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            SleepFrame(sleep=0.15),
        ],
    )

    finals = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert len(finals) == 1

    assert [f for f in down if isinstance(f, InterimTranscriptionFrame)]
    after_final = down[down.index(finals[0]) + 1 :]
    assert not [f for f in after_final if isinstance(f, InterimTranscriptionFrame)]


@pytest.mark.asyncio
async def test_interim_audio_follows_wav_segment_contract():
    service = _make_interim_service(wants_wav=True)

    await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            SleepFrame(sleep=0.12),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    assert service.interim_audio
    # run_interim_stt receives segments in the same format as run_stt.
    with wave.open(io.BytesIO(service.interim_audio[0]), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.readframes(wav.getnframes()) == PCM


@pytest.mark.asyncio
async def test_interims_disabled_by_default():
    # run_stt-only subclass: run_interim_stt stays unimplemented and must
    # never be called when interim_interval isn't set.
    service = _make_capturing_service(wants_wav=False)

    down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    assert not [f for f in down if isinstance(f, InterimTranscriptionFrame)]


@pytest.mark.parametrize("interim_interval", [0.0, -0.1, float("nan"), float("inf")])
def test_interim_interval_must_be_finite_and_positive(interim_interval: float):
    with pytest.raises(ValueError, match="finite value greater than zero"):
        _make_interim_service(interim_interval=interim_interval)


@pytest.mark.asyncio
async def test_transient_interim_failure_does_not_stop_later_interims():
    service = _make_interim_service(interim_failures=1)

    down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            SleepFrame(sleep=0.15),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    assert len(service.interim_audio) >= 2
    assert [f for f in down if isinstance(f, InterimTranscriptionFrame)]


@pytest.mark.asyncio
async def test_cleanup_cancels_interim_task():
    service = _make_interim_service(task_manager=TaskManager())
    task = service.create_task(service._interim_task_handler(), "interim_stt")
    service._interim_task = task

    await service.cleanup()

    assert service._interim_task is None
    assert task.cancelled()
