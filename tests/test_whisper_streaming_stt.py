#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio

import pytest

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.whisper.stt import WhisperStreamingSTTService
from pipecat.tests.utils import SleepFrame, run_test

SAMPLE_RATE = 16000
# 16-bit mono PCM: 2 bytes per sample.
ONE_SECOND_PCM = b"\x01\x02" * SAMPLE_RATE


def _make_service() -> WhisperStreamingSTTService:
    """Build a WhisperStreamingSTTService with stubbed models.

    Defined as a factory (not a module-level class) so this concrete subclass
    isn't picked up by the service-discovery scan in test_service_init.py.
    The interim stub returns text that changes as the buffer grows, so dedup
    behavior can be asserted; the final stub returns a fixed transcript.
    """

    class _FakeStreamingSTTService(WhisperStreamingSTTService):
        def _load(self):
            self._model = object()
            self._interim_model = object()

        async def _transcribe(self, model, audio: bytes) -> str:
            await asyncio.sleep(0.01)
            if model is self._model:
                return "final transcript"
            return f"partial after {len(audio)} bytes"

    return _FakeStreamingSTTService(sample_rate=SAMPLE_RATE)


@pytest.mark.asyncio
async def test_interims_while_speaking_then_finalized_transcription():
    service = _make_service()

    down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=ONE_SECOND_PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            SleepFrame(sleep=0.7),  # let the interim loop run a few passes
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
    service = _make_service()

    # A single audio frame: the buffer stops growing, so every interim pass
    # after the first returns identical text and must not be re-pushed.
    down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=ONE_SECOND_PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            SleepFrame(sleep=0.9),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    interims = [f.text for f in down if isinstance(f, InterimTranscriptionFrame)]
    assert len(interims) == len(set(interims))


@pytest.mark.asyncio
async def test_no_interims_below_minimum_audio():
    service = _make_service()

    # 0.125s of audio is below the 0.5s interim minimum: no interim passes.
    down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(
                audio=ONE_SECOND_PCM[: SAMPLE_RATE // 4],
                sample_rate=SAMPLE_RATE,
                num_channels=1,
            ),
            SleepFrame(sleep=0.6),
            VADUserStoppedSpeakingFrame(),
        ],
    )

    assert not [f for f in down if isinstance(f, InterimTranscriptionFrame)]
    # The final pass still runs on whatever was buffered.
    assert [f for f in down if isinstance(f, TranscriptionFrame)]


@pytest.mark.asyncio
async def test_no_interims_after_user_stops_speaking():
    service = _make_service()

    down, _ = await run_test(
        service,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            InputAudioRawFrame(audio=ONE_SECOND_PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            SleepFrame(sleep=0.4),
            VADUserStoppedSpeakingFrame(),
            # Audio keeps arriving after the turn (e.g. background noise); the
            # interim task must be gone and push nothing for it.
            InputAudioRawFrame(audio=ONE_SECOND_PCM, sample_rate=SAMPLE_RATE, num_channels=1),
            SleepFrame(sleep=0.6),
        ],
    )

    finals = [f for f in down if isinstance(f, TranscriptionFrame)]
    assert len(finals) == 1

    after_final = down[down.index(finals[0]) + 1 :]
    assert not [f for f in after_final if isinstance(f, InterimTranscriptionFrame)]
