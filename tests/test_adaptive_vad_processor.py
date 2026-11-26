#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    InputAudioRawFrame,
    SpeechControlParamsFrame,
    VADParamsUpdateFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.audio.adaptive_vad_processor import AdaptiveVADProcessor
from pipecat.tests.utils import run_test

_SAMPLE_RATE = 16000
# 400ms of audio fills the volume tracker window so volume becomes measurable.
_CHUNK_BYTES = int(_SAMPLE_RATE * 0.4) * 2


def _loud_frame() -> InputAudioRawFrame:
    # Full-scale square wave -> high measured loudness.
    audio = b"\xff\x7f" * (_CHUNK_BYTES // 2)
    return InputAudioRawFrame(audio=audio, sample_rate=_SAMPLE_RATE, num_channels=1)


def _silent_frame() -> InputAudioRawFrame:
    return InputAudioRawFrame(
        audio=b"\x00" * _CHUNK_BYTES, sample_rate=_SAMPLE_RATE, num_channels=1
    )


class TestAdaptiveVADProcessor(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_frames(self):
        """Frames pass through unchanged."""
        processor = AdaptiveVADProcessor()
        await run_test(
            processor,
            frames_to_send=[_silent_frame()],
            expected_down_frames=[InputAudioRawFrame],
        )

    async def test_emits_update_when_noise_floor_rises(self):
        """A rising noise floor emits a VADParamsUpdateFrame upstream."""
        processor = AdaptiveVADProcessor(
            update_period_secs=0.0, margin=0.1, smoothing_factor=1.0, min_threshold=0.0
        )
        params = VADParams(min_volume=0.1)
        frames = [SpeechControlParamsFrame(vad_params=params)]
        frames += [_loud_frame() for _ in range(3)]
        _, up = await run_test(
            processor,
            frames_to_send=frames,
            expected_up_frames=[VADParamsUpdateFrame],
        )
        update = next(f for f in up if isinstance(f, VADParamsUpdateFrame))
        self.assertGreater(update.params.min_volume, params.min_volume)

    async def test_no_update_while_speaking(self):
        """No parameter update is emitted while the user is speaking."""
        processor = AdaptiveVADProcessor(
            update_period_secs=0.0, smoothing_factor=1.0, min_threshold=0.0
        )
        frames = [
            SpeechControlParamsFrame(vad_params=VADParams(min_volume=0.1)),
            VADUserStartedSpeakingFrame(),
            _loud_frame(),
            _loud_frame(),
        ]
        _, up = await run_test(
            processor,
            frames_to_send=frames,
            expected_up_frames=[],
        )
        self.assertFalse(any(isinstance(f, VADParamsUpdateFrame) for f in up))

    async def test_no_update_without_known_params(self):
        """Without a SpeechControlParamsFrame first, nothing is emitted."""
        processor = AdaptiveVADProcessor(update_period_secs=0.0, smoothing_factor=1.0)
        _, up = await run_test(
            processor,
            frames_to_send=[_loud_frame(), _loud_frame()],
            expected_up_frames=[],
        )
        self.assertFalse(any(isinstance(f, VADParamsUpdateFrame) for f in up))
