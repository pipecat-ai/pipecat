#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the eval bot-audio transcriber."""

import unittest

from pipecat.evals.transcribe import EvalTranscriber
from pipecat.frames.frames import TranscriptionFrame
from pipecat.utils.time import time_now_iso8601


class _IdentityResampler:
    async def resample(self, audio: bytes, source_sample_rate: int, target_sample_rate: int):
        return audio


class _CaptureEvalSTT:
    def __init__(self):
        self.last_audio: bytes | None = None

    async def run_stt(self, audio: bytes):
        self.last_audio = audio
        yield TranscriptionFrame("hello", "", time_now_iso8601())


class TestEvalTranscriber(unittest.IsolatedAsyncioTestCase):
    async def test_transcribe_passes_raw_pcm_to_service(self):
        service = _CaptureEvalSTT()
        transcriber = EvalTranscriber(service, padding_secs=0)
        transcriber._resampler = _IdentityResampler()

        pcm = b"\x01\x02" * 16
        text = await transcriber.transcribe(pcm, 16000)

        self.assertEqual(text, "hello")
        self.assertEqual(service.last_audio, pcm)


if __name__ == "__main__":
    unittest.main()
