#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the segmented STT segment-format contract."""

import io
import unittest
import wave

from pipecat.frames.frames import VADUserStoppedSpeakingFrame
from pipecat.services.stt_service import SegmentedSTTService


class _CaptureSegmentSTT(SegmentedSTTService):
    def __init__(self, *, segment_audio_format: str = "wav"):
        super().__init__(sample_rate=16000)
        self._segment_audio_format_name = segment_audio_format
        self.captured_audio: bytes | None = None

    def _segment_audio_format(self) -> str:
        return self._segment_audio_format_name

    async def run_stt(self, audio: bytes):
        self.captured_audio = audio
        if False:
            yield None


class TestSegmentedSTTService(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.pcm = b"\x01\x02" * 8

    async def _run_segment(self, service: _CaptureSegmentSTT):
        service._sample_rate = 16000
        service._audio_buffer = bytearray(self.pcm)
        service._user_speaking = True

        async def _consume(generator):
            async for _ in generator:
                pass

        service.process_generator = _consume
        await service._handle_user_stopped_speaking(VADUserStoppedSpeakingFrame())

    async def test_pcm_mode_passes_raw_buffer(self):
        service = _CaptureSegmentSTT(segment_audio_format="pcm")
        await self._run_segment(service)
        self.assertEqual(service.captured_audio, self.pcm)

    async def test_wav_mode_wraps_buffer(self):
        service = _CaptureSegmentSTT(segment_audio_format="wav")
        await self._run_segment(service)

        self.assertIsNotNone(service.captured_audio)
        with wave.open(io.BytesIO(service.captured_audio), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.readframes(wav_file.getnframes()), self.pcm)

    async def test_buffer_cleared_after_segment(self):
        service = _CaptureSegmentSTT(segment_audio_format="pcm")
        await self._run_segment(service)
        self.assertEqual(service._audio_buffer, bytearray())


if __name__ == "__main__":
    unittest.main()
