#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that STT and TTS services stop working once the provider rejects them."""

import unittest
from collections.abc import AsyncGenerator

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TextFrame,
    TTSAudioRawFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.status import ServiceStatus
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import run_test


class CountingSTTService(STTService):
    """STT service that counts how often it's asked to transcribe."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transcribe_calls = 0

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        self.transcribe_calls += 1
        yield None


class CountingSegmentedSTTService(SegmentedSTTService):
    """Segmented STT service that counts how often it's asked to transcribe."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.transcribe_calls = 0

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        self.transcribe_calls += 1
        yield None


class CountingTTSService(TTSService):
    """TTS service that counts how often it's asked to synthesize."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.synthesize_calls = 0

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        self.synthesize_calls += 1
        yield TTSAudioRawFrame(
            audio=b"\x00" * 320, sample_rate=self.sample_rate, num_channels=1, context_id=context_id
        )


def audio_frame() -> InputAudioRawFrame:
    return InputAudioRawFrame(audio=b"\x00" * 320, sample_rate=16000, num_channels=1)


class TestSTTStatusGating(unittest.IsolatedAsyncioTestCase):
    async def test_audio_is_transcribed_while_the_service_is_healthy(self):
        service = CountingSTTService()

        await run_test(
            service,
            frames_to_send=[audio_frame(), audio_frame()],
            expected_down_frames=None,
        )

        self.assertEqual(service.transcribe_calls, 2)

    async def test_audio_is_dropped_once_the_service_is_misconfigured(self):
        service = CountingSTTService()
        await service._set_status(ServiceStatus.MISCONFIGURED)

        await run_test(
            service,
            frames_to_send=[audio_frame(), audio_frame()],
            expected_down_frames=None,
        )

        self.assertEqual(service.transcribe_calls, 0)

    async def test_a_degraded_service_keeps_transcribing(self):
        service = CountingSTTService()
        await service._set_status(ServiceStatus.DEGRADED)

        await run_test(
            service,
            frames_to_send=[audio_frame()],
            expected_down_frames=None,
        )

        self.assertEqual(service.transcribe_calls, 1)


class TestSegmentedSTTStatusGating(unittest.IsolatedAsyncioTestCase):
    async def test_segment_is_dropped_once_the_service_is_misconfigured(self):
        service = CountingSegmentedSTTService()
        await service._set_status(ServiceStatus.MISCONFIGURED)

        await run_test(
            service,
            frames_to_send=[audio_frame(), VADUserStoppedSpeakingFrame()],
            expected_down_frames=None,
        )

        self.assertEqual(service.transcribe_calls, 0)
        # The buffered audio is released rather than growing for the rest of
        # the session.
        self.assertEqual(len(service._audio_buffer), 0)


class TestTTSStatusGating(unittest.IsolatedAsyncioTestCase):
    async def test_text_is_synthesized_while_the_service_is_healthy(self):
        service = CountingTTSService()

        await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=None,
        )

        self.assertEqual(service.synthesize_calls, 1)

    async def test_text_is_dropped_once_the_service_is_misconfigured(self):
        service = CountingTTSService()
        await service._set_status(ServiceStatus.MISCONFIGURED)

        await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=None,
        )

        self.assertEqual(service.synthesize_calls, 0)

    async def test_dropped_text_is_reported(self):
        service = CountingTTSService()
        await service._set_status(ServiceStatus.MISCONFIGURED)

        messages = []
        handler_id = logger.add(messages.append, level="WARNING", format="{message}")
        try:
            await run_test(
                service,
                frames_to_send=[TextFrame("this should be spoken")],
                expected_down_frames=None,
            )
        finally:
            logger.remove(handler_id)

        warning = next((m for m in messages if "this should be spoken" in m), None)
        self.assertIsNotNone(warning, f"dropped text was not reported: {messages}")
        self.assertIn("misconfigured", warning)
