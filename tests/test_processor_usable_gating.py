#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that STT and TTS services stop working once the provider rejects them."""

import unittest
from collections.abc import AsyncGenerator

import httpx
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    TextFrame,
    TTSAudioRawFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import run_test
from pipecat.utils.errors import ErrorCategory


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

    async def test_audio_is_dropped_once_the_service_is_unusable(self):
        service = CountingSTTService()
        await service.set_usable(False)

        await run_test(
            service,
            frames_to_send=[audio_frame(), audio_frame()],
            expected_down_frames=None,
        )

        self.assertEqual(service.transcribe_calls, 0)

    async def test_a_service_that_can_still_work_keeps_transcribing(self):
        service = CountingSTTService()

        await run_test(
            service,
            frames_to_send=[audio_frame()],
            expected_down_frames=None,
        )

        self.assertEqual(service.transcribe_calls, 1)


class TestSegmentedSTTStatusGating(unittest.IsolatedAsyncioTestCase):
    async def test_segment_is_dropped_once_the_service_is_unusable(self):
        service = CountingSegmentedSTTService()
        await service.set_usable(False)

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

    async def test_text_is_dropped_once_the_service_is_unusable(self):
        service = CountingTTSService()
        await service.set_usable(False)

        await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=None,
        )

        self.assertEqual(service.synthesize_calls, 0)

    async def test_dropped_text_is_reported(self):
        service = CountingTTSService()
        await service.set_usable(False)

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
        self.assertIn("no longer usable", warning)


class TestApplicationErrorClassification(unittest.IsolatedAsyncioTestCase):
    """Application code a service invokes must not affect the service's health."""

    async def test_failing_text_transformer_leaves_the_service_usable(self):
        service = CountingTTSService()

        async def failing_transform(text: str, aggregation_type) -> str:
            # A transformer calling some API that rejects its own credentials.
            request = httpx.Request("POST", "https://translate.example.com/v1")
            raise httpx.HTTPStatusError(
                "Unauthorized", request=request, response=httpx.Response(401, request=request)
            )

        service.add_text_transformer(failing_transform)

        _, up = await run_test(
            service,
            frames_to_send=[TextFrame("hello")],
            expected_down_frames=None,
            expected_up_frames=[ErrorFrame],
        )

        self.assertEqual(up[0].category, ErrorCategory.APPLICATION)
        self.assertTrue(service.is_usable)
        # The turn produces no audio: a transformer may exist to remove
        # something, so speaking the untransformed text isn't safe.
        self.assertEqual(service.synthesize_calls, 0)
