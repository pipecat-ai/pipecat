#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import io
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pipecat.services.nvidia.transcribe.stt as transcribe_module
from pipecat.frames.frames import (
    ErrorFrame,
    STTMetadataFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language


class TestNvidiaTranscribeSTTService(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="  Hello world.  "))]
        )
        self.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=self.response))
            ),
            close=AsyncMock(),
        )
        self.factory = self.enterContext(
            patch.object(transcribe_module, "AsyncOpenAI", return_value=self.client)
        )
        self.service = transcribe_module.NvidiaTranscribeSTTService(base_url="http://nim.test/v1")
        self.service.start_processing_metrics = AsyncMock()
        self.service.stop_processing_metrics = AsyncMock()
        self.service._handle_transcription = AsyncMock()
        self.addAsyncCleanup(self.service.cleanup)
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\x01\x02" * 160)
        self.audio = audio.getvalue()

    async def transcribe(self):
        return [frame async for frame in self.service.run_stt(self.audio)]

    async def test_audio_only_request_and_response(self):
        frames = await self.transcribe()
        self.factory.assert_called_once_with(
            api_key="not-needed", base_url="http://nim.test/v1", max_retries=2, timeout=10.0
        )
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "nvidia-nemotron-3p5-transcribe")
        self.assertFalse(request["stream"])
        self.assertIsNone(request["extra_body"])
        self.assertEqual(
            request["messages"],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(self.audio).decode("ascii"),
                                "format": "wav",
                            },
                        }
                    ],
                }
            ],
        )
        self.assertEqual(self.service._trailing_silence_secs, 0.5)
        self.assertTrue(self.service.can_generate_metrics())
        self.assertIsInstance(frames[0], TranscriptionFrame)
        self.assertEqual(frames[0].text, "Hello world.")
        self.assertIs(frames[0].result, self.response)
        self.service._handle_transcription.assert_awaited_once_with("Hello world.", True, None)
        self.service.start_processing_metrics.assert_awaited_once()
        self.service.stop_processing_metrics.assert_awaited_once()

    def test_metadata_uses_transcribe_latency(self):
        frame = self.service.service_metadata_frame()
        self.assertIsInstance(frame, STTMetadataFrame)
        self.assertEqual(frame.ttfs_p99_latency, 2.5)

    async def test_settings_frame_changes_next_prompt_and_language(self):
        await self.transcribe()
        await self.service.process_frame(
            STTUpdateSettingsFrame(
                delta=self.service.Settings(prompt="Use Acme.", language=Language.EN_US)
            ),
            FrameDirection.DOWNSTREAM,
        )
        await self.transcribe()
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["extra_body"], {"language": "en"})
        self.assertEqual(
            request["messages"][0]["content"][1], {"type": "text", "text": "Use Acme."}
        )
        await self.service.process_frame(
            STTUpdateSettingsFrame(delta=self.service.Settings(prompt=None, language=None)),
            FrameDirection.DOWNSTREAM,
        )
        await self.transcribe()
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertIsNone(request["extra_body"])
        self.assertEqual(len(request["messages"][0]["content"]), 1)

    async def test_constructor_settings_delta(self):
        service = transcribe_module.NvidiaTranscribeSTTService(
            api_key="test-key",
            base_url="http://nim.test/v1",
            timeout=3.0,
            trailing_silence_secs=0.25,
            settings=transcribe_module.NvidiaTranscribeSTTService.Settings(
                model="custom-model", language=Language.ES_ES, prompt="Use Madrid."
            ),
        )
        self.addAsyncCleanup(service.cleanup)
        self.assertEqual(service._trailing_silence_secs, 0.25)
        self.factory.assert_called_with(
            api_key="test-key", base_url="http://nim.test/v1", max_retries=2, timeout=3.0
        )
        service._handle_transcription = AsyncMock()
        frames = [frame async for frame in service.run_stt(self.audio)]
        request = self.client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "custom-model")
        self.assertEqual(request["extra_body"], {"language": "es"})
        self.assertEqual(request["messages"][0]["content"][1]["text"], "Use Madrid.")
        self.assertEqual(frames[0].language, "es")

    async def test_empty_content_yields_nothing(self):
        for content in ("   ", "", None):
            self.response.choices[0].message.content = content
            self.assertEqual(await self.transcribe(), [])
        self.service._handle_transcription.assert_not_awaited()

    async def test_exception_yields_error_and_stops_metrics(self):
        self.client.chat.completions.create.side_effect = RuntimeError("NIM unavailable")
        frames = await self.transcribe()
        self.assertEqual(len(frames), 1)
        self.assertIsInstance(frames[0], ErrorFrame)
        self.assertIn("NIM unavailable", frames[0].error)
        self.service.stop_processing_metrics.assert_awaited_once()

    async def test_cleanup_closes_client(self):
        await self.service.cleanup()
        self.client.close.assert_awaited_once()

    def test_contextual_prompt(self):
        self.assertEqual(
            self.service.contextual_prompt(["Acme", "Thiruvananthapuram"], "travel"),
            "This audio belongs to the travel field and may contain the following words or phrases: "
            "Acme, Thiruvananthapuram. Transcribe the English audio into text, "
            "ensuring all punctuation marks are included.",
        )
