#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mistral text-to-speech service implementation.

This module provides integration with Mistral's Voxtral TTS API for
generating speech from text input using HTTP streaming with Server-Sent Events.
"""

import base64
import struct
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from mistralai.client import Mistral
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Mistral TTS, you need to `pip install pipecat-ai[mistral]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class MistralTTSSettings(TTSSettings):
    """Settings for MistralTTSService.

    Parameters:
        model: TTS model identifier.
        voice: Voice identifier.
        language: Language for speech synthesis.
    """

    pass


class MistralTTSService(TTSService):
    """Mistral Text-to-Speech service using the Voxtral TTS API.

    This service uses Mistral's streaming TTS API to generate PCM-encoded audio
    at 24kHz. The API returns base64-encoded float32 PCM chunks via Server-Sent
    Events, which are converted to int16 for the Pipecat pipeline.
    """

    Settings = MistralTTSSettings
    _settings: Settings

    MISTRAL_SAMPLE_RATE = 24000

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        sample_rate: Optional[int] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize Mistral TTS service.

        Args:
            api_key: Mistral API key for authentication.
            sample_rate: Output audio sample rate in Hz. Audio is resampled from
                Mistral's native 24kHz when a different rate is requested.
            settings: Runtime-updatable settings.
            **kwargs: Additional keyword arguments passed to TTSService.
        """
        # Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="voxtral-mini-tts-2603",
            voice=None,
            language=None,
        )

        # Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._client = Mistral(api_key=api_key)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Mistral TTS service supports metrics generation.
        """
        return True

    @staticmethod
    def _float32_to_int16(data: bytes) -> bytes:
        """Convert float32 PCM audio data to int16 PCM.

        Args:
            data: Raw bytes containing float32 LE PCM samples.

        Returns:
            Raw bytes containing int16 LE PCM samples.
        """
        n = len(data) // 4
        floats = struct.unpack(f"<{n}f", data)
        return struct.pack(f"<{n}h", *(min(32767, max(-32768, int(f * 32767))) for f in floats))

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Mistral's TTS API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech data.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_tts_usage_metrics(text)

            async with await self._client.audio.speech.complete_async(
                input=text,
                model=self._settings.model,
                voice_id=self._settings.voice,
                response_format="pcm",
                stream=True,
            ) as event_stream:
                async for event in event_stream:
                    if event.event == "speech.audio.delta":
                        audio_bytes = base64.b64decode(event.data.audio_data)
                        audio_int16 = self._float32_to_int16(audio_bytes)
                        audio_data = await self._resampler.resample(
                            audio_int16, self.MISTRAL_SAMPLE_RATE, self.sample_rate
                        )
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(
                            audio_data, self.sample_rate, 1, context_id=context_id
                        )
                    elif event.event == "speech.audio.done":
                        if hasattr(event.data, "usage") and event.data.usage:
                            logger.debug(f"{self}: Usage info: {event.data.usage}")
        except Exception as e:
            logger.error(f"{self} error generating TTS: {e}")
            yield ErrorFrame(error=f"Error generating TTS: {e}")
