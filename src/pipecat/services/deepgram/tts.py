#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram text-to-speech service implementation.

This module provides integration with Deepgram's text-to-speech API
for generating speech from text using various voice models.
"""

import asyncio
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from deepgram import DeepgramClient, DeepgramClientOptions
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Deepgram, you need to `pip install pipecat-ai[deepgram]`.")
    raise Exception(f"Missing module: {e}")


class DeepgramTTSService(TTSService):
    """Deepgram text-to-speech service.

    Provides text-to-speech synthesis using Deepgram's streaming API.
    Supports various voice models and audio encoding formats with
    configurable sample rates and quality settings.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "aura-2-helena-en",
        base_url: str = "",
        sample_rate: Optional[int] = None,
        encoding: str = "linear16",
        **kwargs,
    ):
        """Initialize the Deepgram TTS service.

        Args:
            api_key: Deepgram API key for authentication.
            voice: Voice model to use for synthesis. Defaults to "aura-2-helena-en".
            base_url: Custom base URL for Deepgram API. Uses default if empty.
            sample_rate: Audio sample rate in Hz. If None, uses service default.
            encoding: Audio encoding format. Defaults to "linear16".
            **kwargs: Additional arguments passed to parent TTSService class.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._settings = {
            "encoding": encoding,
        }
        self.set_voice(voice)

        # Keep the base URL for potential custom endpoints
        self._base_url = base_url if base_url else "https://api.beta.deepgram.com"
        
        client_options = DeepgramClientOptions(url=base_url)
        self._deepgram_client = DeepgramClient(api_key, config=client_options)

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, as Deepgram TTS service supports metrics generation.
        """
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Deepgram's TTS API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech, plus start/stop frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # Build URL with parameters
        url = f"{self._base_url}/v1/speak"
        
        headers = {
            "Authorization": f"Token {self._api_key}",
            "Content-Type": "application/json"
        }

        params = {
            "model": self._voice_id,
            "encoding": self._settings["encoding"],
            "sample_rate": self.sample_rate,
            "container": "none"
        }
        
        payload = {
            "text": text,
        }

        try:
            await self.start_ttfb_metrics()

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_text}")

                    await self.start_tts_usage_metrics(text)
                    yield TTSStartedFrame()

                    first_chunk = True
                    async for chunk in response.content.iter_chunked(1024):
                        if first_chunk:
                            await self.stop_ttfb_metrics()
                            first_chunk = False
                        
                        if chunk:
                            yield TTSAudioRawFrame(audio=chunk, sample_rate=self.sample_rate, num_channels=1)

            yield TTSStoppedFrame()

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
            yield ErrorFrame(f"Error getting audio: {str(e)}")
