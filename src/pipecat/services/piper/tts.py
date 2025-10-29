#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Piper TTS service implementation."""

from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


# This assumes a running TTS service running: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_HTTP.md
class PiperTTSService(TTSService):
    """Piper TTS service implementation.

    Provides integration with Piper's HTTP TTS server for text-to-speech
    synthesis. Supports streaming audio generation with configurable sample
    rates and automatic WAV header removal.
    """

    def __init__(
        self,
        *,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession,
        # When using Piper, the sample rate of the generated audio depends on the
        # voice model being used.
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the Piper TTS service.

        Args:
            base_url: Base URL for the Piper TTS HTTP server.
            aiohttp_session: aiohttp ClientSession for making HTTP requests.
            sample_rate: Output sample rate. If None, uses the voice model's native rate.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        if base_url.endswith("/"):
            logger.warning("Base URL ends with a slash, this is not allowed.")
            base_url = base_url[:-1]

        self._base_url = base_url
        self._session = aiohttp_session
        self._settings = {"base_url": base_url}

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Piper service supports metrics generation.
        """
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Piper's HTTP API.

        Args:
            text: The text to convert to speech.

        Yields:
            Frame: Audio frames containing the synthesized speech and status frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        headers = {
            "Content-Type": "application/json",
        }
        try:
            await self.start_ttfb_metrics()

            async with self._session.post(
                self._base_url, json={"text": text}, headers=headers
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(
                        f"{self} error getting audio (status: {response.status}, error: {error})"
                    )
                    yield ErrorFrame(
                        f"Error getting audio (status: {response.status}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()

                CHUNK_SIZE = self.chunk_size

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(CHUNK_SIZE), strip_wav_header=True
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
