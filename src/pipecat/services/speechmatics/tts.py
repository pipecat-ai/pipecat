#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechmatics TTS service integration."""

import os
from typing import AsyncGenerator, Optional
from urllib.parse import urlencode

import aiohttp
from loguru import logger
from pydantic import BaseModel

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
    from speechmatics.rt import __version__
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Speechmatics, you need to `pip install pipecat-ai[speechmatics]`."
    )
    raise Exception(f"Missing module: {e}")


class SpeechmaticsTTSService(TTSService):
    """Speechmatics TTS service implementation.

    This service provides text-to-speech synthesis using the Speechmatics HTTP API.
    It converts text to speech and returns raw PCM audio data for real-time playback.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Speechmatics TTS service.

        Parameters:
            voice: Voice model to use for synthesis. Defaults to "sarah".
        """

        voice: str = "sarah"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        aiohttp_session: aiohttp.ClientSession | None = None,
        sample_rate: Optional[int] = 16000,
        params: InputParams | None = None,
        **kwargs,
    ):
        """Initialize the Speechmatics TTS service.

        Args:
            api_key: Speechmatics API key for authentication. Uses environment variable
                `SPEECHMATICS_API_KEY` if not provided.
            base_url: Base URL for Speechmatics TTS API. Defaults to
                `https://preview.tts.speechmatics.com`.
            aiohttp_session: Shared aiohttp session for HTTP requests.
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            params: Optional[InputParams]: Input parameters for the service.
            **kwargs: Additional arguments passed to TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Service parameters
        self._api_key: str = api_key or os.getenv("SPEECHMATICS_API_KEY")
        self._base_url: str = base_url or "https://preview.tts.speechmatics.com"
        self._session = aiohttp_session or aiohttp.ClientSession()

        # Check we have required attributes
        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")
        if not self._base_url:
            raise ValueError("Missing Speechmatics base URL")

        # Default parameters
        self._params = params or SpeechmaticsTTSService.InputParams()

        # Set voice from parameters
        self.set_voice(self._params.voice)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Speechmatics service supports metrics generation.
        """
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Speechmatics' HTTP API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
        }

        url = _get_endpoint_url(self._base_url, self._voice_id, self.sample_rate)

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_message = f"Speechmatics TTS error: HTTP {response.status}"
                    logger.error(error_message)
                    yield ErrorFrame(error=error_message)
                    return

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()

                # Process the response in streaming chunks
                first_chunk = True
                buffer = b""

                # Helper to move all complete 2-byte int16 samples from buffer into a frame
                def _emit_complete_samples():
                    nonlocal buffer
                    if len(buffer) < 2:
                        return None
                    complete_samples = len(buffer) // 2
                    complete_bytes = complete_samples * 2

                    audio_data = buffer[:complete_bytes]
                    buffer = buffer[complete_bytes:]  # Keep remaining bytes for next iteration

                    return TTSAudioRawFrame(
                        audio=audio_data,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )

                async for chunk in response.content.iter_any():
                    if not chunk:
                        continue
                    if first_chunk:
                        await self.stop_ttfb_metrics()
                        first_chunk = False

                    buffer += chunk

                    # Emit a frame for all complete samples currently in buffer
                    frame = _emit_complete_samples()
                    if frame:
                        yield frame

                # Process any remaining bytes in buffer after streaming ends
                frame = _emit_complete_samples()
                if frame:
                    yield frame

        except Exception as e:
            logger.exception(f"Error generating TTS: {e}")
            yield ErrorFrame(error=f"Speechmatics TTS error: {str(e)}")
        finally:
            yield TTSStoppedFrame()


def _get_endpoint_url(base_url: str, voice: str, sample_rate: int) -> str:
    """Format the TTS endpoint URL with voice, output format, and version params.

    Args:
        base_url: The base URL for the TTS endpoint.
        voice: The voice model to use.
        sample_rate: The audio sample rate.

    Returns:
        str: The formatted TTS endpoint URL.
    """
    query_params = {}
    query_params["output_format"] = f"pcm_{sample_rate}"
    query_params["sm-app"] = f"pipecat/{__version__}"
    query = urlencode(query_params)

    return f"{base_url}/generate/{voice}?{query}"
