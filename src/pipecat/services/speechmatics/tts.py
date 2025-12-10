#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechmatics TTS service integration."""

import asyncio
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
from pipecat.utils.network import exponential_backoff_time
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

    SPEECHMATICS_SAMPLE_RATE = 16000

    class InputParams(BaseModel):
        """Optional input parameters for Speechmatics TTS configuration.

        Parameters:
            max_retries: Maximum number of retries for TTS requests. Defaults to 5.
        """

        max_retries: int = 5

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://preview.tts.speechmatics.com",
        voice_id: str = "sarah",
        aiohttp_session: aiohttp.ClientSession,
        sample_rate: Optional[int] = SPEECHMATICS_SAMPLE_RATE,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Speechmatics TTS service.

        Args:
            api_key: Speechmatics API key for authentication.
            base_url: Base URL for Speechmatics TTS API.
            voice_id: Voice model to use for synthesis.
            aiohttp_session: Shared aiohttp session for HTTP requests.
            sample_rate: Audio sample rate in Hz.
            params: Optional[InputParams]: Input parameters for the service.
            **kwargs: Additional arguments passed to TTSService.
        """
        if sample_rate and sample_rate != self.SPEECHMATICS_SAMPLE_RATE:
            logger.warning(
                f"Speechmatics TTS only supports {self.SPEECHMATICS_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {sample_rate}Hz may cause issues."
            )
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Service parameters
        self._api_key: str = api_key
        self._base_url: str = base_url
        self._session = aiohttp_session

        # Check we have required attributes
        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")

        # Default parameters
        self._params = params or SpeechmaticsTTSService.InputParams()

        # Set voice from constructor parameter
        self.set_voice(voice_id)

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
        # Log the TTS started frame
        logger.debug(f"{self}: Generating TTS [{text}]")

        # HTTP headers
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # HTTP payload
        payload = {
            "text": text,
        }

        # Complete HTTP URL
        url = _get_endpoint_url(self._base_url, self._voice_id, self.sample_rate)

        try:
            # Start TTS TTFB metrics
            await self.start_ttfb_metrics()

            # Track attempt
            attempt = 0

            # Keep retrying until we get a 200 response or timeout
            while True:
                async with self._session.post(url, json=payload, headers=headers) as response:
                    """Evaluate response from TTS service."""

                    # 503 : Service unavailable
                    if response.status == 503:
                        """Calculate the backoff time and retry."""

                        try:
                            # Calculate the backoff time
                            backoff_time = exponential_backoff_time(
                                attempt=attempt, min_wait=0.25, max_wait=8.0, multiplier=0.5
                            )

                            # Increment attempt
                            attempt += 1

                            # Check if we've exceeded the maximum number of attempts
                            if attempt >= self._params.max_retries:
                                raise ValueError()

                            # Report error frame
                            yield ErrorFrame(
                                error=f"Service unavailable [503] (attempt {attempt}, retry in {backoff_time:.2f}s)"
                            )

                            # Wait before retrying
                            await asyncio.sleep(backoff_time)

                            # Retry
                            continue

                        except (ValueError, ArithmeticError):
                            yield ErrorFrame(
                                error=f"Service unavailable [503] (attempts {attempt})",
                            )
                            return

                    # != 200 : Error
                    if response.status != 200:
                        yield ErrorFrame(error=f"Service unavailable [{response.status}]")
                        return

                    # Update Pipecat metrics
                    await self.start_tts_usage_metrics(text)

                    # Emit the TTS started frame
                    yield TTSStartedFrame()

                    # Process the response in streaming chunks
                    first_chunk = True
                    buffer = b""

                    # Iterate over each audio data chunk from the TTS API
                    async for chunk in response.content.iter_any():
                        if not chunk:
                            continue
                        if first_chunk:
                            await self.stop_ttfb_metrics()
                            first_chunk = False

                        buffer += chunk

                        # Emit all complete 2-byte int16 samples from buffer
                        if len(buffer) >= 2:
                            complete_samples = len(buffer) // 2
                            complete_bytes = complete_samples * 2

                            audio_data = buffer[:complete_bytes]
                            buffer = buffer[complete_bytes:]

                            # Emit the audio frame
                            yield TTSAudioRawFrame(
                                audio=audio_data,
                                sample_rate=self.sample_rate,
                                num_channels=1,
                            )

                    # Successfully processed the response, break out of retry loop
                    break

        except Exception as e:
            yield ErrorFrame(error=f"Error generating TTS: {e}")
        finally:
            # Emit the TTS stopped frame
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
