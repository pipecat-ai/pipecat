#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Uplift TTS service integration."""

import asyncio
from typing import AsyncGenerator, Optional

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


class UpliftTTSService(TTSService):
    """Uplift TTS service implementation.
       This service provides text-to-speech synthesis using the Uplift HTTP API.
    """

    class InputParams(BaseModel):
        """Optional input parameters for Uplift TTS configuration.

        Parameters:
            max_retries: Maximum number of retries for TTS requests. Defaults to 5.
        """
        max_retries: int = 5

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.upliftai.org/v1/synthesis/text-to-speech",
        voice_id: str = "v_meklc281",
        aiohttp_session: aiohttp.ClientSession,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Uplift TTS service.

        Args:
            api_key: Uplift API key for authentication.
            base_url: Base URL for Uplift TTS API.
            voice_id: Voice model to use for synthesis.
            aiohttp_session: Shared aiohttp session for HTTP requests.
            params: Optional[InputParams]: Input parameters for the service.
            **kwargs: Additional arguments passed to TTSService.
        """
        super().__init__(**kwargs)

        # Service parameters
        self._api_key: str = api_key
        self._base_url: str = base_url
        self._session = aiohttp_session

        # Check we have required attributes
        if not self._api_key:
            raise ValueError("Missing Uplift API key")

        # Default parameters
        self._params = params or UpliftTTSService.InputParams()

        # Set voice from constructor parameter
        self.set_voice(voice_id)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Uplift's simple TTS endpoint."""

        #TTS has started
        yield TTSStartedFrame()

        #Headers and payload
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }


        payload = {
            "text": text,
            "voiceId": self._voice_id,
            "outputFormat": "WAV_22050_16"
        }

        #POST request and read audio
        try:
            async with self._session.post(self._base_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    yield ErrorFrame(error=f"Uplift TTS error: {response.status}")
                    return

                audio_bytes = await response.read()

            #Wraping audio bytes in a Pipecat frame
            yield TTSAudioRawFrame(
                audio=audio_bytes,
                sample_rate=self.sample_rate,
                num_channels=1
            )

        # 5️⃣ Handle exceptions
        except Exception as e:
            yield ErrorFrame(error=f"Error generating TTS: {e}")

        #TTS has ended
        finally:
            yield TTSStoppedFrame()
