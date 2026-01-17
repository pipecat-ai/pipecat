#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Uplift TTS service integration."""

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
            max_retries: Maximum number of retries for TTS requests.
            voice_id: Optional default voice ID override.
        """

        max_retries: int = 5
        voice_id: Optional[str] = None

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
        super().__init__(**kwargs)

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session

        if not self._api_key:
            raise ValueError("Missing Uplift API key")

        self._params = params or UpliftTTSService.InputParams()

        # Determine final voice ID
        final_voice_id = self._params.voice_id or voice_id
        self.set_voice(final_voice_id)

        logger.debug(f"Uplift TTS initialized with voice_id={final_voice_id}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Uplift's TTS endpoint."""

        # TTS has started
        yield TTSStartedFrame()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "voiceId": self._voice_id,
            "outputFormat": "WAV_22050_16",
        }

        try:
            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    yield ErrorFrame(
                        error=f"Uplift TTS error: {response.status}"
                    )
                    return

                audio_bytes = await response.read()

            # Wrap audio bytes in a Pipecat frame
            yield TTSAudioRawFrame(
                audio=audio_bytes,
                sample_rate=self.sample_rate,
                num_channels=1,
            )

        except Exception as e:
            yield ErrorFrame(error=f"Error generating TTS: {e}")

        finally:
            # TTS has ended
            yield TTSStoppedFrame()
