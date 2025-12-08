#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""[Hathora-hosted](https://models.hathora.dev) speech-to-text services."""

import os
from typing import Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

class ParakeetTDTSTTService(SegmentedSTTService):
    """Parakeet TDT is a multilingual automatic speech recognition model
    with word-level timestamps.

    This service uses the Hathora-hosted Parakeet model via the HTTP API.

    [Documentation](https://models.hathora.dev/model/nvidia-parakeet-tdt-0.6b-v3)
    """

    def __init__(
        self,
        *,
        base_url = None,
        api_key = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the Hathora-hosted Parakeet STT service.

        Args:
            base_url: Base URL for the Hathora Parakeet STT API.
            api_key: API key for authentication with the Hathora service;
                provisiion one [here](https://models.hathora.dev/tokens).
            start_time: Start time in seconds for the time window.
            end_time: End time in seconds for the time window.
        """
        super().__init__(
            **kwargs,
        )
        self._base_url = base_url
        self._api_key = api_key
        self._start_time = start_time
        self._end_time = end_time

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes):
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            url = f"{self._base_url}"

            url_query_params = []
            if self._start_time is not None:
                url_query_params.append(f"start_time={self._start_time}")
            if self._end_time is not None:
                url_query_params.append(f"end_time={self._end_time}")
            url_query_params.append(f"sample_rate={self.sample_rate}")

            if len(url_query_params) > 0:
                url += "?" + "&".join(url_query_params)

            api_key = self._api_key or os.getenv("HATHORA_API_KEY")

            form_data = aiohttp.FormData()
            form_data.add_field("file", audio, filename="audio.wav", content_type="application/octet-stream")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=form_data,
                ) as resp:
                    response = await resp.json()

            if response and "text" in response:
                text = response["text"].strip()
                if text:  # Only yield non-empty text
                    await self.stop_ttfb_metrics()
                    await self.stop_processing_metrics()
                    logger.debug(f"Transcription: [{text}]")
                    yield TranscriptionFrame(
                        text,
                        self._user_id,
                        time_now_iso8601(),
                        Language("en"), # TODO: the parakeet hathora API doesn't accept a language but says it's multilingual
                        result=response,
                    )

        except Exception as e:
            logger.error(f"Hathora error: {e}")
            yield ErrorFrame(f"Hathora error: {str(e)}")
