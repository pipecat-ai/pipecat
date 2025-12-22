#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""[Hathora-hosted](https://models.hathora.dev) speech-to-text services."""

import base64
import os
from typing import AsyncGenerator, Optional

import aiohttp

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

from .utils import ConfigOption


class HathoraSTTService(SegmentedSTTService):
    """This service supports several different speech-to-text models hosted by Hathora.

    [Documentation](https://models.hathora.dev)
    """

    def __init__(
        self,
        *,
        model: str,
        language: Optional[str] = None,
        model_config: Optional[list[ConfigOption]] = None,
        base_url: str = "https://api.models.hathora.dev/inference/v1/stt",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Hathora STT service.

        Args:
            model: Model to use; find available models
                [here](https://models.hathora.dev).
            language: Language code (if supported by model).
            model_config: Some models support additional config, refer to
                [docs](https://models.hathora.dev) for each model to see
                what is supported.
            base_url: Base API URL for the Hathora STT service.
            api_key: API key for authentication with the Hathora service;
                provision one [here](https://models.hathora.dev/tokens).
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(
            **kwargs,
        )
        self._model = model
        self._language = language
        self._model_config = model_config
        self._base_url = base_url
        self._api_key = api_key or os.getenv("HATHORA_API_KEY")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    @traced_stt
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run speech-to-text on the provided audio data.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: Frames containing transcription results (typically TextFrame).
        """
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            url = f"{self._base_url}"

            payload = {
                "model": self._model,
            }

            if self._language is not None:
                payload["language"] = self._language
            if self._model_config is not None:
                payload["model_config"] = [
                    {"name": option.name, "value": option.value} for option in self._model_config
                ]

            base64_audio = base64.b64encode(audio).decode("utf-8")
            payload["audio"] = base64_audio

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                ) as resp:
                    response = await resp.json()

            if response and "text" in response:
                text = response["text"].strip()
                if text:  # Only yield non-empty text
                    # Hathora's API currently doesn't return language info
                    # so we default to the requested language or "en"
                    response_language = self._language or "en"
                    await self._handle_transcription(text, True, response_language)
                    yield TranscriptionFrame(
                        text,
                        self._user_id,
                        time_now_iso8601(),
                        Language(response_language),
                        result=response,
                    )

            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
