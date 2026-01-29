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
from pydantic import BaseModel

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

    class InputParams(BaseModel):
        """Optional input parameters for Hathora STT configuration.

        Parameters:
            language: Language code (if supported by model).
            config: Some models support additional config, refer to
                [docs](https://models.hathora.dev) for each model to see
                what is supported.
        """

        language: Optional[str] = None
        config: Optional[list[ConfigOption]] = None

    def __init__(
        self,
        *,
        model: str,
        sample_rate: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.models.hathora.dev/inference/v1/stt",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Hathora STT service.

        Args:
            model: Model to use; find available models
                [here](https://models.hathora.dev).
            sample_rate: The sample rate for audio input. If None, will be determined
                from the start frame.
            api_key: API key for authentication with the Hathora service;
                provision one [here](https://models.hathora.dev/tokens).
            base_url: Base API URL for the Hathora STT service.
            params: Configuration parameters.
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(
            sample_rate=sample_rate,
            **kwargs,
        )
        self._model = model
        self._api_key = api_key or os.getenv("HATHORA_API_KEY")
        self._base_url = base_url

        params = params or HathoraSTTService.InputParams()

        self._settings = {
            "language": params.language,
            "config": params.config,
        }

        self.set_model_name(model)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True
        """
        return True

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run speech-to-text on the provided audio data.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: Frames containing transcription results (typically TextFrame).
        """
        try:
            await self.start_processing_metrics()

            url = f"{self._base_url}"

            payload = {
                "model": self._model,
            }

            if self._settings["language"] is not None:
                payload["language"] = self._settings["language"]
            if self._settings["config"] is not None:
                payload["model_config"] = [
                    {"name": option.name, "value": option.value}
                    for option in self._settings["config"]
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
                    response_language = self._settings["language"] or "en"
                    await self._handle_transcription(text, True, response_language)
                    yield TranscriptionFrame(
                        text,
                        self._user_id,
                        time_now_iso8601(),
                        Language(response_language),
                        result=response,
                    )

            await self.stop_processing_metrics()

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
