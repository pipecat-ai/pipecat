#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.ai_services import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

try:
    from openai import AsyncOpenAI
    from openai.types.audio import Transcription
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI, you need to `pip install pipecat-ai[openai]`. Also, set `OPENAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class BaseWhisperSTTService(SegmentedSTTService):
    """Base class for Whisper-based speech-to-text services.

    Provides common functionality for services implementing the Whisper API interface,
    including metrics generation and error handling.

    Args:
        model: Name of the Whisper model to use.
        api_key: Service API key. Defaults to None.
        base_url: Service API base URL. Defaults to None.
        **kwargs: Additional arguments passed to SegmentedSTTService.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_model_name(model)
        self._client = self._create_client(api_key, base_url)

    def _create_client(self, api_key: Optional[str], base_url: Optional[str]):
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def set_model(self, model: str):
        self.set_model_name(model)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            response = await self._transcribe(audio)

            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()

            text = response.text.strip()

            if text:
                logger.debug(f"Transcription: [{text}]")
                yield TranscriptionFrame(text, "", time_now_iso8601())
            else:
                logger.warning("Received empty transcription from API")

        except Exception as e:
            logger.exception(f"Exception during transcription: {e}")
            yield ErrorFrame(f"Error during transcription: {str(e)}")

    async def _transcribe(self, audio: bytes) -> Transcription:
        """Transcribe audio data to text.

        Args:
            audio: Raw audio data in WAV format.

        Returns:
            Transcription: Object containing the transcribed text.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
