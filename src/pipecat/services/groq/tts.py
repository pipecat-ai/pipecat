#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Groq text-to-speech service implementation."""

import io
import wave
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from groq import AsyncGroq
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Groq, you need to `pip install pipecat-ai[groq]`.")
    raise Exception(f"Missing module: {e}")


class GroqTTSService(TTSService):
    """Groq text-to-speech service implementation.

    Provides text-to-speech synthesis using Groq's TTS API. The service
    operates at a fixed 48kHz sample rate and supports various voices
    and output formats.
    """

    class InputParams(BaseModel):
        """Input parameters for Groq TTS configuration.

        Parameters:
            language: Language for speech synthesis. Defaults to English.
            speed: Speech speed multiplier. Defaults to 1.0.
        """

        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0

    GROQ_SAMPLE_RATE = 48000  # Groq TTS only supports 48kHz sample rate

    def __init__(
        self,
        *,
        api_key: str,
        output_format: str = "wav",
        params: Optional[InputParams] = None,
        model_name: str = "playai-tts",
        voice_id: str = "Celeste-PlayAI",
        sample_rate: Optional[int] = GROQ_SAMPLE_RATE,
        **kwargs,
    ):
        """Initialize Groq TTS service.

        Args:
            api_key: Groq API key for authentication.
            output_format: Audio output format. Defaults to "wav".
            params: Additional input parameters for voice customization.
            model_name: TTS model to use. Defaults to "playai-tts".
            voice_id: Voice identifier to use. Defaults to "Celeste-PlayAI".
            sample_rate: Audio sample rate. Must be 48000 Hz for Groq TTS.
            **kwargs: Additional arguments passed to parent TTSService class.
        """
        if sample_rate != self.GROQ_SAMPLE_RATE:
            logger.warning(f"Groq TTS only supports {self.GROQ_SAMPLE_RATE}Hz sample rate. ")

        super().__init__(
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or GroqTTSService.InputParams()

        self._api_key = api_key
        self._model_name = model_name
        self._output_format = output_format
        self._voice_id = voice_id
        self._params = params

        self._settings = {
            "model": model_name,
            "voice_id": voice_id,
            "output_format": output_format,
            "language": str(params.language) if params.language else "en",
            "speed": params.speed,
            "sample_rate": sample_rate,
        }

        self._client = AsyncGroq(api_key=self._api_key)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Groq TTS service supports metrics generation.
        """
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Groq's TTS API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech data.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        measuring_ttfb = True
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

        try:
            response = await self._client.audio.speech.create(
                model=self._model_name,
                voice=self._voice_id,
                response_format=self._output_format,
                input=text,
            )

            async for data in response.iter_bytes():
                if measuring_ttfb:
                    await self.stop_ttfb_metrics()
                    measuring_ttfb = False

                with wave.open(io.BytesIO(data)) as w:
                    channels = w.getnchannels()
                    frame_rate = w.getframerate()
                    num_frames = w.getnframes()
                    bytes = w.readframes(num_frames)
                    yield TTSAudioRawFrame(bytes, frame_rate, channels)
        except Exception as e:
            logger.error(f"{self} exception: {e}")

        yield TTSStoppedFrame()
