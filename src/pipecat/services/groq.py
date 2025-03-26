#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#


from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.ai_services import TTSService
from pipecat.services.base_whisper import BaseWhisperSTTService, Transcription
from pipecat.services.openai import OpenAILLMService
from pipecat.transcriptions.language import Language

try:
    from groq import AsyncGroq
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Groq, you need to `pip install pipecat-ai[groq]`. Also, set a `GROQ_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


class GroqLLMService(OpenAILLMService):
    """A service for interacting with Groq's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Groq's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.

    Args:
        api_key (str): The API key for accessing Groq's API
        base_url (str, optional): The base URL for Groq API. Defaults to "https://api.groq.com/openai/v1"
        model (str, optional): The model identifier to use. Defaults to "llama-3.3-70b-versatile"
        **kwargs: Additional keyword arguments passed to OpenAILLMService
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "llama-3.3-70b-versatile",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    def create_client(self, api_key=None, base_url=None, **kwargs):
        """Create OpenAI-compatible client for Groq API endpoint."""
        logger.debug(f"Creating Groq client with api {base_url}")
        return super().create_client(api_key, base_url, **kwargs)


class GroqSTTService(BaseWhisperSTTService):
    """Groq Whisper speech-to-text service.

    Uses Groq's Whisper API to convert audio to text. Requires a Groq API key
    set via the api_key parameter or GROQ_API_KEY environment variable.

    Args:
        model: Whisper model to use. Defaults to "whisper-large-v3-turbo".
        api_key: Groq API key. Defaults to None.
        base_url: API base URL. Defaults to "https://api.groq.com/openai/v1".
        language: Language of the audio input. Defaults to English.
        prompt: Optional text to guide the model's style or continue a previous segment.
        temperature: Optional sampling temperature between 0 and 1. Defaults to 0.0.
        **kwargs: Additional arguments passed to BaseWhisperSTTService.
    """

    def __init__(
        self,
        *,
        model: str = "whisper-large-v3-turbo",
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        language: Optional[Language] = Language.EN,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            language=language,
            prompt=prompt,
            temperature=temperature,
            **kwargs,
        )

    async def _transcribe(self, audio: bytes) -> Transcription:
        assert self._language is not None  # Assigned in the BaseWhisperSTTService class

        # Build kwargs dict with only set parameters
        kwargs = {
            "file": ("audio.wav", audio, "audio/wav"),
            "model": self.model_name,
            "response_format": "json",
            "language": self._language,
        }

        if self._prompt is not None:
            kwargs["prompt"] = self._prompt

        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        return await self._client.audio.transcriptions.create(**kwargs)


class GroqTTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0
        seed: Optional[int] = None

    GROQ_SAMPLE_RATE = 48000  # Groq TTS only supports 48kHz sample rate

    def __init__(
        self,
        *,
        api_key: str,
        output_format: str = "wav",
        params: InputParams = InputParams(),
        model_name: str = "playai-tts",
        voice_id: str = "Celeste-PlayAI",
        sample_rate: Optional[int] = GROQ_SAMPLE_RATE,
        **kwargs,
    ):
        if sample_rate != self.GROQ_SAMPLE_RATE:
            logger.warning(f"Groq TTS only supports {self.GROQ_SAMPLE_RATE}Hz sample rate. ")
        super().__init__(
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        self._api_key = api_key
        self._model_name = model_name
        self._output_format = output_format
        self._voice_id = voice_id
        self._params = params

        self._client = AsyncGroq(api_key=self._api_key)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        measuring_ttfb = True
        await self.start_ttfb_metrics()
        yield TTSStartedFrame()

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
            # remove wav header if present
            if data.startswith(b"RIFF"):
                data = data[44:]
                if len(data) == 0:
                    continue
            yield TTSAudioRawFrame(data, self.sample_rate, 1)

        yield TTSStoppedFrame()
