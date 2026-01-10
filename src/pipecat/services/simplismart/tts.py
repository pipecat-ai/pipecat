from typing import AsyncGenerator, Optional
import httpx
from openai import BadRequestError
from pipecat.frames.frames import (
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

TIMEOUT = 60.0
class SimplismartTTSService(TTSService):
    """OpenAI Text-to-Speech service that generates audio from text.

    This service uses the OpenAI TTS API to generate PCM-encoded audio at 24kHz.
    Supports multiple voice models and configurable parameters for high-quality
    speech synthesis with streaming audio output.
    """

    OPENAI_SAMPLE_RATE = 24000  # OpenAI TTS always outputs at 24kHz

    class InputParams(BaseModel):
        max_tokens: int = 1000
        temperature: float = 0.7
        top_p: float = 0.9
        repetition_penalty: float = 1.5

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        voice: str = "tara",
        model: str = "orpheus",
        sample_rate: Optional[int] = 24000,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize OpenAI TTS service.

        Args:
            api_key: OpenAI API key for authentication. If None, uses environment variable.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            voice: Voice ID to use for synthesis. Defaults to "alloy".
            model: TTS model to use. Defaults to "gpt-4o-mini-tts".
            sample_rate: Output audio sample rate in Hz. If None, uses OpenAI's default 24kHz.
            instructions: Optional instructions to guide voice synthesis behavior.
            **kwargs: Additional keyword arguments passed to TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self.set_model_name(model)
        self.set_voice(voice)
        self._base_url = base_url
        self._params = params or SimplismartTTSService.InputParams()
        self._headers = {
            "Authorization": f"Bearer {api_key}",
        }


    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        logger.info(f"Switching TTS model to: [{model}]")
        self.set_model_name(model)

    async def start(self, frame: StartFrame):
        """Start the OpenAI TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self.sample_rate != self.OPENAI_SAMPLE_RATE:
            logger.warning(
                f"OpenAI TTS requires {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {self.sample_rate}Hz may cause issues."
            )

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using OpenAI's TTS API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech data.
        """
        payload = self._params.model_dump()
        payload["prompt"] = text
        payload["voice"] = self._voice_id

        await self.start_ttfb_metrics()

        timeout = httpx.Timeout(TIMEOUT, read=TIMEOUT)

        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", self._base_url, json=payload, headers=self._headers) as r:
                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()
                async for chunk in r.aiter_bytes(1024):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, 24000, 1)
                        yield frame
            yield TTSStoppedFrame()