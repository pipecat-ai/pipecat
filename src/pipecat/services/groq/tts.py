#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import Frame, TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.tracing import AttachmentStrategy, is_tracing_available, traced

try:
    from groq import AsyncGroq
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Groq, you need to `pip install pipecat-ai[groq]`.")
    raise Exception(f"Missing module: {e}")


class GroqTTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0

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
        return True

    @traced(attachment_strategy=AttachmentStrategy.NONE, name="groq_tts")
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio from text using Groq API with tracing."""
        logger.debug(f"{self}: Generating TTS [{text}]")
        measuring_ttfb = True

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()

            params = {
                "model": self._model_name,
                "voice": self._voice_id,
                "response_format": self._output_format,
                "input": text,
                "speed": self._params.speed,
            }

            response = await self._client.audio.speech.create(**params)

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

        except Exception as e:
            logger.error(f"{self} error generating TTS: {e}")
            if measuring_ttfb:
                await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()

        finally:
            if is_tracing_available():
                from opentelemetry import trace

                from pipecat.utils.tracing.helpers import add_tts_span_attributes

                current_span = trace.get_current_span()
                service_name = self.__class__.__name__.replace("TTSService", "").lower()

                ttfb_ms = None
                if hasattr(self._metrics, "ttfb_ms") and self._metrics.ttfb_ms is not None:
                    ttfb_ms = self._metrics.ttfb_ms

                add_tts_span_attributes(
                    span=current_span,
                    service_name=service_name,
                    model=self._model_name,
                    voice_id=self._voice_id,
                    text=text,
                    settings=self._settings,
                    character_count=len(text),
                    operation_name="tts",
                    ttfb_ms=ttfb_ms,
                )
