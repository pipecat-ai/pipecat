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
