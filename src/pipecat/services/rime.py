#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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
from pipecat.services.ai_services import TTSService


class RimeHttpTTSService(TTSService):
    class InputParams(BaseModel):
        pause_between_brackets: Optional[bool] = False
        phonemize_between_brackets: Optional[bool] = False
        inline_speed_alpha: Optional[str] = None
        speed_alpha: Optional[float] = 1.0
        reduce_latency: Optional[bool] = False

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "eva",
        model: str = "mist",
        sample_rate: int = 24000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._base_url = "https://users.rime.ai/v1/rime-tts"
        self._settings = {
            "samplingRate": sample_rate,
            "speedAlpha": params.speed_alpha,
            "reduceLatency": params.reduce_latency,
            "pauseBetweenBrackets": params.pause_between_brackets,
            "phonemizeBetweenBrackets": params.phonemize_between_brackets,
        }
        self.set_voice(voice_id)
        self.set_model_name(model)

        if params.inline_speed_alpha:
            self._settings["inlineSpeedAlpha"] = params.inline_speed_alpha

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        headers = {
            "Accept": "audio/pcm",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = self._settings.copy()
        payload["text"] = text
        payload["speaker"] = self._voice_id
        payload["modelId"] = self._model_name

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            async with aiohttp.ClientSession() as session:
                async with session.post(self._base_url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        error_message = f"Rime TTS error: HTTP {response.status}"
                        logger.error(error_message)
                        yield ErrorFrame(error=error_message)
                        return

                    # Process the streaming response
                    chunk_size = 8192
                    first_chunk = True

                    async for chunk in response.content.iter_chunked(chunk_size):
                        if first_chunk:
                            await self.stop_ttfb_metrics()
                            first_chunk = False

                        if chunk:
                            frame = TTSAudioRawFrame(chunk, self._settings["samplingRate"], 1)
                            yield frame

            yield TTSStoppedFrame()

        except Exception as e:
            logger.exception(f"Error generating TTS: {e}")
            yield ErrorFrame(error=f"Rime TTS error: {str(e)}")

        finally:
            yield TTSStoppedFrame()
