#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from cartesia import AsyncCartesia

from typing import AsyncGenerator

from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.services.ai_services import TTSService

from loguru import logger


class CartesiaTTSService(TTSService):

    def __init__(
            self,
            *,
            api_key: str,
            voice_id: str,
            model_id: str = "sonic-english",
            encoding: str = "pcm_s16le",
            sample_rate: int = 16000,
            **kwargs):
        super().__init__(**kwargs)

        self._api_key = api_key
        self._model_id = model_id
        self._output_format = {
            "container": "raw",
            "encoding": encoding,
            "sample_rate": sample_rate,
        }

        try:
            self._client = AsyncCartesia(api_key=self._api_key)
            self._voice = self._client.voices.get(id=voice_id)
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            await self.start_ttfb_metrics()

            chunk_generator = await self._client.tts.sse(
                stream=True,
                transcript=text,
                voice_embedding=self._voice["embedding"],
                model_id=self._model_id,
                output_format=self._output_format,
            )

            async for chunk in chunk_generator:
                await self.stop_ttfb_metrics()
                yield AudioRawFrame(chunk["audio"], self._output_format["sample_rate"], 1)
        except Exception as e:
            logger.error(f"{self} exception: {e}")
