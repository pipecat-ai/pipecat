#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from cartesia.tts import AsyncCartesiaTTS

from typing import AsyncGenerator

from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.services.ai_services import TTSService

from loguru import logger


class CartesiaTTSService(TTSService):

    def __init__(
            self,
            *,
            api_key: str,
            voice_name: str,
            model_id: str = "upbeat-moon",
            output_format: str = "pcm_16000",
            **kwargs):
        super().__init__(**kwargs)

        self._api_key = api_key
        self._voice_name = voice_name
        self._model_id = model_id
        self._output_format = output_format

        try:
            self._client = AsyncCartesiaTTS(api_key=self._api_key)
            voices = self._client.get_voices()
            voice_id = voices[self._voice_name]["id"]
            self._voice = self._client.get_voice_embedding(voice_id=voice_id)
        except Exception as e:
            logger.error(f"Cartesia initialization error: {e}")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            chunk_generator = await self._client.generate(
                stream=True,
                transcript=text,
                voice=self._voice,
                model_id=self._model_id,
                output_format=self._output_format,
            )

            async for chunk in chunk_generator:
                yield AudioRawFrame(chunk["audio"], chunk["sampling_rate"], 1)
        except Exception as e:
            logger.error(f"Cartesia exception: {e}")
