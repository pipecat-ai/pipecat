#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp

from typing import AsyncGenerator, Literal
from pydantic import BaseModel

from pipecat.frames.frames import AudioRawFrame, ErrorFrame, Frame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.ai_services import TTSService

from loguru import logger


class ElevenLabsTTSService(TTSService):
    class InputParams(BaseModel):
        output_format: Literal["pcm_16000", "pcm_22050", "pcm_24000", "pcm_44100"] = "pcm_16000"

    def __init__(
            self,
            *,
            api_key: str,
            voice_id: str,
            aiohttp_session: aiohttp.ClientSession,
            model: str = "eleven_turbo_v2_5",
            params: InputParams = InputParams(),
            **kwargs):
        super().__init__(**kwargs)

        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._params = params
        self._aiohttp_session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice_id = voice

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"

        payload = {"text": text, "model_id": self._model}

        querystring = {
            "output_format": self._params.output_format
        }

        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        await self.start_ttfb_metrics()

        async with self._aiohttp_session.post(url, json=payload, headers=headers, params=querystring) as r:
            if r.status != 200:
                text = await r.text()
                logger.error(f"{self} error getting audio (status: {r.status}, error: {text})")
                yield ErrorFrame(f"Error getting audio (status: {r.status}, error: {text})")
                return

            await self.start_tts_usage_metrics(text)

            await self.push_frame(TTSStartedFrame())
            async for chunk in r.content:
                if len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    frame = AudioRawFrame(chunk, 16000, 1)
                    yield frame
            await self.push_frame(TTSStoppedFrame())
