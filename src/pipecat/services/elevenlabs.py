#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp

from typing import AsyncGenerator

from pipecat.frames.frames import AudioRawFrame, ErrorFrame, Frame, MetricsFrame
from pipecat.services.ai_services import TTSService

from loguru import logger


class ElevenLabsTTSService(TTSService):

    def __init__(
            self,
            *,
            api_key: str,
            voice_id: str,
            aiohttp_session: aiohttp.ClientSession,
            model: str = "eleven_turbo_v2",
            **kwargs):
        super().__init__(**kwargs)

        self._api_key = api_key
        self._voice_id = voice_id
        self._model = model
        self._aiohttp_session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    async def set_voice(self, voice: str):
        logger.debug(f"Switching TTS voice to: [{voice}]")
        self._voice_id = voice

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")
        if self.can_generate_metrics() and self.metrics_enabled:
            characters = {
                "processor": self.name,
                "value": len(text),
            }
            logger.debug(f"{self.name} Characters: {characters['value']}")
            await self.push_frame(MetricsFrame(characters=[characters]))
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"

        payload = {"text": text, "model_id": self._model}

        querystring = {
            "output_format": "pcm_16000",
            "optimize_streaming_latency": 2}

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

            async for chunk in r.content:
                if len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    frame = AudioRawFrame(chunk, 16000, 1)
                    yield frame
