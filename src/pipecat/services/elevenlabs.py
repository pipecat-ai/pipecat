#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp

from typing import AsyncGenerator

from pipecat.frames.frames import AudioRawFrame, ErrorFrame, Frame, TTSStartedFrame, TTSStoppedFrame
from pipecat.services.ai_services import TTSService

from loguru import logger


class ElevenLabsTTSService(TTSService):

    def __init__(
            self,
            *,
            aiohttp_session: aiohttp.ClientSession,
            api_key: str,
            voice_id: str,
            model: str = "eleven_turbo_v2",
            **kwargs):
        super().__init__(**kwargs)

        self._api_key = api_key
        self._voice_id = voice_id
        self._aiohttp_session = aiohttp_session
        self._model = model

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Transcribing text: {text}")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"

        payload = {"text": text, "model_id": self._model}

        querystring = {
            "output_format": "pcm_16000",
            "optimize_streaming_latency": 2}

        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        async with self._aiohttp_session.post(url, json=payload, headers=headers, params=querystring) as r:
            if r.status != 200:
                logger.error(f"Audio fetch status code: {r.status}, error: {r.text}")
                yield ErrorFrame(f"Audio fetch status code: {r.status}, error: {r.text}")
                return

            yield TTSStartedFrame()
            async for chunk in r.content:
                if len(chunk) > 0:
                    frame = AudioRawFrame(chunk, 16000, 1)
                    yield frame
            yield TTSStoppedFrame()
