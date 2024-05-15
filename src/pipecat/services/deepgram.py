#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp

from typing import AsyncGenerator

from pipecat.frames.frames import AudioRawFrame, Frame
from pipecat.services.ai_services import TTSService

from loguru import logger


class DeepgramTTSService(TTSService):

    def __init__(
            self,
            *,
            aiohttp_session: aiohttp.ClientSession,
            api_key: str,
            voice: str = "alpha-asteria-en-v2",
            **kwargs):
        super().__init__(**kwargs)

        self._voice = voice
        self._api_key = api_key
        self._aiohttp_session = aiohttp_session

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.info(f"Running Deepgram TTS for {text}")
        base_url = "https://api.beta.deepgram.com/v1/speak"
        request_url = f"{base_url}?model={self._voice}&encoding=linear16&container=none&sample_rate=16000"
        headers = {"authorization": f"token {self._api_key}"}
        body = {"text": text}
        async with self._aiohttp_session.post(request_url, headers=headers, json=body) as r:
            async for data in r.content:
                frame = AudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
                yield frame
