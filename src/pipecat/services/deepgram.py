#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import aiohttp

from typing import AsyncGenerator

from pipecat.frames.frames import AudioRawFrame, ErrorFrame, Frame
from pipecat.services.ai_services import TTSService

from loguru import logger


class DeepgramTTSService(TTSService):

    def __init__(
            self,
            *,
            aiohttp_session: aiohttp.ClientSession,
            api_key: str,
            voice: str = "aura-helios-en",
            base_url: str = "https://api.deepgram.com/v1/speak",
            **kwargs):
        super().__init__(**kwargs)

        self._voice = voice
        self._api_key = api_key
        self._aiohttp_session = aiohttp_session
        self._base_url = base_url

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        base_url = self._base_url
        request_url = f"{base_url}?model={self._voice}&encoding=linear16&container=none&sample_rate=16000"
        headers = {"authorization": f"token {self._api_key}"}
        body = {"text": text}

        try:
            await self.start_ttfb_metrics()
            async with self._aiohttp_session.post(request_url, headers=headers, json=body) as r:
                if r.status != 200:
                    response_text = await r.text()
                    # If we get a a "Bad Request: Input is unutterable", just print out a debug log.
                    # All other unsuccesful requests should emit an error frame. If not specifically
                    # handled by the running PipelineTask, the ErrorFrame will cancel the task.
                    if "unutterable" in response_text:
                        logger.debug(f"Unutterable text: [{text}]")
                        return

                    logger.error(
                        f"Error getting audio (status: {r.status}, error: {response_text})")
                    yield ErrorFrame(f"Error getting audio (status: {r.status}, error: {response_text})")
                    return

                async for data in r.content:
                    await self.stop_ttfb_metrics()
                    frame = AudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
                    yield frame
        except Exception as e:
            logger.error(f"Deepgram exception: {e}")
