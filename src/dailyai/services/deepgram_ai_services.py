import aiohttp
import asyncio
import os

import requests

from collections.abc import AsyncGenerator
from dailyai.services.ai_services import TTSService


class DeepgramTTSService(TTSService):
    def __init__(self, *, aiohttp_session, api_key, voice="alpha-asteria-en-v2"):
        super().__init__()

        self._voice = voice
        self._api_key = api_key
        self._aiohttp_session = aiohttp_session

    def get_mic_sample_rate(self):
        return 24000

    async def run_tts(self, sentence) -> AsyncGenerator[bytes, None]:
        self.logger.info(f"Running deepgram tts for {sentence}")
        base_url = "https://api.beta.deepgram.com/v1/speak"
        request_url = f"{base_url}?model={self._voice}&encoding=linear16&container=none&sample_rate=16000"
        headers = {"authorization": f"token {self._api_key}"}
        body = {"text": sentence}
        async with self._aiohttp_session.post(request_url, headers=headers, json=body) as r:
            async for data in r.content:
                yield data
