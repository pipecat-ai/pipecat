import os
import aiohttp
import requests

from dailyai.services.ai_services import TTSService


class DeepgramAIService(TTSService):
    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        api_key,
        voice,
        sample_rate=16000
    ):
        super().__init__()

        self._api_key = api_key
        self._voice = voice
        self._sample_rate = sample_rate
        self._aiohttp_session = aiohttp_session

    async def run_tts(self, sentence):
        self.logger.info(f"Running deepgram tts for {sentence}")
        base_url = "https://api.beta.deepgram.com/v1/speak"
        request_url = f"{base_url}?model={self._voice}&encoding=linear16&container=none&sample_rate={self._sample_rate}"
        headers = {"authorization": f"token {self._api_key}", "Content-Type": "application/json"}
        data = {"text": sentence}

        async with self._aiohttp_session.post(
            request_url, headers=headers, json=data
        ) as r:
            async for chunk in r.content:
                if chunk:
                    yield chunk
