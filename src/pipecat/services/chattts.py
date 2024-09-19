

import aiohttp
from loguru import logger
import requests
from typing import AsyncGenerator
from pipecat.frames.frames import Frame, AudioRawFrame, TTSStartedFrame, TTSStoppedFrame, ErrorFrame
from pipecat.services.ai_services import TTSService


class ChatTTSTTSService(TTSService):
    def __init__(self, *, api_url: str, aiohttp_session: aiohttp.ClientSession, **kwargs):
        super().__init__(**kwargs)
        self.api_url = api_url
        self._aiohttp_session = aiohttp_session
        print(f"chattts url: {self.api_url}")

    async def set_model(self, model: str):
        pass

    async def set_voice(self, voice: str):
        pass

    async def set_language(self, language: str):
        pass

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        try:

            logger.debug(f"ChatTTS Generating TTS: [{text}]")
            payload = {
                "texts": [text],
            }
            await self.start_ttfb_metrics()
            async with self._aiohttp_session.post(self.api_url, json=payload) as response:

                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Error getting audio: {text}")
                await self.start_tts_usage_metrics(text)
                await self.push_frame(TTSStartedFrame())

                async for chunk in response.content.iter_chunked(1024):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = AudioRawFrame(chunk, 24000, 1)
                        yield frame

                await self.push_frame(TTSStoppedFrame())

        except requests.exceptions.RequestException as e:
            yield ErrorFrame(f"Request to ChatTTS failed: {e}")

        except Exception as e:
            yield ErrorFrame(f"Unexpected error: {str(e)}")
