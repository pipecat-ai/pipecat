

import aiohttp
from loguru import logger
import requests
from typing import AsyncGenerator
from pipecat.frames.frames import Frame, AudioRawFrame, OutputAudioRawFrame, TTSStartedFrame, TTSStoppedFrame, ErrorFrame
from pipecat.services.ai_services import TTSService


class ChatTTSTTSService(TTSService):
    def __init__(
        self,
        *,
        api_url: str,
        aiohttp_session: aiohttp.ClientSession,
        sample_rate: int = 24000,
        num_channels: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.api_url = api_url
        self._aiohttp_session = aiohttp_session
        self._settings = {
            "sample_rate": sample_rate,
            "num_channels": num_channels,
        }

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
                    yield ErrorFrame("Error getting audio: {text}")
                    return
                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()

                async for chunk in response.content.iter_chunked(1024):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = OutputAudioRawFrame(
                            audio=chunk,
                            sample_rate=self._settings["sample_rate"],
                            num_channels=self._settings["num_channels"],
                        )
                        yield frame

                yield TTSStoppedFrame()

        except requests.exceptions.RequestException as e:
            yield ErrorFrame(f"Request to ChatTTS failed: {e}")

        except Exception as e:
            yield ErrorFrame(f"Unexpected error: {str(e)}")
