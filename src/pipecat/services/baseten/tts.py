import os
import asyncio
import base64
import json
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional, Tuple, Union
import requests

import aiohttp
from loguru import logger
from pydantic import BaseModel

import pyaudio

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import (
    TTSService,
    AudioContextWordTTSService,
    WordTTSService
)
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Baseten, you need to `pip install pipecat-ai[baseten]`.")
    raise Exception(f"Missing module: {e}")

class BasetenTTSService(TTSService):
    def __init__(self, 
                 *, 
                 api_key: str, 
                 base_url: str,
                 model: str = "orpheus-3b",
                 voice: str = "tara",
                 language: str = "en",
                 temperature: float = 0.6,
                 sample_rate: Optional[int] = None,
                 **kwargs
                 ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self.set_model_name(model)
        self.set_voice(voice)
        self._chunk_size = 4096
        self._base_url = base_url

        self._payload = {
            "prompt": "",
            "max_tokens": 4096,
            "voice": voice,
            "stop_token_ids": [128258, 128009],
            "repetition_penalty": 1.1,
            "temperature": temperature,
            "top_p": 0.9,
        }

        self._headers={"Authorization": f"Api-Key {api_key}"}

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        logger.info(f"Switching TTS model to: [{model}]")
        self.set_model_name(model)

    async def start(self, frame: StartFrame):
        await super().start(frame)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")


        try:
            # set up your request payload
            self._payload["prompt"] = text

            await self.start_ttfb_metrics()

            # fire the POST, await the entire body
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._base_url,
                    headers=self._headers,
                    json=self._payload,     # no more stream=True
                ) as resp:
                    # error handling
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Baseten TTS API error: {error_text}")
                        await self.push_error(ErrorFrame(f"Baseten TTS API error: {error_text}"))
                        raise Exception(f"Baseten TTS API returned status {resp.status}: {error_text}")

                    
                    await self.start_tts_usage_metrics(text)

                    yield TTSStartedFrame()

                    async for chunk in resp.content.iter_chunked(self._chunk_size):
                        if chunk:
                            await self.stop_ttfb_metrics()
                            yield TTSAudioRawFrame(audio=chunk,
                                                sample_rate=self.sample_rate,
                                                num_channels=1)


                    yield TTSStoppedFrame()
        
        except Exception as e:
            logger.error(f"{self} exception: {e}")