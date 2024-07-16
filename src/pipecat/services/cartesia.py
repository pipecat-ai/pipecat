#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import uuid
import base64
import asyncio

from typing import AsyncGenerator

from pipecat.processors.frame_processor import FrameDirection
from pipecat.frames.frames import Frame, AudioRawFrame, StartInterruptionFrame
from pipecat.services.ai_services import TTSService

from loguru import logger

# See .env.example for Cartesia configuration needed
try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Cartesia, you need to `pip install pipecat-ai[cartesia]`. Also, set `CARTESIA_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class CartesiaTTSService(TTSService):

    def __init__(
            self,
            *,
            api_key: str,
            cartesia_version: str = "2024-06-10",
            url: str = "wss://api.cartesia.ai/tts/websocket",
            voice_id: str,
            model_id: str = "sonic-english",
            encoding: str = "pcm_s16le",
            sample_rate: int = 16000,
            **kwargs):
        super().__init__(**kwargs)

        self._api_key = api_key
        self._cartesia_version = cartesia_version
        self._url = url
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = {
            "container": "raw",
            "encoding": encoding,
            "sample_rate": sample_rate,
        }
        self._language = "en"

        self._websocket = None
        self._context_id = None
        self._receive_task = None
        self._waiting_for_ttfb = False

    def can_generate_metrics(self) -> bool:
        return True

    async def connect(self):
        try:
            self._websocket = await websockets.connect(
                f"{self._url}?api_key={self._api_key}&cartesia_version={self._cartesia_version}"
            )
        except Exception as e:
            logger.exception(f"{self} initialization error: {e}")

    async def disconnect(self):
        try:
            if self._websocket:
                ws = self._websocket
                self._websocket = None
                await ws.close()
        except Exception as e:
            logger.exception(f"{self} error closing websocket: {e}")

    async def handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super().handle_interruption(frame, direction)
        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None
            await self.disconnect()
        await self.stop_all_metrics()

    async def _receive_task_handler(self):
        async for message in self._websocket:
            msg = json.loads(message)
            if not msg:
                continue
            # logger.debug(f"Received message: {msg}")
            if self._waiting_for_ttfb:
                await self.stop_ttfb_metrics()
                self._waiting_for_ttfb = False
            if msg["done"]:
                self._context_id = None
                if self._receive_task:
                    self._receive_task.cancel()
                    self._receive_task = None
                return
            frame = AudioRawFrame(
                audio=base64.b64decode(msg["data"]),
                sample_rate=self._output_format["sample_rate"],
                num_channels=1
            )
            await self.push_frame(frame)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._websocket:
                await self.connect()

            if not self._waiting_for_ttfb:
                await self.start_ttfb_metrics()
                self._waiting_for_ttfb = True

            if not self._context_id:
                self._context_id = str(uuid.uuid4())

            msg = {
                "transcript": text,
                "continue": True,
                "context_id": self._context_id,
                "model_id": self._model_id,
                "voice": {
                    "mode": "id",
                    "id": self._voice_id
                },
                "output_format": self._output_format,
                "language": self._language,
            }
            # logger.debug(f"SENDING MESSAGE {json.dumps(msg)}")
            await self._websocket.send(json.dumps(msg))
            if not self._receive_task:
                # todo: how do we await this task at the app level, so the program doesn't exit?
                #       we can't await here because we need this function to return
                self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
            yield None
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
