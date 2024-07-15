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

from pipecat.frames.frames import AudioRawFrame, CancelFrame, EndFrame, Frame, StartFrame
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
        super().__init__(aggregate_sentences=True, **kwargs)

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

        self._context_id = None
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        try:
            self._websocket = await websockets.connect(
                f"{self._url}?api_key={self._api_key}&cartesia_version={self._cartesia_version}"
            )
            # self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
        except Exception as e:
            logger.exception(f"{self} initialization error: {e}")

    async def _receive_task_handler(self):
        logger.debug("TOP OF RECEIVE TASK ...")
        async for message in self._websocket:
            logger.debug("RECEIVE TASK LOOP")
            msg = json.loads(message)
            if not msg:
                continue
            logger.debug(f"Received message: {msg}")
            if msg["done"]:
                logger.debug(f"This was a 'done' message, shut down the receive task.")
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

        # async for message in self._websocket:
        #     utterance = json.loads(message)
        #     if not utterance:
        #         continue

        #     logger.debug(f"Received utterance: {utterance}")
        #     return

        #     # TODO: PORT FROM GLADIA
        #     if "error" in utterance:
        #         message = utterance["message"]
        #         logger.error(f"Gladia error: {message}")
        #     elif "confidence" in utterance:
        #         type = utterance["type"]
        #         confidence = utterance["confidence"]
        #         transcript = utterance["transcription"]
        #         if confidence >= self._confidence:
        #             if type == "final":
        #                 await self.queue_frame(TranscriptionFrame(transcript, "", int(time.time_ns() / 1000000)))
        #             else:
        # await self.queue_frame(InterimTranscriptionFrame(transcript, "",
        # int(time.time_ns() / 1000000)))

    async def stop(self, frame: EndFrame):
        self._context_id = None
        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None
        return

    async def cancel(self, frame: CancelFrame):
        self._context_id = None
        if self._receive_task:
            self._receive_task.cancel()
            self._receive_task = None
        return

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")
        logger.debug(
            f"model_id: {self._model_id}, voice_id: {self._voice_id}, language: {self._language}"
        )

        try:
            await self.start_ttfb_metrics()

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
                logger.debug(f"SENDING FIRST MESSAGE {json.dumps(msg)}")
                await self._websocket.send(json.dumps(msg))
                logger.debug("AWAITING FIRST RESPONSE MESSAGE")
                message = await self._websocket.recv()
                msg = json.loads(message)
                logger.debug(f"Received message: {msg}")
                if (msg["type"] == "error"):
                    logger.error(f"Error: {msg}")
                    return
                frame = AudioRawFrame(
                    audio=base64.b64decode(msg["data"]),
                    sample_rate=self._output_format["sample_rate"],
                    num_channels=1
                )
                yield frame
                if not msg["done"]:
                    logger.debug("CREATING RECEIVE TASK")
                    self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
                    # todo: how do we await this task at the app level, so the program doesn't exit?
                    #       we can't await here because we need this function to return
                    # await self._receive_task
            else:
                msg = {
                    "transcript": text,
                    "continue": True,
                    "context_id": self._context_id,
                }
                await asyncio.sleep(0.350)
                logger.debug(f"SENDING FOLLOW MESSAGE {json.dumps(msg)}")
                await self._websocket.send(json.dumps(msg))
                yield None
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
