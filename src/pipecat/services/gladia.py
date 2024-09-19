#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json

from typing import AsyncGenerator, Optional
from pydantic.main import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame)
from pipecat.services.ai_services import STTService
from pipecat.utils.time import time_now_iso8601

from loguru import logger

# See .env.example for Gladia configuration needed
try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Gladia, you need to `pip install pipecat-ai[gladia]`. Also, set `GLADIA_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class GladiaSTTService(STTService):
    class InputParams(BaseModel):
        sample_rate: Optional[int] = 16000
        language: Optional[str] = "english"
        transcription_hint: Optional[str] = None
        endpointing: Optional[int] = 200
        prosody: Optional[bool] = None

    def __init__(self,
                 *,
                 api_key: str,
                 url: str = "wss://api.gladia.io/audio/text/audio-transcription",
                 confidence: float = 0.5,
                 params: InputParams = InputParams(),
                 **kwargs):
        super().__init__(sync=False, **kwargs)

        self._api_key = api_key
        self._url = url
        self._params = params
        self._confidence = confidence

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._websocket = await websockets.connect(self._url)
        self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())
        await self._setup_gladia()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._websocket.close()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._websocket.close()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self.start_processing_metrics()
        await self._send_audio(audio)
        await self.stop_processing_metrics()
        yield None

    async def _setup_gladia(self):
        configuration = {
            "x_gladia_key": self._api_key,
            "encoding": "WAV/PCM",
            "model_type": "fast",
            "language_behaviour": "manual",
            **self._params.model_dump(exclude_none=True)
        }

        await self._websocket.send(json.dumps(configuration))

    async def _send_audio(self, audio: bytes):
        message = {
            'frames': base64.b64encode(audio).decode("utf-8")
        }
        await self._websocket.send(json.dumps(message))

    async def _receive_task_handler(self):
        async for message in self._websocket:
            utterance = json.loads(message)
            if not utterance:
                continue

            if "error" in utterance:
                message = utterance["message"]
                logger.error(f"Gladia error: {message}")
            elif "confidence" in utterance:
                type = utterance["type"]
                confidence = utterance["confidence"]
                transcript = utterance["transcription"]
                if confidence >= self._confidence:
                    if type == "final":
                        await self.push_frame(TranscriptionFrame(transcript, "", time_now_iso8601()))
                    else:
                        await self.push_frame(InterimTranscriptionFrame(transcript, "", time_now_iso8601()))
