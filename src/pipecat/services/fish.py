#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import uuid
from typing import AsyncGenerator, Literal, Optional

from loguru import logger
from pydantic import BaseModel
from tenacity import AsyncRetrying, RetryCallState, stop_after_attempt, wait_exponential

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import TTSService
from pipecat.transcriptions.language import Language

try:
    import ormsgpack
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Fish Audio, you need to `pip install pipecat-ai[fish]`. Also, set `FISH_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")

# FishAudio supports various output formats
FishAudioOutputFormat = Literal["opus", "mp3", "pcm", "wav"]


class FishAudioTTSService(TTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        latency: Optional[str] = "normal"  # "normal" or "balanced"
        prosody_speed: Optional[float] = 1.0  # Speech speed (0.5-2.0)
        prosody_volume: Optional[int] = 0  # Volume adjustment in dB

    def __init__(
        self,
        *,
        api_key: str,
        model: str,  # This is the reference_id
        output_format: FishAudioOutputFormat = "pcm",
        sample_rate: int = 24000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._base_url = "wss://api.fish.audio/v1/tts/live"
        self._websocket = None
        self._receive_task = None
        self._request_id = None
        self._started = False

        self._settings = {
            "sample_rate": sample_rate,
            "latency": params.latency,
            "format": output_format,
            "prosody": {
                "speed": params.prosody_speed,
                "volume": params.prosody_volume,
            },
            "reference_id": model,
        }

        self.set_model_name(model)

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        self._settings["reference_id"] = model
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await self._connect_websocket()
        self._receive_task = self.get_event_loop().create_task(self._receive_task_handler())

    async def _disconnect(self):
        await self._disconnect_websocket()
        if self._receive_task:
            self._receive_task.cancel()
            await self._receive_task
            self._receive_task = None

    async def _connect_websocket(self):
        try:
            logger.debug("Connecting to Fish Audio")
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websockets.connect(self._base_url, extra_headers=headers)

            # Send initial start message with ormsgpack
            start_message = {"event": "start", "request": {"text": "", **self._settings}}
            await self._websocket.send(ormsgpack.packb(start_message))
            logger.debug("Sent start message to Fish Audio")
        except Exception as e:
            logger.error(f"Fish Audio initialization error: {e}")
            self._websocket = None

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Fish Audio")
                # Send stop event with ormsgpack
                stop_message = {"event": "stop"}
                await self._websocket.send(ormsgpack.packb(stop_message))
                await self._websocket.close()
                self._websocket = None
            self._request_id = None
            self._started = False
        except Exception as e:
            logger.error(f"Error closing websocket: {e}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                if isinstance(message, bytes):
                    msg = ormsgpack.unpackb(message)
                    if isinstance(msg, dict):
                        event = msg.get("event")
                        if event == "audio":
                            await self.stop_ttfb_metrics()
                            audio_data = msg.get("audio")
                            # Only process larger chunks to remove msgpack overhead
                            if audio_data and len(audio_data) > 1024:
                                frame = TTSAudioRawFrame(
                                    audio_data, self._settings["sample_rate"], 1
                                )
                                await self.push_frame(frame)
                                continue

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _reconnect_websocket(self, retry_state: RetryCallState):
        logger.warning(f"Fish Audio reconnecting (attempt: {retry_state.attempt_number})")
        await self._disconnect_websocket()
        await self._connect_websocket()

    async def _receive_task_handler(self):
        while True:
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=1, min=4, max=10),
                    before_sleep=self._reconnect_websocket,
                    reraise=True,
                ):
                    with attempt:
                        await self._receive_messages()
            except asyncio.CancelledError:
                break
            except Exception as e:
                message = f"Fish Audio error receiving messages: {e}"
                logger.error(message)
                await self.push_error(ErrorFrame(message, fatal=True))
                break

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSSpeakFrame):
            await self.pause_processing_frames()
        elif isinstance(frame, LLMFullResponseEndFrame) and self._request_id:
            await self.pause_processing_frames()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.resume_processing_frames()

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._request_id = None

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating Fish TTS: [{text}]")
        try:
            if not self._websocket or self._websocket.closed:
                await self._connect()

            if not self._request_id:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._request_id = str(uuid.uuid4())

            # Send the text
            text_message = {
                "event": "text",
                "text": text,
            }
            try:
                await self._get_websocket().send(ormsgpack.packb(text_message))
                await self.start_tts_usage_metrics(text)

                # Send flush event to force audio generation
                flush_message = {"event": "flush"}
                await self._get_websocket().send(ormsgpack.packb(flush_message))
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()

            yield None

        except Exception as e:
            logger.error(f"Error generating TTS: {e}")
            yield ErrorFrame(f"Error: {str(e)}")
