#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import uuid
from typing import AsyncGenerator, Literal, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import ormsgpack
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Fish Audio, you need to `pip install pipecat-ai[fish]`.")
    raise Exception(f"Missing module: {e}")

# FishAudio supports various output formats
FishAudioOutputFormat = Literal["opus", "mp3", "pcm", "wav"]


class FishAudioTTSService(InterruptibleTTSService):
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
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or FishAudioTTSService.InputParams()

        self._api_key = api_key
        self._base_url = "wss://api.fish.audio/v1/tts/live"
        self._websocket = None
        self._receive_task = None
        self._request_id = None
        self._started = False

        self._settings = {
            "sample_rate": 0,
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
        self._settings["sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.open:
                return

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
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Fish Audio")
                # Send stop event with ormsgpack
                stop_message = {"event": "stop"}
                await self._websocket.send(ormsgpack.packb(stop_message))
                await self._websocket.close()
        except Exception as e:
            logger.error(f"Error closing websocket: {e}")
        finally:
            self._request_id = None
            self._started = False
            self._websocket = None

    async def flush_audio(self):
        """Flush any buffered audio by sending a flush event to Fish Audio."""
        logger.trace(f"{self}: Flushing audio buffers")
        if not self._websocket or self._websocket.closed:
            return
        flush_message = {"event": "flush"}
        await self._get_websocket().send(ormsgpack.packb(flush_message))

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._request_id = None

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                if isinstance(message, bytes):
                    msg = ormsgpack.unpackb(message)
                    if isinstance(msg, dict):
                        event = msg.get("event")
                        if event == "audio":
                            audio_data = msg.get("audio")
                            # Only process larger chunks to remove msgpack overhead
                            if audio_data and len(audio_data) > 1024:
                                frame = TTSAudioRawFrame(audio_data, self.sample_rate, 1)
                                await self.push_frame(frame)
                                await self.stop_ttfb_metrics()
                                continue

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating Fish TTS: [{text}]")
        try:
            if not self._websocket or self._websocket.closed:
                await self._connect()

            if not self._request_id:
                await self.start_ttfb_metrics()
                await self.start_tts_usage_metrics(text)
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
