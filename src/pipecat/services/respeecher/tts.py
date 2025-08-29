#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Respeecher Space text-to-speech service implementation."""

import base64
import json
import uuid
from typing import AsyncGenerator, Optional

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
from pipecat.services.tts_service import AudioContextTTSService
from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator
from pipecat.utils.tracing.service_decorators import traced_tts

# See .env.example for Respeecher configuration needed
try:
    from respeecher import SamplingParams
    from respeecher.tts import StreamingEncoding
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Respeecher, you need to `pip install pipecat-ai[respeecher]`.")
    raise Exception(f"Missing module: {e}")


class RespeecherTTSService(AudioContextTTSService):
    """Respeecher Space TTS service with WebSocket streaming and audio contexts.

    Provides text-to-speech using Respeecher's streaming WebSocket API.
    Supports audio context management and voice customization via sampling parameters.
    """

    class InputParams(BaseModel):
        """Input parameters for Respeecher TTS configuration.

        Parameters:
            sampling_params: Sampling parameters used for speech synthesis.
        """

        sampling_params: SamplingParams = SamplingParams()

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "public/tts/en-rt",
        url: str = "wss://api.respeecher.com/v1",
        sample_rate: Optional[int] = None,
        encoding: StreamingEncoding = "pcm_s16le",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Respeecher TTS service.

        Args:
            api_key: Respeecher API key for authentication.
            voice_id: ID of the voice to use for synthesis.
            model: Model path for the Respeecher TTS API.
            url: WebSocket base URL for Respeecher TTS API.
            sample_rate: Audio sample rate. If None, uses default.
            encoding: Audio encoding format.
            params: Additional input parameters for voice customization.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(
            push_text_frames=False,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        self._params = params or RespeecherTTSService.InputParams()
        self._api_key = api_key
        self._url = url
        self._output_format = {
            "encoding": encoding,
            "sample_rate": sample_rate,
        }
        self.set_model_name(model)
        self.set_voice(voice_id)

        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True
        """
        return True

    async def set_model(self, model: str):
        """Set the TTS model.

        Args:
            model: The model name to use for synthesis.
        """
        self._model_id = model
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")
        await self._disconnect()
        await self._connect()

    def _build_msg(self, text: str, context_id: str):
        msg = {
            "transcript": text,
            "context_id": context_id,
            "voice": {
                "id": self._voice_id,
                "sampling_params": self._params.sampling_params.model_dump(exclude_none=True),
            },
            "output_format": self._output_format,
        }

        return json.dumps(msg)

    async def start(self, frame: StartFrame):
        """Start the Respeecher TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._output_format["sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Respeecher TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Stop the Respeecher TTS service.

        Args:
            frame: The end frame.
        """
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
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Respeecher")
            self._websocket = await websocket_connect(
                f"{self._url}/{self._model_name}/tts/websocket?api_key={self._api_key}"
            )
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Respeecher")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._websocket = None

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        for context_id in self._contexts:
            cancel_msg = json.dumps({"context_id": context_id, "cancel": True})
            await self._get_websocket().send(cancel_msg)
            await self.remove_audio_context(context_id)

    async def _receive_messages(self):
        async for message in WatchdogAsyncIterator(
            self._get_websocket(), manager=self.task_manager
        ):
            msg = json.loads(message)
            if not msg or not self.audio_context_available(msg["context_id"]):
                continue
            if msg["type"] == "done":
                await self.stop_ttfb_metrics()
                await self.remove_audio_context(msg["context_id"])
            elif msg["type"] == "chunk":
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(msg["data"]),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                await self.append_to_audio_context(msg["context_id"], frame)
            elif msg["type"] == "error":
                logger.error(f"{self} error: {msg}")
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(ErrorFrame(f"{self} error: {msg['error']}"))
            else:
                logger.error(f"{self} error, unknown message type: {msg}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Respeecher's streaming API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if not self._contexts:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()

            context_id = str(uuid.uuid4())
            await self.create_audio_context(context_id)
            msg = self._build_msg(text=text, context_id=context_id)

            try:
                await self._get_websocket().send(msg)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
