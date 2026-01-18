# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Gradium Text-to-Speech service implementation."""

import base64
import json
from typing import Any, AsyncGenerator, Mapping, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import InterruptibleWordTTSService
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets import ConnectionClosedOK
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Gradium, you need to `pip install pipecat-ai[gradium]`.")
    raise Exception(f"Missing module: {e}")

SAMPLE_RATE = 48000


class GradiumTTSService(InterruptibleWordTTSService):
    """Text-to-Speech service using Gradium's websocket API."""

    class InputParams(BaseModel):
        """Configuration parameters for Gradium TTS service.

        Parameters:
            temp: Temperature to be used for generation, defaults to 0.6.
        """

        temp: Optional[float] = 0.6

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "YTpq7expH9539ERJ",
        url: str = "wss://eu.api.gradium.ai/api/speech/tts",
        model: str = "default",
        json_config: Optional[str] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Gradium TTS service.

        Args:
            api_key: Gradium API key for authentication.
            voice_id: the voice identifier.
            url: Gradium websocket API endpoint.
            model: Model ID to use for synthesis.
            json_config: Optional JSON configuration string for additional model settings.
            params: Additional configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        # Initialize with parent class settings for proper frame handling
        super().__init__(
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=SAMPLE_RATE,
            **kwargs,
        )

        params = params or GradiumTTSService.InputParams()

        # Store service configuration
        self._api_key = api_key
        self._url = url
        self._voice_id = voice_id
        self._json_config = json_config
        self._model = model
        self._settings = {
            "voice_id": voice_id,
            "model_name": model,
            "output_format": "pcm",
        }

        # State tracking
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Gradium service supports metrics generation.
        """
        return True

    async def set_model(self, model: str):
        """Update the TTS model.

        Args:
            model: The model name to use for synthesis.
        """
        self._model = model
        await super().set_model(model)

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Update service settings and reconnect if voice changed."""
        prev_voice = self._voice_id
        await super()._update_settings(settings)
        if not prev_voice == self._voice_id:
            self._settings["voice_id"] = self._voice_id
            logger.info(f"Switching TTS voice to: [{self._voice_id}]")
            await self._disconnect()
            await self._connect()

    def _build_msg(self, text: str = "") -> dict:
        """Build JSON message for Gradium API."""
        return {"text": text, "type": "text"}

    async def start(self, frame: StartFrame):
        """Start the service and establish websocket connection.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close connection.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel current operation and clean up.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Establish websocket connection and start receive task."""
        await super()._connect()

        logger.debug(f"{self}: connecting")

        # If the server disconnected, cancel the receive-task so that it can be reset below.
        if self._websocket is None or self._websocket.state is not State.OPEN:
            if self._receive_task:
                await self.cancel_task(self._receive_task)
                self._receive_task = None

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            logger.debug(f"{self}: setting receive task")
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close websocket connection and clean up tasks."""
        await super()._disconnect()

        logger.debug(f"{self}: disconnecting")
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to Gradium websocket API with configured settings."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            headers = {"x-api-key": self._api_key, "x-api-source": "pipecat"}
            self._websocket = await websocket_connect(self._url, additional_headers=headers)

            setup_msg = {
                "type": "setup",
                "output_format": "pcm",
                "voice_id": self._voice_id,
            }
            if self._json_config is not None:
                setup_msg["json_config"] = self._json_config
            await self._websocket.send(json.dumps(setup_msg))
            ready_msg = await self._websocket.recv()
            ready_msg = json.loads(ready_msg)
            if ready_msg["type"] == "error":
                raise Exception(f"received error {ready_msg['message']}")
            if ready_msg["type"] != "ready":
                raise Exception(f"unexpected first message type {ready_msg['type']}")

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close websocket connection and reset state."""
        try:
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get active websocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def flush_audio(self):
        """Flush any pending audio synthesis."""
        if not self._websocket:
            return
        try:
            msg = {"type": "end_of_stream"}
            await self._websocket.send(json.dumps(msg))
        except ConnectionClosedOK:
            logger.debug(f"{self}: connection closed normally during flush")
        except Exception as e:
            logger.error(f"{self} exception: {e}")

    async def _receive_messages(self):
        """Process incoming websocket messages."""
        # TODO(laurent): This should not be necessary as it should happen when
        # receiving the messages but this does not seem to always be the case
        # and that may lead to a busy polling loop.
        if self._websocket and self._websocket.state is State.CLOSED:
            raise ConnectionClosedOK(None, None)
        async for message in self._get_websocket():
            msg = json.loads(message)

            if msg["type"] == "audio":
                # Process audio chunk
                await self.stop_ttfb_metrics()
                await self.start_word_timestamps()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(msg["audio"]),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                await self.push_frame(frame)

            elif msg["type"] == "text":
                await self.add_word_timestamps([(msg["text"], msg["start_s"])])
            elif msg["type"] == "end_of_stream":
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()

            elif msg["type"] == "error":
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Error: {msg['message']}")

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push frame and handle end-of-turn conditions.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Gradium's streaming API.

        Args:
            text: The text to convert to speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        _state = self._websocket.state if self._websocket is not None else None
        logger.debug(f"{self}: Generating TTS [{text}] {_state}")
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                self._websocket = None
                await self._connect()

            try:
                yield TTSStartedFrame()

                msg = self._build_msg(text=text)
                await self._get_websocket().send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
