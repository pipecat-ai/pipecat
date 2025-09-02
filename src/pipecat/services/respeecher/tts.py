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
from pydantic import BaseModel, TypeAdapter, ValidationError

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
from pipecat.services.tts_service import AudioContextTTSService
from pipecat.utils.tracing.service_decorators import traced_tts

# See .env.example for Respeecher configuration needed
try:
    from respeecher.tts import ContextfulGenerationRequestParams, StreamingOutputFormatParams
    from respeecher.tts import Response as TTSResponse
    from respeecher.voices import (
        SamplingParamsParams as SamplingParams,  # TypedDict instead of a Pydantic model
    )
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

        sampling_params: SamplingParams = {}

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "public/tts/en-rt",
        url: str = "wss://api.respeecher.com/v1",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        aggregate_sentences: bool = False,
        **kwargs,
    ):
        """Initialize the Respeecher TTS service.

        Args:
            api_key: Respeecher API key for authentication.
            voice_id: ID of the voice to use for synthesis.
            model: Model path for the Respeecher TTS API.
            url: WebSocket base URL for Respeecher TTS API.
            sample_rate: Audio sample rate. If None, uses default.
            params: Additional input parameters for voice customization.
            aggregate_sentences: Whether to aggregate text into sentences client-side.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(
            push_text_frames=False,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            aggregate_sentences=aggregate_sentences,
            **kwargs,
        )

        params = params or RespeecherTTSService.InputParams()

        self._api_key = api_key
        self._url = url
        self._output_format: StreamingOutputFormatParams = {
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate or 0,
        }
        self._settings = {"sampling_params": params.sampling_params}
        self.set_model_name(model)
        self.set_voice(voice_id)

        self._context_id: str | None = None
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

    def _build_request(self, text: str, continue_transcript: bool = True):
        assert self._context_id is not None

        request: ContextfulGenerationRequestParams = {
            "transcript": text,
            "continue": continue_transcript,
            "context_id": self._context_id,
            "voice": {
                "id": self._voice_id,
                "sampling_params": self._settings["sampling_params"],
            },
            "output_format": self._output_format,
        }

        return json.dumps(request)

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
        """Cancel the Respeecher TTS service.

        Args:
            frame: The cancel frame.
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
            self._context_id = None
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
        if self._context_id:
            cancel_request = json.dumps({"context_id": self._context_id, "cancel": True})
            await self._get_websocket().send(cancel_request)
            self._context_id = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with context awareness.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, (LLMFullResponseEndFrame, EndFrame)):
            await self.flush_audio()

    async def flush_audio(self):
        """Flush any pending audio and finalize the current context."""
        if not self._context_id or not self._websocket:
            return
        logger.trace(f"{self}: flushing audio")
        flush_request = self._build_request(text="", continue_transcript=False)
        await self._websocket.send(flush_request)
        self._context_id = None

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                response = TypeAdapter(TTSResponse).validate_json(message)
            except ValidationError as e:
                logger.error(f"{self} cannot parse message: {e}")
                continue

            if response.context_id is not None and not self.audio_context_available(
                response.context_id
            ):
                logger.error(
                    f"{self} error, received {response.type} for unknown context_id: {response.context_id}"
                )
                continue

            if response.type == "error":
                logger.error(f"{self} error: {response}")
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(ErrorFrame(f"{self} error: {response.error}"))
                continue

            if response.type == "done":
                await self.stop_ttfb_metrics()
                await self.remove_audio_context(response.context_id)
            elif response.type == "chunk":
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(response.data),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                await self.append_to_audio_context(response.context_id, frame)

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

            if not self._context_id:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._context_id = str(uuid.uuid4())
                await self.create_audio_context(self._context_id)

            generation_request = self._build_request(text=text)

            try:
                await self._get_websocket().send(generation_request)
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
