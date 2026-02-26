#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
# src/pipecat/services/smallest/tts.py

"""Smallest TTS service implementation with WebSocket-based real-time voice."""

import asyncio
import base64
import json
from enum import Enum
from typing import Any, AsyncGenerator, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import AudioContextWordTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import websockets
    from websockets import ClientConnection
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install websockets`.")
    raise Exception(f"Missing module: {e}")


class SmallestTTSModel(str, Enum):
    """Supported models for the Smallest API."""

    LIGHTNING_V2 = "lightning-v2"
    LIGHTNING_V3 = "lightning-v3.1"


def language_to_smallest_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Smallest language format.

    Args:
        language: The language to convert.

    Returns:
        The Smallest-specific language code, or None if not supported.
    """
    BASE_LANGUAGES = {
        Language.AR: "ar",
        Language.BN: "bn",
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.IT: "it",
        Language.KN: "kn",
        Language.MR: "mr",
        Language.NL: "nl",
        Language.PL: "pl",
        Language.RU: "ru",
        Language.TA: "ta",
    }

    result = BASE_LANGUAGES.get(language)

    if not result:
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


def get_sample_rate(model: str, input: int | None):
    """For LIGHTNING_V3 sample rate is 44000.

    Args:
        model (str): The model name.
        input (int | None): Input Sample Rate

    Returns: int
    """
    if model == SmallestTTSModel.LIGHTNING_V3:
        return 44_000
    return input


def get_smallest_url(model: str) -> str:
    """Get the WebSocket URL for the specified Smallest model.

    Args:
        model: The model name.

    Returns:
        The WebSocket URL for the model.

    Raises:
        ValueError: If the model is invalid.
    """
    if model == SmallestTTSModel.LIGHTNING_V2:
        return "wss://waves-api.smallest.ai/api/v1/lightning-v2/get_speech/stream?timeout=120"
    if model == SmallestTTSModel.LIGHTNING_V3:
        return "wss://waves-api.smallest.ai/api/v1/lightning-v3.1/get_speech/stream?timeout=120"
    else:
        raise ValueError(f"Invalid model: {model}")


class SmallestTTSService(AudioContextWordTTSService):
    """Smallest AI Text-to-Speech service using WebSocket streaming API.

    This service implements the Smallest AI TTS API for real-time speech synthesis
    using WebSocket connections. It supports the Lightning v2 and v3.1 models.
    """

    class InputParams(BaseModel):
        """Input parameters for Smallest TTS service."""

        language: Optional[Language] = Language.EN
        speed: Optional[Union[str, float]] = 1.0
        consistency: Optional[float] = Field(default=0.5, ge=0, le=1)
        similarity: Optional[float] = Field(default=0, ge=0, le=1)
        enhancement: Optional[int] = Field(default=1, ge=0, le=2)

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        url: str = "",
        model: str = "lightning-v2",
        sample_rate: Optional[int] = 24000,
        params: InputParams = InputParams(),
        text_aggregator: Optional[BaseTextAggregator] = None,
        **kwargs: Any,
    ):
        """Initialize the Smallest TTS service.

        Args:
            api_key: Smallest AI API key.
            voice_id: Voice identifier to use for synthesis.
            url: (Optional) Custom WebSocket URL.
            model: Model to use (lightning-v2 or lightning-v3.1).
            sample_rate: Audio sample rate (8000-24000).
            params: Additional input parameters.
            text_aggregator: Text aggregation strategy.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            text_aggregator=text_aggregator or SkipTagsAggregator([("<spell>", "</spell>")]),
            reconnect_on_error=True,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url or get_smallest_url(model)
        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en",
            "speed": params.speed,
            "consistency": params.consistency,
            "similarity": params.similarity,
            "enhancement": params.enhancement,
        }
        self.set_model_name(model)
        self.set_voice(voice_id)

        self._receive_task: asyncio.Task[None] | None = None
        self._websocket: ClientConnection | None = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Smallest service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Smallest language format.

        Args:
            language: The language to convert.

        Returns:
            The Smallest-specific language code, or None if not supported.
        """
        return language_to_smallest_language(language)

    async def set_model(self, model: str):
        """Set the TTS model and reconnect.

        Args:
            model: The model name to use for synthesis.
        """
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")
        self._url = get_smallest_url(model)
        await self._disconnect()
        await self._connect()

    def _build_msg(self, text: str, context_id: str) -> dict[str, Any]:
        """Build a message payload for the Smallest API.

        Args:
            text: Text to synthesize.
            context_id: Context Id for the message.

        Returns:
            Message dictionary formatted for Smallest API.
        """
        msg: dict[str, Any] = {
            "text": text,
            "voice_id": self._voice_id,
            "language": self._settings["language"],
            "speed": self._settings["speed"],
            "consistency": self._settings["consistency"],
            "similarity": self._settings["similarity"],
            "enhancement": self._settings["enhancement"],
            "numerals": True,
            "request_id": context_id,
        }

        return msg

    async def start(self, frame: StartFrame):
        """Start the Smallest TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._sample_rate = get_sample_rate(self.model_name, frame.audio_out_sample_rate)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Smallest TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Smallest TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        """Flush any pending audio and finalize the current context."""
        context_id = self.get_active_audio_context_id()
        if not context_id or not self._websocket:
            return
        logger.debug(f"{self}: flushing audio")
        self.reset_active_audio_context()
        cancel_msg = json.dumps({"flush": True, "request_id": context_id})
        await self._websocket.send(cancel_msg)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)

    async def _connect(self):
        """Establish WebSocket connection and start receive task."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect WebSocket and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to Smallest AI WebSocket endpoint."""
        try:
            if self._websocket and self._websocket.state == websockets.State.OPEN:
                return

            logger.debug(f"Connecting to Smallest AI TTS: {self._url}")

            self._websocket = await websocket_connect(
                self._url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
            )

            await self._call_event_handler("on_connected")
            logger.debug("Connected to Smallest AI TTS")
        except Exception as e:
            logger.error(f"{self} connection error: {e}")
            self._websocket = None
            await self.push_error(error_msg=f"Connection error: {e}", exception=e)
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Disconnect from Smallest AI WebSocket."""
        try:
            await self.stop_all_metrics()

            if self._websocket and self._websocket.state != websockets.State.CLOSED:
                logger.debug("Disconnecting from Smallest AI")
                await self._websocket.close()
                logger.debug("Disconnected from Smallest AI")
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the active WebSocket connection.

        Returns:
            The WebSocket connection.

        Raises:
            Exception: If WebSocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        context_id = self.get_active_audio_context_id()
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        if context_id:
            cancel_msg = json.dumps({"flush": True, "request_id": context_id})
            await self._get_websocket().send(cancel_msg)

    async def _receive_task_handler(self, error_callback):
        """Task handler for receiving WebSocket messages with error handling.

        Args:
            error_callback: Callback function to report errors.
        """
        try:
            await self._receive_messages()
        except asyncio.CancelledError:
            # Task was cancelled, this is normal during shutdown
            logger.debug(f"{self} receive task cancelled")
        except Exception as e:
            logger.error(f"{self} receive task error: {e}")
            if error_callback:
                await error_callback(e)

    async def _receive_messages(self):
        """Handle incoming WebSocket messages from Smallest AI."""
        async for message in self._get_websocket():
            msg: dict[str, Any] = json.loads(message)
            status = msg.get("status")
            received_request_id = msg.get("request_id") or ""

            if not self.audio_context_available(received_request_id):
                logger.debug(f"Ignoring message from unavailable context: {received_request_id}")
                continue

            if status == "complete":
                # Request completed — finalize metrics and close the audio context
                # so the next context can be processed without waiting for timeout.
                logger.trace(f"Received complete for request {received_request_id}")
                await self.stop_all_metrics()
                await self.add_word_timestamps(
                    [("TTSStoppedFrame", 0), ("Reset", 0)], received_request_id
                )
                await self.remove_audio_context(received_request_id)

            elif status == "chunk":
                audio_data = base64.b64decode(msg["data"]["audio"])

                await self.stop_ttfb_metrics()
                await self.start_word_timestamps()
                frame = TTSAudioRawFrame(
                    audio=audio_data,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=received_request_id,
                )
                await self.append_to_audio_context(received_request_id, frame)
            elif status == "error":
                logger.error(f"{self} API error: {msg}")
                await self.stop_all_metrics()
                error_msg = msg.get("error", "Unknown error")
                await self.push_frame(ErrorFrame(f"{self} error: {error_msg}"))
                await self.push_frame(TTSStoppedFrame(context_id=received_request_id))
                await self.remove_audio_context(received_request_id)

            else:
                logger.warning(f"{self} unknown message type: {msg}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Smallest AI streaming WebSocket API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID provided by the pipecat framework.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            # Ensure WebSocket is connected
            if not self._websocket or self._websocket.state == websockets.State.CLOSED:
                await self._connect()

            try:
                if not self.has_active_audio_context():
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame(context_id=context_id)
                    await self.create_audio_context(context_id)

                yield TTSTextFrame(text, aggregated_by="sentence", context_id=context_id)

                # Send text to synthesize
                msg = self._build_msg(text, context_id)
                await self._get_websocket().send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)

            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                yield ErrorFrame(error=f"Error sending message: {e}")
                return
            yield None

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {str(e)}")
