#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smallest AI text-to-speech service implementations.

This module provides WebSocket-based and HTTP-based integrations with Smallest
AI's Waves API for real-time text-to-speech synthesis.
"""

import base64
import json
from enum import Enum
from typing import AsyncGenerator, Optional, Union

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
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")


class SmallestTTSModel(str, Enum):
    """Available Smallest AI TTS models."""

    LIGHTNING_V2 = "lightning-v2"
    LIGHTNING_V3_1 = "lightning-v3.1"


def language_to_smallest_tts_language(language: Language) -> Optional[str]:
    """Convert a Language enum to a Smallest TTS language string.

    Args:
        language: The Language enum value to convert.

    Returns:
        The Smallest language code string, or None if unsupported.
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


class SmallestTTSService(InterruptibleTTSService):
    """Smallest AI real-time text-to-speech service using WebSocket streaming.

    Provides real-time text-to-speech synthesis using Smallest AI's WebSocket API.
    Supports streaming audio generation with configurable voice parameters and
    language settings. Handles interruptions by reconnecting the WebSocket.

    Example::

        tts = SmallestTTSService(
            api_key="your-api-key",
            voice_id="sophia",
            params=SmallestTTSService.InputParams(
                language=Language.EN,
                speed=1.0,
            ),
        )
    """

    class InputParams(BaseModel):
        """Configuration parameters for Smallest TTS service.

        Parameters:
            language: Language for synthesis. Defaults to English.
            speed: Speech speed multiplier. Defaults to 1.0.
            consistency: Consistency level for voice generation (0-1). Defaults to 0.5.
            similarity: Similarity level for voice generation (0-1). Defaults to 0.
            enhancement: Enhancement level for voice generation (0-2). Defaults to 1.
        """

        language: Optional[Language] = Language.EN
        speed: Optional[Union[str, float]] = 1.0
        consistency: Optional[float] = Field(default=0.5, ge=0, le=1)
        similarity: Optional[float] = Field(default=0, ge=0, le=1)
        enhancement: Optional[int] = Field(default=1, ge=0, le=2)

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "sophia",
        base_url: str = "wss://waves-api.smallest.ai",
        model: str = SmallestTTSModel.LIGHTNING_V3_1,
        sample_rate: Optional[int] = 24000,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Smallest AI WebSocket TTS service.

        Args:
            api_key: Smallest AI API key for authentication.
            voice_id: Voice identifier for synthesis.
            base_url: Base WebSocket URL for the Smallest API.
            model: TTS model to use. Defaults to "lightning-v3.1".
            sample_rate: Audio sample rate in Hz. Defaults to 24000.
            params: Configuration parameters for the TTS service.
            **kwargs: Additional arguments passed to parent InterruptibleTTSService.
        """
        params = params or SmallestTTSService.InputParams()
        model_str = model.value if isinstance(model, Enum) else model
        lang_str = (
            language_to_smallest_tts_language(params.language) if params.language else "en"
        )

        super().__init__(
            aggregate_sentences=True,
            push_text_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=TTSSettings(model=model_str, voice=voice_id, language=lang_str),
            **kwargs,
        )

        self._api_key = api_key
        self._websocket_url = f"{base_url}/api/v1/{model_str}/get_speech/stream"

        self._tts_params = {
            "language": lang_str,
            "speed": params.speed,
            "consistency": params.consistency,
            "similarity": params.similarity,
            "enhancement": params.enhancement,
        }

        self._receive_task = None
        self._context_id: Optional[str] = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Smallest service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Smallest service language format.

        Args:
            language: The language to convert.

        Returns:
            The Smallest-specific language code, or None if not supported.
        """
        return language_to_smallest_tts_language(language)

    def _build_msg(self, text: str) -> dict:
        """Build a WebSocket message for the Smallest API.

        Args:
            text: The text to synthesize.

        Returns:
            Dictionary with the API message payload.
        """
        msg = {
            "text": text,
            "voice_id": self._settings.voice,
            "language": self._tts_params["language"],
            "speed": self._tts_params["speed"],
            "consistency": self._tts_params["consistency"],
            "similarity": self._tts_params["similarity"],
            "enhancement": self._tts_params["enhancement"],
        }

        if self._context_id:
            msg["request_id"] = self._context_id

        return msg

    async def start(self, frame: StartFrame):
        """Start the Smallest TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
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

    async def _connect(self):
        """Connect to Smallest WebSocket and start receive task."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from Smallest WebSocket and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to the Smallest API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Smallest")

            self._websocket = await websocket_connect(
                self._websocket_url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
            )

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Smallest connection error: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Smallest")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._context_id = None
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the WebSocket connection if available.

        Returns:
            The active WebSocket connection.

        Raises:
            Exception: If no WebSocket connection is available.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle an interruption by resetting state.

        Args:
            frame: The interruption frame.
            direction: The direction of frame processing.
        """
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._context_id = None

    async def _receive_messages(self):
        """Receive and process messages from the Smallest WebSocket API."""
        async for message in self._get_websocket():
            msg = json.loads(message)
            status = msg.get("status")

            if status == "complete":
                msg_request_id = msg.get("request_id")
                if self._context_id and msg_request_id and msg_request_id == self._context_id:
                    await self.stop_all_metrics()
                    await self.push_frame(TTSStoppedFrame(context_id=self._context_id))
                    self._context_id = None
            elif status == "chunk":
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(msg["data"]["audio"]),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=self._context_id,
                )
                await self.push_frame(frame)
            elif status == "error":
                logger.error(f"{self} error: {msg}")
                await self.push_frame(TTSStoppedFrame(context_id=self._context_id))
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Smallest TTS error: {msg.get('error', msg)}")
                self._context_id = None
            else:
                logger.warning(f"{self} unknown message status: {msg}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Smallest's WebSocket streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: TTSStartedFrame to signal start; audio arrives via WebSocket.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                await self.start_ttfb_metrics()
                self._context_id = context_id
                yield TTSStartedFrame(context_id=context_id)

                msg = self._build_msg(text=text)
                await self._get_websocket().send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield ErrorFrame(error=f"Smallest TTS send error: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Smallest TTS error: {e}")


class SmallestHttpTTSService(TTSService):
    """Smallest AI text-to-speech service using the HTTP API.

    Provides text-to-speech synthesis using Smallest AI's HTTP REST API.
    Suitable for applications that prefer simpler HTTP-based communication
    over WebSocket connections.

    Example::

        tts = SmallestHttpTTSService(
            api_key="your-api-key",
            voice_id="anushka",
            params=SmallestHttpTTSService.InputParams(
                language=Language.HI,
                speed=1.2,
            ),
        )
    """

    class InputParams(BaseModel):
        """Configuration parameters for Smallest HTTP TTS service.

        Parameters:
            language: Language code for synthesis. Defaults to "en".
            speed: Speech speed multiplier. Defaults to 1.0.
            consistency: Consistency level for voice generation.
            similarity: Similarity level for voice generation.
            enhancement: Enhancement level for voice generation.
        """

        language: str = "en"
        speed: float = 1.0
        consistency: Optional[float] = None
        similarity: Optional[float] = None
        enhancement: Optional[float] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "sophia",
        model: str = SmallestTTSModel.LIGHTNING_V3_1,
        base_url: str = "https://waves-api.smallest.ai",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Smallest AI HTTP TTS service.

        Args:
            api_key: Smallest AI API key for authentication.
            voice_id: Voice identifier for synthesis.
            model: TTS model to use. Defaults to "lightning-v3.1".
            base_url: Base URL for the Smallest API.
            sample_rate: Audio sample rate in Hz.
            params: Configuration parameters for the TTS service.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        params = params or SmallestHttpTTSService.InputParams()
        model_str = model.value if isinstance(model, Enum) else model

        super().__init__(
            sample_rate=sample_rate,
            settings=TTSSettings(model=model_str, voice=voice_id, language=params.language),
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_url = f"{self._base_url}/api/v1/{model_str}/get_speech"

        self._tts_params = {
            "language": params.language,
            "speed": params.speed,
            "consistency": params.consistency,
            "similarity": params.similarity,
            "enhancement": params.enhancement,
        }

        self._session = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Smallest HTTP service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Smallest HTTP TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        try:
            import aiohttp

            self._session = aiohttp.ClientSession()
        except ModuleNotFoundError as e:
            logger.error(f"Exception: {e}")
            logger.error("In order to use Smallest HTTP TTS, you need to `pip install aiohttp`.")
            raise Exception(f"Missing module: {e}")

    async def stop(self, frame: EndFrame):
        """Stop the Smallest HTTP TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the Smallest HTTP TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        if self._session:
            await self._session.close()
            self._session = None

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using the Smallest HTTP API.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: TTSStartedFrame, TTSAudioRawFrame chunks, and TTSStoppedFrame.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        if not self._session:
            yield ErrorFrame(error="Smallest HTTP TTS session not initialized")
            return

        try:
            await self.start_ttfb_metrics()

            payload = {
                "voice_id": self._settings.voice,
                "text": text,
                "sample_rate": self.sample_rate,
            }

            for key, value in self._tts_params.items():
                if value is not None:
                    payload[key] = value

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            yield TTSStartedFrame(context_id=context_id)

            async with self._session.post(
                self._model_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"{self} API error: {error_text}")
                    yield ErrorFrame(error=f"Smallest API error: {error_text}")
                    return

                result = await response.read()

            await self.stop_ttfb_metrics()
            await self.start_tts_usage_metrics(text)

            yield TTSAudioRawFrame(
                audio=result,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Smallest TTS error: {e}")
        finally:
            yield TTSStoppedFrame(context_id=context_id)
