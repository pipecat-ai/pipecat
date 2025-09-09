#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Neuphonic text-to-speech service implementations.

This module provides WebSocket and HTTP-based integrations with Neuphonic's
text-to-speech API for real-time audio synthesis.
"""

import asyncio
import base64
import json
from typing import Any, AsyncGenerator, Mapping, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

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
from pipecat.services.tts_service import InterruptibleTTSService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Neuphonic, you need to `pip install pipecat-ai[neuphonic]`.")
    raise Exception(f"Missing module: {e}")


def language_to_neuphonic_lang_code(language: Language) -> Optional[str]:
    """Convert a Language enum to Neuphonic language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Neuphonic language code, or None if not supported.
    """
    BASE_LANGUAGES = {
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.NL: "nl",
        Language.AR: "ar",
        Language.FR: "fr",
        Language.PT: "pt",
        Language.RU: "ru",
        Language.HI: "HI",
        Language.ZH: "zh",
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        # Convert enum value to string and get the base language part (e.g. es-ES -> es)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Look up the base code in our supported languages
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


class NeuphonicTTSService(InterruptibleTTSService):
    """Neuphonic real-time text-to-speech service using WebSocket streaming.

    Provides real-time text-to-speech synthesis using Neuphonic's WebSocket API.
    Supports interruption handling, keepalive connections, and configurable voice
    parameters for high-quality speech generation.
    """

    class InputParams(BaseModel):
        """Input parameters for Neuphonic TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English.
            speed: Speech speed multiplier. Defaults to 1.0.
        """

        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: Optional[str] = None,
        url: str = "wss://api.neuphonic.com",
        sample_rate: Optional[int] = 22050,
        encoding: str = "pcm_linear",
        params: Optional[InputParams] = None,
        aggregate_sentences: Optional[bool] = True,
        **kwargs,
    ):
        """Initialize the Neuphonic TTS service.

        Args:
            api_key: Neuphonic API key for authentication.
            voice_id: ID of the voice to use for synthesis.
            url: WebSocket URL for the Neuphonic API.
            sample_rate: Audio sample rate in Hz. Defaults to 22050.
            encoding: Audio encoding format. Defaults to "pcm_linear".
            params: Additional input parameters for TTS configuration.
            aggregate_sentences: Whether to aggregate sentences within the TTSService.
            **kwargs: Additional arguments passed to parent InterruptibleTTSService.
        """
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,
            push_stop_frames=True,
            stop_frame_timeout_s=2.0,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or NeuphonicTTSService.InputParams()

        self._api_key = api_key
        self._url = url
        self._settings = {
            "lang_code": self.language_to_service_language(params.language),
            "speed": params.speed,
            "encoding": encoding,
            "sampling_rate": sample_rate,
        }
        self.set_voice(voice_id)

        # Indicates if we have sent TTSStartedFrame. It will reset to False when
        # there's an interruption or TTSStoppedFrame.
        self._started = False
        self._cumulative_time = 0

        self._receive_task = None
        self._keepalive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Neuphonic service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Neuphonic service language format.

        Args:
            language: The language to convert.

        Returns:
            The Neuphonic-specific language code, or None if not supported.
        """
        return language_to_neuphonic_lang_code(language)

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Update service settings and reconnect with new configuration."""
        if "voice_id" in settings:
            self.set_voice(settings["voice_id"])

        await super()._update_settings(settings)
        await self._disconnect()
        await self._connect()
        logger.info(f"Switching TTS to settings: [{self._settings}]")

    async def start(self, frame: StartFrame):
        """Start the Neuphonic TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Neuphonic TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Neuphonic TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        """Flush any pending audio synthesis by sending stop command."""
        if self._websocket:
            msg = {"text": "<STOP>"}
            await self._websocket.send(json.dumps(msg))

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame downstream with special handling for stop conditions.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):
            self._started = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with special handling for speech control.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        # If we received a TTSSpeakFrame and the LLM response included text (it
        # might be that it's only a function calling response) we pause
        # processing more frames until we receive a BotStoppedSpeakingFrame.
        if isinstance(frame, TTSSpeakFrame):
            await self.pause_processing_frames()
        elif isinstance(frame, LLMFullResponseEndFrame) and self._started:
            await self.pause_processing_frames()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.resume_processing_frames()

    async def _connect(self):
        """Connect to Neuphonic WebSocket and start background tasks."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        """Disconnect from Neuphonic WebSocket and clean up tasks."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to Neuphonic API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Neuphonic")

            tts_config = {
                **self._settings,
                "voice_id": self._voice_id,
            }

            query_params = []
            for key, value in tts_config.items():
                if value is not None:
                    query_params.append(f"{key}={value}")

            url = f"{self._url}/speak/{self._settings['lang_code']}"
            if query_params:
                url += f"?{'&'.join(query_params)}"

            headers = {"x-api-key": self._api_key}

            self._websocket = await websocket_connect(url, additional_headers=headers)
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Neuphonic")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._started = False
            self._websocket = None

    async def _receive_messages(self):
        """Receive and process messages from Neuphonic WebSocket."""
        async for message in self._websocket:
            if isinstance(message, str):
                msg = json.loads(message)
                if msg.get("data") and msg["data"].get("audio"):
                    await self.stop_ttfb_metrics()

                    audio = base64.b64decode(msg["data"]["audio"])
                    frame = TTSAudioRawFrame(audio, self.sample_rate, 1)
                    await self.push_frame(frame)

    async def _keepalive_task_handler(self):
        """Handle keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 10
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            await self._send_keepalive()

    async def _send_keepalive(self):
        """Send keepalive message to maintain connection."""
        if self._websocket:
            # Send empty text for keepalive
            msg = {"text": ""}
            await self._websocket.send(json.dumps(msg))

    async def _send_text(self, text: str):
        """Send text to Neuphonic WebSocket for synthesis."""
        if self._websocket:
            msg = {"text": f"{text} <STOP>"}
            logger.debug(f"Sending text to websocket: {msg}")
            await self._websocket.send(json.dumps(msg))

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Neuphonic's streaming API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"Generating TTS: [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True
                    self._cumulative_time = 0

                await self._send_text(text)
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


class NeuphonicHttpTTSService(TTSService):
    """Neuphonic text-to-speech service using HTTP streaming.

    Provides text-to-speech synthesis using Neuphonic's HTTP API with server-sent
    events for streaming audio delivery. Suitable for applications that prefer
    HTTP-based communication over WebSocket connections.
    """

    class InputParams(BaseModel):
        """Input parameters for Neuphonic HTTP TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English.
            speed: Speech speed multiplier. Defaults to 1.0.
        """

        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: Optional[str] = None,
        aiohttp_session: aiohttp.ClientSession,
        url: str = "https://api.neuphonic.com",
        sample_rate: Optional[int] = 22050,
        encoding: Optional[str] = "pcm_linear",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Neuphonic HTTP TTS service.

        Args:
            api_key: Neuphonic API key for authentication.
            voice_id: ID of the voice to use for synthesis.
            aiohttp_session: Shared aiohttp session for HTTP requests.
            url: Base URL for the Neuphonic HTTP API.
            sample_rate: Audio sample rate in Hz. Defaults to 22050.
            encoding: Audio encoding format. Defaults to "pcm_linear".
            params: Additional input parameters for TTS configuration.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or NeuphonicHttpTTSService.InputParams()

        self._api_key = api_key
        self._session = aiohttp_session
        self._base_url = url.rstrip("/")
        self._lang_code = self.language_to_service_language(params.language) or "en"
        self._speed = params.speed
        self._encoding = encoding
        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Neuphonic HTTP service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Neuphonic service language format.

        Args:
            language: The language to convert.

        Returns:
            The Neuphonic-specific language code, or None if not supported.
        """
        return language_to_neuphonic_lang_code(language)

    async def start(self, frame: StartFrame):
        """Start the Neuphonic HTTP TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

    async def flush_audio(self):
        """Flush any pending audio synthesis.

        Note:
            HTTP-based service doesn't require explicit flushing.
        """
        pass

    def _parse_sse_message(self, message: str) -> dict | None:
        """Parse a Server-Sent Event message.

        Args:
            message: The SSE message to parse.

        Returns:
            Parsed message dictionary or None if not a data message.
        """
        message = message.strip()

        if not message or "data" not in message:
            return None

        try:
            # Split on ": " and take the part after "data: "
            _, data_content = message.split(": ", 1)

            if not data_content or data_content == "[DONE]":
                return None

            message_dict = json.loads(data_content)

            # Check for errors in the response
            if message_dict.get("errors") is not None:
                raise Exception(
                    f"Neuphonic API error {message_dict.get('status_code', 'unknown')}: {message_dict['errors']}"
                )

            return message_dict
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to parse SSE message: {e}")
            return None

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Neuphonic streaming API.

        Args:
            text: The text to convert to speech.

        Yields:
            Frame: Audio frames containing the synthesized speech and status information.
        """
        logger.debug(f"Generating TTS: [{text}]")

        url = f"{self._base_url}/sse/speak/{self._lang_code}"

        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "text": text,
            "lang_code": self._lang_code,
            "encoding": self._encoding,
            "sampling_rate": self.sample_rate,
            "speed": self._speed,
        }

        if self._voice_id:
            payload["voice_id"] = self._voice_id

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    error_message = f"Neuphonic API error: HTTP {response.status} - {error_text}"
                    logger.error(error_message)
                    yield ErrorFrame(error=error_message)
                    return

                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()

                # Process SSE stream line by line
                async for line in response.content:
                    if not line:
                        continue

                    message = line.decode("utf-8", errors="ignore")
                    if not message.strip():
                        continue

                    try:
                        parsed_message = self._parse_sse_message(message)

                        if (
                            parsed_message is not None
                            and parsed_message.get("data", {}).get("audio") is not None
                        ):
                            audio_b64 = parsed_message["data"]["audio"]
                            audio_bytes = base64.b64decode(audio_b64)

                            await self.stop_ttfb_metrics()
                            yield TTSAudioRawFrame(audio_bytes, self.sample_rate, 1)

                    except Exception as e:
                        logger.error(f"Error processing SSE message: {e}")
                        # Don't yield error frame for individual message failures
                        continue

        except asyncio.CancelledError:
            logger.debug("TTS generation cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error in run_tts: {e}")
            yield ErrorFrame(error=f"Neuphonic TTS error: {str(e)}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
