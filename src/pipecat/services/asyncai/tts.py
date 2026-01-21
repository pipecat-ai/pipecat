#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Async text-to-speech service implementations."""

import asyncio
import base64
import json
import uuid
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

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
from pipecat.services.tts_service import AudioContextTTSService, TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import websockets
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Async, you need to `pip install pipecat-ai[asyncai]`.")
    raise Exception(f"Missing module: {e}")


def language_to_async_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Async language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Async language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.EN: "en",
        Language.FR: "fr",
        Language.ES: "es",
        Language.DE: "de",
        Language.IT: "it",
        Language.PT: "pt",
        Language.NL: "nl",
        Language.AR: "ar",
        Language.RU: "ru",
        Language.RO: "ro",
        Language.JA: "ja",
        Language.HE: "he",
        Language.HY: "hy",
        Language.TR: "tr",
        Language.HI: "hi",
        Language.ZH: "zh",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


class AsyncAITTSService(AudioContextTTSService):
    """Async TTS service with WebSocket streaming.

    Provides text-to-speech using Async's streaming WebSocket API.
    """

    class InputParams(BaseModel):
        """Input parameters for Async TTS configuration.

        Parameters:
            language: Language to use for synthesis.
        """

        language: Optional[Language] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        version: str = "v1",
        url: str = "wss://api.async.ai/text_to_speech/websocket/ws",
        model: str = "asyncflow_multilingual_v1.0",
        sample_rate: Optional[int] = None,
        encoding: str = "pcm_s16le",
        container: str = "raw",
        params: Optional[InputParams] = None,
        aggregate_sentences: Optional[bool] = True,
        **kwargs,
    ):
        """Initialize the Async TTS service.

        Args:
            api_key: Async API key.
            voice_id: UUID of the voice to use for synthesis. See docs for a full list:
                https://docs.async.ai/list-voices-16699698e0
            version: Async API version.
            url: WebSocket URL for Async TTS API.
            model: TTS model to use (e.g., "asyncflow_multilingual_v1.0").
            sample_rate: Audio sample rate.
            encoding: Audio encoding format.
            container: Audio container format.
            params: Additional input parameters for voice customization.
            aggregate_sentences: Whether to aggregate sentences within the TTSService.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            pause_frame_processing=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or AsyncAITTSService.InputParams()

        self._api_key = api_key
        self._api_version = version
        self._url = url
        self._settings = {
            "output_format": {
                "container": container,
                "encoding": encoding,
                "sample_rate": 0,
            },
            "language": self.language_to_service_language(params.language)
            if params.language
            else None,
        }

        self.set_model_name(model)
        self.set_voice(voice_id)

        self._receive_task = None
        self._keepalive_task = None
        self._started = False
        self._context_id = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Async service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Async language format.

        Args:
            language: The language to convert.

        Returns:
            The Async-specific language code, or None if not supported.
        """
        return language_to_async_language(language)

    def _build_msg(self, text: str = "", context_id: str = "", force: bool = False) -> str:
        msg = {"transcript": text, "context_id": context_id, "force": force}
        return json.dumps(msg)

    async def start(self, frame: StartFrame):
        """Start the Async TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["output_format"]["sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Async TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Async TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Async")
            self._websocket = await websocket_connect(
                f"{self._url}?api_key={self._api_key}&version={self._api_version}"
            )
            init_msg = {
                "model_id": self._model_name,
                "voice": {"mode": "id", "id": self._voice_id},
                "output_format": self._settings["output_format"],
                "language": self._settings["language"],
            }

            await self._get_websocket().send(json.dumps(init_msg))

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Async")
                # Close all contexts and the socket
                if self._context_id:
                    await self._websocket.send(json.dumps({"terminate": True}))
                await self._websocket.close()
                logger.debug("Disconnected from Async")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            self._context_id = None
            self._started = False
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def flush_audio(self):
        """Flush any pending audio."""
        if not self._context_id or not self._websocket:
            return
        logger.trace(f"{self}: flushing audio")
        msg = self._build_msg(text=" ", context_id=self._context_id, force=True)
        await self._websocket.send(msg)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame downstream with special handling for stop conditions.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            self._started = False

    async def _receive_messages(self):
        async for message in self._get_websocket():
            msg = json.loads(message)
            if not msg:
                continue

            received_ctx_id = msg.get("context_id")
            # Handle final messages first, regardless of context availability
            # At the moment, this message is received AFTER the close_context message is
            # sent, so it doesn't serve any functional purpose. For now, we'll just log it.
            if msg.get("final") is True:
                logger.trace(f"Received final message for context {received_ctx_id}")
                continue

            # Check if this message belongs to the current context.
            if not self.audio_context_available(received_ctx_id):
                if self._context_id == received_ctx_id:
                    logger.debug(
                        f"Received a delayed message, recreating the context: {self._context_id}"
                    )
                    await self.create_audio_context(self._context_id)
                else:
                    # This can happen if a message is received _after_ we have closed a context
                    # due to user interruption but _before_ the `isFinal` message for the context
                    # is received.
                    logger.debug(f"Ignoring message from unavailable context: {received_ctx_id}")
                    continue

            if msg.get("audio"):
                await self.stop_ttfb_metrics()
                audio = base64.b64decode(msg["audio"])
                frame = TTSAudioRawFrame(audio, self.sample_rate, 1)
                await self.append_to_audio_context(received_ctx_id, frame)

    async def _keepalive_task_handler(self):
        """Send periodic keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 10
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            try:
                if self._websocket and self._websocket.state is State.OPEN:
                    if self._context_id:
                        keepalive_message = {
                            "transcript": " ",
                            "context_id": self._context_id,
                        }
                        logger.trace("Sending keepalive message")
                    else:
                        # It's possible to have a user interruption which clears the context
                        # without generating a new TTS response. In this case, we'll just send
                        # an empty message to keep the connection alive.
                        keepalive_message = {"transcript": " "}
                        logger.trace("Sending keepalive without context")
                    await self._websocket.send(json.dumps(keepalive_message))
            except websockets.ConnectionClosed as e:
                logger.warning(f"{self} keepalive error: {e}")
                break

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """Handle interruption by closing the current context."""
        await super()._handle_interruption(frame, direction)

        # Close the current context when interrupted without closing the websocket
        if self._context_id and self._websocket:
            try:
                await self._websocket.send(
                    json.dumps(
                        {"context_id": self._context_id, "close_context": True, "transcript": ""}
                    )
                )
            except Exception as e:
                logger.error(f"Error closing context on interruption: {e}")
            self._context_id = None
            self._started = False

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Async API websocket endpoint.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True

                    if not self._context_id:
                        self._context_id = str(uuid.uuid4())
                    if not self.audio_context_available(self._context_id):
                        await self.create_audio_context(self._context_id)

                    msg = self._build_msg(text=text, force=True, context_id=self._context_id)
                    await self._get_websocket().send(msg)
                    await self.start_tts_usage_metrics(text)
                else:
                    if self._websocket and self._context_id:
                        msg = self._build_msg(text=text, force=True, context_id=self._context_id)
                        await self._get_websocket().send(msg)

            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame()
                self._started = False
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


class AsyncAIHttpTTSService(TTSService):
    """HTTP-based Async TTS service.

    Provides text-to-speech using Async's HTTP streaming API for simpler,
    non-WebSocket integration. Suitable for use cases where streaming WebSocket
    connection is not required or desired.
    """

    class InputParams(BaseModel):
        """Input parameters for Async API.

        Parameters:
            language: Language to use for synthesis.
        """

        language: Optional[Language] = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "asyncflow_multilingual_v1.0",
        url: str = "https://api.async.ai",
        version: str = "v1",
        sample_rate: Optional[int] = None,
        encoding: str = "pcm_s16le",
        container: str = "raw",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Async TTS service.

        Args:
            api_key: Async API key.
            voice_id: ID of the voice to use for synthesis.
            aiohttp_session: An aiohttp session for making HTTP requests.
            model: TTS model to use (e.g., "asyncflow_multilingual_v1.0").
            url: Base URL for Async API.
            version: API version string for Async API.
            sample_rate: Audio sample rate.
            encoding: Audio encoding format.
            container: Audio container format.
            params: Additional input parameters for voice customization.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or AsyncAIHttpTTSService.InputParams()

        self._api_key = api_key
        self._base_url = url
        self._api_version = version
        self._settings = {
            "output_format": {
                "container": container,
                "encoding": encoding,
                "sample_rate": 0,
            },
            "language": self.language_to_service_language(params.language)
            if params.language
            else None,
        }
        self.set_voice(voice_id)
        self.set_model_name(model)

        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Async HTTP service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Async language format.

        Args:
            language: The language to convert.

        Returns:
            The Async-specific language code, or None if not supported.
        """
        return language_to_async_language(language)

    async def start(self, frame: StartFrame):
        """Start the Async HTTP TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["output_format"]["sample_rate"] = self.sample_rate

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Async's HTTP streaming API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            voice_config = {"mode": "id", "id": self._voice_id}
            await self.start_ttfb_metrics()
            payload = {
                "model_id": self._model_name,
                "transcript": text,
                "voice": voice_config,
                "output_format": self._settings["output_format"],
                "language": self._settings["language"],
            }
            yield TTSStartedFrame()
            headers = {
                "version": self._api_version,
                "x-api-key": self._api_key,
                "Content-Type": "application/json",
            }
            url = f"{self._base_url}/text_to_speech/streaming"

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    await self.push_error(error_msg=f"Async API error: {error_text}")
                    raise Exception(f"Async API returned status {response.status}: {error_text}")

                # Read streaming bytes; stop TTFB on the *first* received chunk
                buffer = bytearray()
                async for chunk in response.content.iter_chunked(64 * 1024):
                    if not chunk:
                        continue
                    await self.stop_ttfb_metrics()
                    buffer.extend(chunk)
                audio_data = bytes(buffer)

            await self.start_tts_usage_metrics(text)

            frame = TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
            )

            yield frame

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
