#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Async text-to-speech service implementations."""

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Mapping, Optional

import aiohttp
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
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import AudioContextTTSService, TextAggregationMode, TTSService
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


@dataclass
class AsyncAITTSSettings(TTSSettings):
    """Settings for Async AI TTS services.

    Parameters:
        output_container: Audio container format (e.g. "raw").
        output_encoding: Audio encoding format (e.g. "pcm_s16le").
        output_sample_rate: Audio sample rate in Hz.
    """

    output_container: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    output_encoding: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    output_sample_rate: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> "AsyncAITTSSettings":
        """Construct settings from a plain dict, destructuring legacy nested ``output_format``."""
        flat = dict(settings)
        nested = flat.pop("output_format", None)
        if isinstance(nested, dict):
            flat.setdefault("output_container", nested.get("container"))
            flat.setdefault("output_encoding", nested.get("encoding"))
            flat.setdefault("output_sample_rate", nested.get("sample_rate"))
        return super().from_mapping(flat)


class AsyncAITTSService(AudioContextTTSService):
    """Async TTS service with WebSocket streaming.

    Provides text-to-speech using Async's streaming WebSocket API.
    """

    _settings: AsyncAITTSSettings

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
        url: str = "wss://api.async.com/text_to_speech/websocket/ws",
        model: str = "async_flash_v1.0",
        sample_rate: Optional[int] = None,
        encoding: str = "pcm_s16le",
        container: str = "raw",
        params: Optional[InputParams] = None,
        aggregate_sentences: Optional[bool] = None,
        text_aggregation_mode: Optional[TextAggregationMode] = None,
        **kwargs,
    ):
        """Initialize the Async TTS service.

        Args:
            api_key: Async API key.
            voice_id: UUID of the voice to use for synthesis. See docs for a full list:
                https://docs.async.com/list-voices-16699698e0
            version: Async API version.
            url: WebSocket URL for Async TTS API.
            model: TTS model to use (e.g., "async_flash_v1.0").
            sample_rate: Audio sample rate.
            encoding: Audio encoding format.
            container: Audio container format.
            params: Additional input parameters for voice customization.
            aggregate_sentences: Deprecated. Use text_aggregation_mode instead.

                .. deprecated:: 0.0.104
                    Use ``text_aggregation_mode`` instead.

            text_aggregation_mode: How to aggregate text before synthesis.
            **kwargs: Additional arguments passed to the parent service.
        """
        params = params or AsyncAITTSService.InputParams()

        super().__init__(
            aggregate_sentences=aggregate_sentences,
            text_aggregation_mode=text_aggregation_mode,
            pause_frame_processing=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=AsyncAITTSSettings(
                model=model,
                voice=voice_id,
                output_container=container,
                output_encoding=encoding,
                output_sample_rate=0,
                language=self.language_to_service_language(params.language)
                if params.language
                else None,
            ),
            **kwargs,
        )

        self._api_key = api_key
        self._api_version = version
        self._url = url

        self._receive_task = None
        self._keepalive_task = None

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Settings are stored but not applied to the active connection.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        self._warn_unhandled_updated_settings(changed)

        return changed

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
        self._settings.output_sample_rate = self.sample_rate
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
                "model_id": self._settings.model,
                "voice": {"mode": "id", "id": self._settings.voice},
                "output_format": {
                    "container": self._settings.output_container,
                    "encoding": self._settings.output_encoding,
                    "sample_rate": self._settings.output_sample_rate,
                },
                "language": self._settings.language,
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
                if self.has_active_audio_context():
                    await self._websocket.send(json.dumps({"terminate": True}))
                await self._websocket.close()
                logger.debug("Disconnected from Async")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            await self.remove_active_audio_context()
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def flush_audio(self):
        """Flush any pending audio."""
        context_id = self.get_active_audio_context_id()
        if not context_id or not self._websocket:
            return
        logger.trace(f"{self}: flushing audio")
        msg = self._build_msg(text=" ", context_id=context_id, force=True)
        await self._websocket.send(msg)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame downstream with special handling for stop conditions.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)

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
                if self.get_active_audio_context_id() == received_ctx_id:
                    logger.debug(
                        f"Received a delayed message, recreating the context: {received_ctx_id}"
                    )
                    await self.create_audio_context(received_ctx_id)
                else:
                    # This can happen if a message is received _after_ we have closed a context
                    # due to user interruption but _before_ the `isFinal` message for the context
                    # is received.
                    logger.debug(f"Ignoring message from unavailable context: {received_ctx_id}")
                    continue

            if msg.get("audio"):
                await self.stop_ttfb_metrics()
                audio = base64.b64decode(msg["audio"])
                frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=received_ctx_id)
                await self.append_to_audio_context(received_ctx_id, frame)

    async def _keepalive_task_handler(self):
        """Send periodic keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 10
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            try:
                if self._websocket and self._websocket.state is State.OPEN:
                    context_id = self.get_active_audio_context_id()
                    if context_id:
                        keepalive_message = {
                            "transcript": " ",
                            "context_id": context_id,
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

    async def _close_context(self, context_id: str):
        # Async AI requires explicit context closure to free server-side resources,
        # both on interruption and on normal completion.
        if context_id and self._websocket:
            try:
                await self._websocket.send(
                    json.dumps({"context_id": context_id, "close_context": True, "transcript": ""})
                )
            except Exception as e:
                logger.error(f"{self}: Error closing context {context_id}: {e}")

    async def on_audio_context_interrupted(self, context_id: str):
        """Close the Async AI context when the bot is interrupted."""
        await self._close_context(context_id)

    async def on_audio_context_completed(self, context_id: str):
        """Close the Async AI context after all audio has been played.

        Async AI does not send a server-side signal when a context is
        exhausted, so Pipecat must explicitly close it with
        ``close_context: True`` to free server-side resources.
        """
        await self._close_context(context_id)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Async API websocket endpoint.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self.has_active_audio_context():
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame(context_id=context_id)
                    if not self.audio_context_available(context_id):
                        await self.create_audio_context(context_id)

                msg = self._build_msg(text=text, force=True, context_id=context_id)
                await self._get_websocket().send(msg)
                await self.start_tts_usage_metrics(text)

            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
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

    _settings: AsyncAITTSSettings

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
        model: str = "async_flash_v1.0",
        url: str = "https://api.async.com",
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
            model: TTS model to use (e.g., "async_flash_v1.0").
            url: Base URL for Async API.
            version: API version string for Async API.
            sample_rate: Audio sample rate.
            encoding: Audio encoding format.
            container: Audio container format.
            params: Additional input parameters for voice customization.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        params = params or AsyncAIHttpTTSService.InputParams()

        super().__init__(
            sample_rate=sample_rate,
            settings=AsyncAITTSSettings(
                model=model,
                voice=voice_id,
                output_container=container,
                output_encoding=encoding,
                output_sample_rate=0,
                language=self.language_to_service_language(params.language)
                if params.language
                else None,
            ),
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = url
        self._api_version = version

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
        self._settings.output_sample_rate = self.sample_rate

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Async's HTTP streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            voice_config = {"mode": "id", "id": self._settings.voice}
            await self.start_ttfb_metrics()
            payload = {
                "model_id": self._settings.model,
                "transcript": text,
                "voice": voice_config,
                "output_format": {
                    "container": self._settings.output_container,
                    "encoding": self._settings.output_encoding,
                    "sample_rate": self._settings.output_sample_rate,
                },
                "language": self._settings.language,
            }
            yield TTSStartedFrame(context_id=context_id)
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
                context_id=context_id,
            )

            yield frame

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame(context_id=context_id)
