#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Async text-to-speech service implementations."""

import asyncio
import base64
import json
from typing import AsyncGenerator, Optional

import aiohttp
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
from pipecat.services.tts_service import InterruptibleTTSService, TTSService
from pipecat.transcriptions.language import Language
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
    BASE_LANGUAGES = {
        Language.EN: "en",
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        # Convert enum value to string and get the base language part (e.g. en-En -> en)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Look up the base code in our supported languages
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


class AsyncAITTSService(InterruptibleTTSService):
    """Async TTS service with WebSocket streaming.

    Provides text-to-speech using Async's streaming WebSocket API.
    """

    class InputParams(BaseModel):
        """Input parameters for Async TTS configuration.

        Parameters:
            language: Language to use for synthesis.
        """

        language: Optional[Language] = Language.EN

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        version: str = "v1",
        url: str = "wss://api.async.ai/text_to_speech/websocket/ws",
        model: str = "asyncflow_v2.0",
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
            model: TTS model to use (e.g., "asyncflow_v2.0").
            sample_rate: Audio sample rate.
            encoding: Audio encoding format.
            container: Audio container format.
            params: Additional input parameters for voice customization.
            aggregate_sentences: Whether to aggregate sentences within the TTSService.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,
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
            else "en",
        }

        self.set_model_name(model)
        self.set_voice(voice_id)

        self._receive_task = None
        self._keepalive_task = None
        self._started = False

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

    def _build_msg(self, text: str = "", force: bool = False) -> str:
        msg = {"transcript": text, "force": force}
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
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
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
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Async")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._websocket = None
            self._started = False

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def flush_audio(self):
        """Flush any pending audio."""
        if not self._websocket:
            return
        logger.trace(f"{self}: flushing audio")
        msg = self._build_msg(text=" ", force=True)
        await self._websocket.send(msg)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame downstream with special handling for stop conditions.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):
            self._started = False

    async def _receive_messages(self):
        async for message in self._get_websocket():
            msg = json.loads(message)
            if not msg:
                continue

            elif msg.get("audio"):
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(msg["audio"]),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
                await self.push_frame(frame)
            elif msg.get("error_code"):
                logger.error(f"{self} error: {msg}")
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(ErrorFrame(f"{self} error: {msg['message']}"))
            else:
                logger.error(f"{self} error, unknown message type: {msg}")

    async def _keepalive_task_handler(self):
        """Send periodic keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 3
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            try:
                if self._websocket and self._websocket.state is State.OPEN:
                    keepalive_message = {"transcript": " "}
                    logger.trace("Sending keepalive message")
                    await self._websocket.send(json.dumps(keepalive_message))
            except websockets.ConnectionClosed as e:
                logger.warning(f"{self} keepalive error: {e}")
                break

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

            if not self._started:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._started = True

            msg = self._build_msg(text=text, force=True)

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

        language: Optional[Language] = Language.EN

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "asyncflow_v2.0",
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
            model: TTS model to use (e.g., "asyncflow_v2.0").
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
            else "en",
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
                    logger.error(f"Async API error: {error_text}")
                    await self.push_error(ErrorFrame(f"Async API error: {error_text}"))
                    raise Exception(f"Async API returned status {response.status}: {error_text}")

                audio_data = await response.read()

            await self.start_tts_usage_metrics(text)

            frame = TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
            )

            yield frame

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(f"Error generating TTS: {e}"))
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
