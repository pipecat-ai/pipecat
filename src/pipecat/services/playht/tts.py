#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""PlayHT text-to-speech service implementations.

This module provides integration with PlayHT's text-to-speech API
supporting both WebSocket streaming and HTTP-based synthesis.
"""

import io
import json
import struct
import uuid
import warnings
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
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use PlayHTTTSService, you need to `pip install pipecat-ai[playht]`.")
    raise Exception(f"Missing module: {e}")


def language_to_playht_language(language: Language) -> Optional[str]:
    """Convert a Language enum to PlayHT language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding PlayHT language code, or None if not supported.
    """
    BASE_LANGUAGES = {
        Language.AF: "afrikans",
        Language.AM: "amharic",
        Language.AR: "arabic",
        Language.BN: "bengali",
        Language.BG: "bulgarian",
        Language.CA: "catalan",
        Language.CS: "czech",
        Language.DA: "danish",
        Language.DE: "german",
        Language.EL: "greek",
        Language.EN: "english",
        Language.ES: "spanish",
        Language.FR: "french",
        Language.GL: "galician",
        Language.HE: "hebrew",
        Language.HI: "hindi",
        Language.HR: "croatian",
        Language.HU: "hungarian",
        Language.ID: "indonesian",
        Language.IT: "italian",
        Language.JA: "japanese",
        Language.KO: "korean",
        Language.MS: "malay",
        Language.NL: "dutch",
        Language.PL: "polish",
        Language.PT: "portuguese",
        Language.RU: "russian",
        Language.SQ: "albanian",
        Language.SR: "serbian",
        Language.SV: "swedish",
        Language.TH: "thai",
        Language.TL: "tagalog",
        Language.TR: "turkish",
        Language.UK: "ukrainian",
        Language.UR: "urdu",
        Language.XH: "xhosa",
        Language.ZH: "mandarin",
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


class PlayHTTTSService(InterruptibleTTSService):
    """PlayHT WebSocket-based text-to-speech service.

    Provides real-time text-to-speech synthesis using PlayHT's WebSocket API.
    Supports streaming audio generation with configurable voice engines and
    language settings.
    """

    class InputParams(BaseModel):
        """Input parameters for PlayHT TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English.
            speed: Speech speed multiplier. Defaults to 1.0.
            seed: Random seed for voice consistency.
        """

        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0
        seed: Optional[int] = None

    def __init__(
        self,
        *,
        api_key: str,
        user_id: str,
        voice_url: str,
        voice_engine: str = "Play3.0-mini",
        sample_rate: Optional[int] = None,
        output_format: str = "wav",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the PlayHT WebSocket TTS service.

        Args:
            api_key: PlayHT API key for authentication.
            user_id: PlayHT user ID for authentication.
            voice_url: URL of the voice to use for synthesis.
            voice_engine: Voice engine to use. Defaults to "Play3.0-mini".
            sample_rate: Audio sample rate. If None, uses default.
            output_format: Audio output format. Defaults to "wav".
            params: Additional input parameters for voice customization.
            **kwargs: Additional arguments passed to parent InterruptibleTTSService.
        """
        super().__init__(
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        params = params or PlayHTTTSService.InputParams()

        self._api_key = api_key
        self._user_id = user_id
        self._websocket_url = None
        self._receive_task = None
        self._request_id = None

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "english",
            "output_format": output_format,
            "voice_engine": voice_engine,
            "speed": params.speed,
            "seed": params.seed,
        }
        self.set_model_name(voice_engine)
        self.set_voice(voice_url)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as PlayHT service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to PlayHT service language format.

        Args:
            language: The language to convert.

        Returns:
            The PlayHT-specific language code, or None if not supported.
        """
        return language_to_playht_language(language)

    async def start(self, frame: StartFrame):
        """Start the PlayHT TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the PlayHT TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the PlayHT TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Connect to PlayHT WebSocket and start receive task."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from PlayHT WebSocket and clean up tasks."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to PlayHT websocket."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to PlayHT")

            if not self._websocket_url:
                await self._get_websocket_url()

            if not isinstance(self._websocket_url, str):
                raise ValueError("WebSocket URL is not a string")

            self._websocket = await websocket_connect(self._websocket_url)
        except ValueError as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Disconnect from PlayHT websocket."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from PlayHT")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._request_id = None
            self._websocket = None

    async def _get_websocket_url(self):
        """Retrieve WebSocket URL from PlayHT API."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.play.ht/api/v4/websocket-auth",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "X-User-Id": self._user_id,
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status in (200, 201):
                    data = await response.json()
                    # Handle the new response format with multiple URLs
                    if "websocket_urls" in data:
                        # Select URL based on voice_engine
                        if self._settings["voice_engine"] in data["websocket_urls"]:
                            self._websocket_url = data["websocket_urls"][
                                self._settings["voice_engine"]
                            ]
                        else:
                            raise ValueError(
                                f"Unsupported voice engine: {self._settings['voice_engine']}"
                            )
                    else:
                        raise ValueError("Invalid response: missing websocket_urls")
                else:
                    raise Exception(f"Failed to get WebSocket URL: {response.status}")

    def _get_websocket(self):
        """Get the WebSocket connection if available."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: StartInterruptionFrame, direction: FrameDirection):
        """Handle interruption by stopping metrics and clearing request ID."""
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._request_id = None

    async def _receive_messages(self):
        """Receive messages from PlayHT websocket."""
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                # Skip the WAV header message
                if message.startswith(b"RIFF"):
                    continue
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(message, self.sample_rate, 1)
                await self.push_frame(frame)
            else:
                logger.debug(f"Received text message: {message}")
                try:
                    msg = json.loads(message)
                    if msg.get("type") == "start":
                        # Handle start of stream
                        logger.debug(f"Started processing request: {msg.get('request_id')}")
                    elif msg.get("type") == "end":
                        # Handle end of stream
                        if "request_id" in msg and msg["request_id"] == self._request_id:
                            await self.push_frame(TTSStoppedFrame())
                            self._request_id = None
                    elif "error" in msg:
                        logger.error(f"{self} error: {msg}")
                        await self.push_error(ErrorFrame(f"{self} error: {msg['error']}"))
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON message: {message}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio from text using PlayHT's WebSocket API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            # Reconnect if the websocket is closed
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if not self._request_id:
                await self.start_ttfb_metrics()
                yield TTSStartedFrame()
                self._request_id = str(uuid.uuid4())

            tts_command = {
                "text": text,
                "voice": self._voice_id,
                "voice_engine": self._settings["voice_engine"],
                "output_format": self._settings["output_format"],
                "sample_rate": self.sample_rate,
                "language": self._settings["language"],
                "speed": self._settings["speed"],
                "seed": self._settings["seed"],
                "request_id": self._request_id,
            }

            try:
                await self._get_websocket().send(json.dumps(tts_command))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return

            # The actual audio frames will be handled in _receive_task_handler
            yield None

        except Exception as e:
            logger.error(f"{self} error generating TTS: {e}")
            yield ErrorFrame(f"{self} error: {str(e)}")


class PlayHTHttpTTSService(TTSService):
    """PlayHT HTTP-based text-to-speech service.

    Provides text-to-speech synthesis using PlayHT's HTTP API for simpler,
    non-streaming synthesis. Suitable for use cases where streaming is not
    required and simpler integration is preferred.
    """

    class InputParams(BaseModel):
        """Input parameters for PlayHT HTTP TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English.
            speed: Speech speed multiplier. Defaults to 1.0.
            seed: Random seed for voice consistency.
        """

        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0
        seed: Optional[int] = None

    def __init__(
        self,
        *,
        api_key: str,
        user_id: str,
        voice_url: str,
        voice_engine: str = "Play3.0-mini",
        protocol: Optional[str] = None,
        output_format: str = "wav",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the PlayHT HTTP TTS service.

        Args:
            api_key: PlayHT API key for authentication.
            user_id: PlayHT user ID for authentication.
            voice_url: URL of the voice to use for synthesis.
            voice_engine: Voice engine to use. Defaults to "Play3.0-mini".
            protocol: Protocol to use ("http" or "ws").

                .. deprecated:: 0.0.80
                    This parameter no longer has any effect and will be removed in a future version.
                    Use PlayHTTTSService for WebSocket or PlayHTHttpTTSService for HTTP.

            output_format: Audio output format. Defaults to "wav".
            sample_rate: Audio sample rate. If None, uses default.
            params: Additional input parameters for voice customization.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        # Warn about deprecated protocol parameter if explicitly provided
        if protocol:
            warnings.warn(
                "The 'protocol' parameter is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        params = params or PlayHTHttpTTSService.InputParams()

        self._user_id = user_id
        self._api_key = api_key

        # Check if voice_engine contains protocol information (backward compatibility)
        if "-http" in voice_engine:
            # Extract the base engine name
            voice_engine = voice_engine.replace("-http", "")
        elif "-ws" in voice_engine:
            # Extract the base engine name
            voice_engine = voice_engine.replace("-ws", "")

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "english",
            "output_format": output_format,
            "voice_engine": voice_engine,
            "speed": params.speed,
            "seed": params.seed,
        }
        self.set_model_name(voice_engine)
        self.set_voice(voice_url)

    async def start(self, frame: StartFrame):
        """Start the PlayHT HTTP TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as PlayHT HTTP service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to PlayHT service language format.

        Args:
            language: The language to convert.

        Returns:
            The PlayHT-specific language code, or None if not supported.
        """
        return language_to_playht_language(language)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio from text using PlayHT's HTTP API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            # Prepare the request payload
            payload = {
                "text": text,
                "voice": self._voice_id,
                "voice_engine": self._settings["voice_engine"],
                "output_format": self._settings["output_format"],
                "sample_rate": self.sample_rate,
                "language": self._settings["language"],
            }

            # Add optional parameters if they exist
            if self._settings["speed"] is not None:
                payload["speed"] = self._settings["speed"]
            if self._settings["seed"] is not None:
                payload["seed"] = self._settings["seed"]

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "X-User-Id": self._user_id,
                "Content-Type": "application/json",
                "Accept": "*/*",
            }

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.play.ht/api/v2/tts/stream",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status not in (200, 201):
                        error_text = await response.text()
                        raise Exception(f"PlayHT API error {response.status}: {error_text}")

                    in_header = True
                    buffer = b""

                    CHUNK_SIZE = self.chunk_size

                    async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                        if len(chunk) == 0:
                            continue

                        # Skip the RIFF header
                        if in_header:
                            buffer += chunk
                            if len(buffer) <= 36:
                                continue
                            else:
                                fh = io.BytesIO(buffer)
                                fh.seek(36)
                                (data, size) = struct.unpack("<4sI", fh.read(8))
                                while data != b"data":
                                    fh.read(size)
                                    (data, size) = struct.unpack("<4sI", fh.read(8))
                                # Extract audio data after header
                                audio_data = buffer[fh.tell() :]
                                if len(audio_data) > 0:
                                    await self.stop_ttfb_metrics()
                                    frame = TTSAudioRawFrame(audio_data, self.sample_rate, 1)
                                    yield frame
                                in_header = False
                        elif len(chunk) > 0:
                            await self.stop_ttfb_metrics()
                            frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                            yield frame

        except Exception as e:
            logger.error(f"{self} error generating TTS: {e}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
