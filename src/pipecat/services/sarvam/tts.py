#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam AI text-to-speech service implementation."""

import asyncio
import base64
import json
from typing import Any, AsyncGenerator, Mapping, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    StartFrame,
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
    logger.error("In order to use Sarvam, you need to `pip install pipecat-ai[sarvam]`.")
    raise Exception(f"Missing module: {e}")


def language_to_sarvam_language(language: Language) -> Optional[str]:
    """Convert Pipecat Language enum to Sarvam AI language codes.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Sarvam AI language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.BN: "bn-IN",  # Bengali
        Language.EN: "en-IN",  # English (India)
        Language.GU: "gu-IN",  # Gujarati
        Language.HI: "hi-IN",  # Hindi
        Language.KN: "kn-IN",  # Kannada
        Language.ML: "ml-IN",  # Malayalam
        Language.MR: "mr-IN",  # Marathi
        Language.OR: "od-IN",  # Odia
        Language.PA: "pa-IN",  # Punjabi
        Language.TA: "ta-IN",  # Tamil
        Language.TE: "te-IN",  # Telugu
    }

    return LANGUAGE_MAP.get(language)


class SarvamHttpTTSService(TTSService):
    """Text-to-Speech service using Sarvam AI's API.

    Converts text to speech using Sarvam AI's TTS models with support for multiple
    Indian languages. Provides control over voice characteristics like pitch, pace,
    and loudness.

    Example::

        tts = SarvamHttpTTSService(
            api_key="your-api-key",
            voice_id="anushka",
            model="bulbul:v2",
            aiohttp_session=session,
            params=SarvamHttpTTSService.InputParams(
                language=Language.HI,
                pitch=0.1,
                pace=1.2
            )
        )

        # For bulbul v3 beta with any speaker:
        tts_v3 = SarvamHttpTTSService(
            api_key="your-api-key",
            voice_id="speaker_name",
            model="bulbul:v3,
            aiohttp_session=session,
            params=SarvamHttpTTSService.InputParams(
                language=Language.HI,
                temperature=0.8
            )
        )
    """

    class InputParams(BaseModel):
        """Input parameters for Sarvam TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English (India).
            pitch: Voice pitch adjustment (-0.75 to 0.75). Defaults to 0.0.
            pace: Speech pace multiplier (0.3 to 3.0). Defaults to 1.0.
            loudness: Volume multiplier (0.1 to 3.0). Defaults to 1.0.
            enable_preprocessing: Whether to enable text preprocessing. Defaults to False.
        """

        language: Optional[Language] = Language.EN
        pitch: Optional[float] = Field(default=0.0, ge=-0.75, le=0.75)
        pace: Optional[float] = Field(default=1.0, ge=0.3, le=3.0)
        loudness: Optional[float] = Field(default=1.0, ge=0.1, le=3.0)
        enable_preprocessing: Optional[bool] = False
        temperature: Optional[float] = Field(
            default=0.6,
            ge=0.01,
            le=1.0,
            description="Controls the randomness of the output for bulbul v3 beta. "
            "Lower values make the output more focused and deterministic, while "
            "higher values make it more random. Range: 0.01 to 1.0. Default: 0.6.",
        )

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: str = "anushka",
        model: str = "bulbul:v2",
        base_url: str = "https://api.sarvam.ai",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Sarvam TTS service.

        Args:
            api_key: Sarvam AI API subscription key.
            aiohttp_session: Shared aiohttp session for making requests.
            voice_id: Speaker voice ID (e.g., "anushka", "meera"). Defaults to "anushka".
            model: TTS model to use ("bulbul:v2" or "bulbul:v3-beta" or "bulbul:v3"). Defaults to "bulbul:v2".
            base_url: Sarvam AI API base URL. Defaults to "https://api.sarvam.ai".
            sample_rate: Audio sample rate in Hz (8000, 16000, 22050, 24000). If None, uses default.
            params: Additional voice and preprocessing parameters. If None, uses defaults.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or SarvamHttpTTSService.InputParams()

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session

        # Build base settings common to all models
        self._settings = {
            "language": (
                self.language_to_service_language(params.language) if params.language else "en-IN"
            ),
            "enable_preprocessing": params.enable_preprocessing,
        }

        # Add model-specific parameters
        if model in ("bulbul:v3-beta", "bulbul:v3"):
            self._settings.update(
                {
                    "temperature": getattr(params, "temperature", 0.6),
                    "model": model,
                }
            )
        else:
            self._settings.update(
                {
                    "pitch": params.pitch,
                    "pace": params.pace,
                    "loudness": params.loudness,
                    "model": model,
                }
            )

        self.set_model_name(model)
        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Sarvam AI language format.

        Args:
            language: The language to convert.

        Returns:
            The Sarvam AI-specific language code, or None if not supported.
        """
        return language_to_sarvam_language(language)

    async def start(self, frame: StartFrame):
        """Start the Sarvam TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Sarvam AI's API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            payload = {
                "text": text,
                "target_language_code": self._settings["language"],
                "speaker": self._voice_id,
                "pitch": self._settings["pitch"],
                "pace": self._settings["pace"],
                "loudness": self._settings["loudness"],
                "sample_rate": self.sample_rate,
                "enable_preprocessing": self._settings["enable_preprocessing"],
                "model": self._model_name,
            }

            headers = {
                "api-subscription-key": self._api_key,
                "Content-Type": "application/json",
            }

            url = f"{self._base_url}/text-to-speech"

            yield TTSStartedFrame()

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Sarvam API error: {error_text}")
                    await self.push_error(ErrorFrame(f"Sarvam API error: {error_text}"))
                    return

                response_data = await response.json()

            await self.start_tts_usage_metrics(text)

            # Decode base64 audio data
            if "audios" not in response_data or not response_data["audios"]:
                logger.error("No audio data received from Sarvam API")
                await self.push_error(ErrorFrame("No audio data received"))
                return

            # Get the first audio (there should be only one for single text input)
            base64_audio = response_data["audios"][0]
            audio_data = base64.b64decode(base64_audio)

            # Strip WAV header (first 44 bytes) if present
            if audio_data.startswith(b"RIFF"):
                logger.debug("Stripping WAV header from Sarvam audio data")
                audio_data = audio_data[44:]

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


class SarvamTTSService(InterruptibleTTSService):
    """WebSocket-based text-to-speech service using Sarvam AI.

    Provides streaming TTS with real-time audio generation for multiple Indian languages.
    Supports voice control parameters like pitch, pace, and loudness adjustment.

    Example::

        tts = SarvamTTSService(
            api_key="your-api-key",
            voice_id="anushka",
            model="bulbul:v2",
            params=SarvamTTSService.InputParams(
                language=Language.HI,
                pitch=0.1,
                pace=1.2
            )
        )

        # For bulbul v3 beta with any speaker and temperature:
        # Note: pace and loudness are not supported for bulbul v3 and bulbul v3 beta
        tts_v3 = SarvamTTSService(
            api_key="your-api-key",
            voice_id="speaker_name",
            model="bulbul:v3",
            params=SarvamTTSService.InputParams(
                language=Language.HI,
                temperature=0.8
            )
        )
    """

    class InputParams(BaseModel):
        """Configuration parameters for Sarvam TTS.

        Parameters:
            pitch: Voice pitch adjustment (-0.75 to 0.75). Defaults to 0.0.
            pace: Speech pace multiplier (0.3 to 3.0). Defaults to 1.0.
            loudness: Volume multiplier (0.1 to 3.0). Defaults to 1.0.
            enable_preprocessing: Enable text preprocessing. Defaults to False.
            min_buffer_size: Minimum number of characters to buffer before generating audio.
                Lower values reduce latency but may affect quality. Defaults to 50.
            max_chunk_length: Maximum number of characters processed in a single chunk.
                Controls memory usage and processing efficiency. Defaults to 200.
            output_audio_codec: Audio codec format. Defaults to "linear16".
            output_audio_bitrate: Audio bitrate. Defaults to "128k".
            language: Target language for synthesis. Supports Bengali (bn-IN), English (en-IN),
                Gujarati (gu-IN), Hindi (hi-IN), Kannada (kn-IN), Malayalam (ml-IN),
                Marathi (mr-IN), Odia (od-IN), Punjabi (pa-IN), Tamil (ta-IN),
                Telugu (te-IN). Defaults to en-IN.

                Available Speakers:
            Female: anushka, manisha, vidya, arya
            Male: abhilash, karun, hitesh
        """

        pitch: Optional[float] = Field(default=0.0, ge=-0.75, le=0.75)
        pace: Optional[float] = Field(default=1.0, ge=0.3, le=3.0)
        loudness: Optional[float] = Field(default=1.0, ge=0.1, le=3.0)
        enable_preprocessing: Optional[bool] = False
        min_buffer_size: Optional[int] = 50
        max_chunk_length: Optional[int] = 200
        output_audio_codec: Optional[str] = "linear16"
        output_audio_bitrate: Optional[str] = "128k"
        language: Optional[Language] = Language.EN
        temperature: Optional[float] = Field(
            default=0.6,
            ge=0.01,
            le=1.0,
            description="Controls the randomness of the output for bulbul v3 beta. "
            "Lower values make the output more focused and deterministic, while "
            "higher values make it more random. Range: 0.01 to 1.0. Default: 0.6.",
        )

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "bulbul:v2",
        voice_id: str = "anushka",
        url: str = "wss://api.sarvam.ai/text-to-speech/ws",
        aiohttp_session: Optional[aiohttp.ClientSession] = None,
        aggregate_sentences: Optional[bool] = True,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Sarvam TTS service with voice and transport configuration.

        Args:
            api_key: Sarvam API key for authenticating TTS requests.
            model: Identifier of the Sarvam speech model (default "bulbul:v2").
                Supports "bulbul:v2", "bulbul:v3-beta" and "bulbul:v3".
            voice_id: Voice identifier for synthesis (default "anushka").
            url: WebSocket URL for connecting to the TTS backend (default production URL).
            aiohttp_session: Optional shared aiohttp session. To maintain backward compatibility.

                .. deprecated:: 0.0.81
                    aiohttp_session is no longer used. This parameter will be removed in a future version.

            aggregate_sentences: Whether to merge multiple sentences into one audio chunk (default True).
            sample_rate: Desired sample rate for the output audio in Hz (overrides default if set).
            params: Optional input parameters to override global configuration.
            **kwargs: Optional keyword arguments forwarded to InterruptibleTTSService (such as
                `push_stop_frames`, `sample_rate`, task manager parameters, event hooks, etc.)
                to customize transport behavior or enable metrics support.

        This method sets up the internal TTS configuration mapping, constructs the WebSocket
        URL based on the chosen model, and initializes state flags before connecting.
        """
        # Initialize parent class first
        super().__init__(
            aggregate_sentences=aggregate_sentences,
            push_text_frames=True,
            pause_frame_processing=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            **kwargs,
        )
        params = params or SarvamTTSService.InputParams()
        if aiohttp_session is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The 'aiohttp_session' parameter is deprecated and will be removed in a future version. ",
                    DeprecationWarning,
                    stacklevel=2,
                )
        # WebSocket endpoint URL
        self._websocket_url = f"{url}?model={model}"
        self._api_key = api_key
        self.set_model_name(model)
        self.set_voice(voice_id)
        # Build base settings common to all models
        self._settings = {
            "target_language_code": (
                self.language_to_service_language(params.language) if params.language else "en-IN"
            ),
            "speaker": voice_id,
            "speech_sample_rate": 0,
            "enable_preprocessing": params.enable_preprocessing,
            "min_buffer_size": params.min_buffer_size,
            "max_chunk_length": params.max_chunk_length,
            "output_audio_codec": params.output_audio_codec,
            "output_audio_bitrate": params.output_audio_bitrate,
        }

        # Add model-specific parameters
        if model in ("bulbul:v3-beta", "bulbul:v3"):
            self._settings.update(
                {
                    "temperature": getattr(params, "temperature", 0.6),
                    "model": model,
                }
            )
        else:
            self._settings.update(
                {
                    "pitch": params.pitch,
                    "pace": params.pace,
                    "loudness": params.loudness,
                    "model": model,
                }
            )
        self._started = False

        self._receive_task = None
        self._keepalive_task = None
        self._disconnecting = False

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Sarvam AI language format.

        Args:
            language: The language to convert.

        Returns:
            The Sarvam AI-specific language code, or None if not supported.
        """
        return language_to_sarvam_language(language)

    async def start(self, frame: StartFrame):
        """Start the Sarvam TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

        self._settings["speech_sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Sarvam TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Sarvam TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        """Flush any pending audio synthesis by sending stop command."""
        if self._websocket:
            msg = {"type": "flush"}
            await self._websocket.send(json.dumps(msg))

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame downstream with special handling for stop conditions.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            self._started = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame and flush audio if it's the end of a full response."""
        if isinstance(frame, LLMFullResponseEndFrame):
            await self.flush_audio()
        return await super().process_frame(frame, direction)

    async def _update_settings(self, settings: Mapping[str, Any]):
        """Update service settings and reconnect if voice changed."""
        prev_voice = self._voice_id
        await super()._update_settings(settings)
        if not prev_voice == self._voice_id:
            logger.info(f"Switching TTS voice to: [{self._voice_id}]")
            await self._send_config()

    async def _connect(self):
        """Connect to Sarvam WebSocket and start background tasks."""
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(
                self._keepalive_task_handler(),
            )

    async def _disconnect(self):
        """Disconnect from Sarvam WebSocket and clean up tasks."""
        try:
            # First, set a flag to prevent new operations
            self._disconnecting = True

            # Cancel background tasks BEFORE closing websocket
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=2.0)
                self._receive_task = None

            if self._keepalive_task:
                await self.cancel_task(self._keepalive_task, timeout=2.0)
                self._keepalive_task = None

            # Now close the websocket
            await self._disconnect_websocket()

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            # Reset state only after everything is cleaned up
            self._started = False
            self._websocket = None
            self._disconnecting = False

    async def _connect_websocket(self):
        """Establish WebSocket connection to Sarvam API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._websocket = await websocket_connect(
                self._websocket_url,
                additional_headers={
                    "api-subscription-key": self._api_key,
                },
            )
            logger.debug("Connected to Sarvam TTS Websocket")
            await self._send_config()

            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _send_config(self):
        """Send initial configuration message."""
        if not self._websocket:
            raise Exception("WebSocket not connected")
        self._settings["speaker"] = self._voice_id
        logger.debug(f"Config being sent is {self._settings}")
        config_message = {"type": "config", "data": self._settings}

        try:
            await self._websocket.send(json.dumps(config_message))
            logger.debug("Configuration sent successfully")
        except Exception as e:
            logger.error(f"Failed to send config: {str(e)}")
            await self.push_frame(ErrorFrame(f"Failed to send config: {str(e)}"))
            raise

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Sarvam")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._started = False
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and process messages from Sarvam WebSocket."""
        async for message in self._get_websocket():
            if isinstance(message, str):
                msg = json.loads(message)
                if msg.get("type") == "audio":
                    # Check for interruption before processing audio
                    await self.stop_ttfb_metrics()
                    audio = base64.b64decode(msg["data"]["audio"])
                    frame = TTSAudioRawFrame(audio, self.sample_rate, 1)
                    await self.push_frame(frame)
                elif msg.get("type") == "error":
                    error_msg = msg["data"]["message"]
                    logger.error(f"TTS Error: {error_msg}")

                    # If it's a timeout error, the connection might need to be reset
                    if "too long" in error_msg.lower() or "timeout" in error_msg.lower():
                        logger.warning("Connection timeout detected, service may need restart")

                    await self.push_frame(ErrorFrame(f"TTS Error: {error_msg}"))

    async def _keepalive_task_handler(self):
        """Handle keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 20
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            await self._send_keepalive()

    async def _send_keepalive(self):
        """Send keepalive message to maintain connection."""
        if self._disconnecting:
            return

        if self._websocket and self._websocket.state == State.OPEN:
            msg = {"type": "ping"}
            await self._websocket.send(json.dumps(msg))

    async def _send_text(self, text: str):
        """Send text to Sarvam WebSocket for synthesis."""
        if self._disconnecting:
            logger.warning("Service is disconnecting, ignoring text send")
            return

        if self._websocket and self._websocket.state == State.OPEN:
            msg = {"type": "text", "data": {"text": text}}
            await self._websocket.send(json.dumps(msg))
        else:
            logger.warning("WebSocket not ready, cannot send text")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech audio frames from input text using Sarvam TTS.

        Sends text over WebSocket for synthesis and yields corresponding audio or status frames.

        Args:
            text: The text input to synthesize.

        Yields:
            Frame objects including TTSStartedFrame, TTSAudioRawFrame(s), or TTSStoppedFrame.
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
