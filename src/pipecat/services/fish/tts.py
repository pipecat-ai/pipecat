#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio text-to-speech service implementation.

This module provides integration with Fish Audio's real-time TTS WebSocket API
for streaming text-to-speech synthesis with customizable voice parameters.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, ClassVar, Dict, Literal, Mapping, Optional

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
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import ormsgpack
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Fish Audio, you need to `pip install pipecat-ai[fish]`.")
    raise Exception(f"Missing module: {e}")

# FishAudio supports various output formats
FishAudioOutputFormat = Literal["opus", "mp3", "pcm", "wav"]


@dataclass
class FishAudioTTSSettings(TTSSettings):
    """Settings for Fish Audio TTS service.

    Parameters:
        fish_sample_rate: Audio sample rate sent to the API.
        latency: Latency mode ("normal" or "balanced"). Defaults to "normal".
        format: Audio output format.
        normalize: Whether to normalize audio output. Defaults to True.
        prosody_speed: Speech speed multiplier (0.5-2.0). Defaults to 1.0.
        prosody_volume: Volume adjustment in dB. Defaults to 0.
        reference_id: Reference ID of the voice model.
    """

    fish_sample_rate: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    latency: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    format: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    normalize: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prosody_speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prosody_volume: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    reference_id: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[Dict[str, str]] = {"voice_id": "voice", "sample_rate": "fish_sample_rate"}

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> "FishAudioTTSSettings":
        """Construct settings from a plain dict, destructuring legacy nested ``prosody``."""
        flat = dict(settings)
        nested = flat.pop("prosody", None)
        if isinstance(nested, dict):
            flat.setdefault("prosody_speed", nested.get("speed"))
            flat.setdefault("prosody_volume", nested.get("volume"))
        return super().from_mapping(flat)


class FishAudioTTSService(InterruptibleTTSService):
    """Fish Audio text-to-speech service with WebSocket streaming.

    Provides real-time text-to-speech synthesis using Fish Audio's WebSocket API.
    Supports various audio formats, customizable prosody controls, and streaming
    audio generation with interruption handling.
    """

    _settings: FishAudioTTSSettings

    class InputParams(BaseModel):
        """Input parameters for Fish Audio TTS configuration.

        Parameters:
            language: Language for synthesis. Defaults to English.
            latency: Latency mode ("normal" or "balanced"). Defaults to "normal".
            normalize: Whether to normalize audio output. Defaults to True.
            prosody_speed: Speech speed multiplier (0.5-2.0). Defaults to 1.0.
            prosody_volume: Volume adjustment in dB. Defaults to 0.
        """

        language: Optional[Language] = Language.EN
        latency: Optional[str] = "normal"  # "normal" or "balanced"
        normalize: Optional[bool] = True
        prosody_speed: Optional[float] = 1.0  # Speech speed (0.5-2.0)
        prosody_volume: Optional[int] = 0  # Volume adjustment in dB

    def __init__(
        self,
        *,
        api_key: str,
        reference_id: Optional[str] = None,  # This is the voice ID
        model: Optional[str] = None,  # Deprecated
        model_id: str = "s1",
        output_format: FishAudioOutputFormat = "pcm",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Fish Audio TTS service.

        Args:
            api_key: Fish Audio API key for authentication.
            reference_id: Reference ID of the voice model to use for synthesis.
            model: Deprecated. Reference ID of the voice model to use for synthesis.

              .. deprecated:: 0.0.74
                The `model` parameter is deprecated and will be removed in version 0.1.0.
                Use `reference_id` instead to specify the voice model.

            model_id: Specify which Fish Audio TTS model to use (e.g. "s1")
            output_format: Audio output format. Defaults to "pcm".
            sample_rate: Audio sample rate. If None, uses default.
            params: Additional input parameters for voice customization.
            **kwargs: Additional arguments passed to the parent service.
        """
        params = params or FishAudioTTSService.InputParams()

        # Validation for model and reference_id parameters
        if model and reference_id:
            raise ValueError(
                "Cannot specify both 'model' and 'reference_id'. Use 'reference_id' only."
            )

        if model is None and reference_id is None:
            raise ValueError("Must specify 'reference_id' (or deprecated 'model') parameter.")

        if model:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'model' is deprecated and will be removed in a future version. "
                    "Use 'reference_id' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            reference_id = model

        super().__init__(
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=FishAudioTTSSettings(
                model=model_id,
                voice=reference_id,
                fish_sample_rate=0,
                latency=params.latency,
                format=output_format,
                normalize=params.normalize,
                prosody_speed=params.prosody_speed,
                prosody_volume=params.prosody_volume,
                reference_id=reference_id,
            ),
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = "wss://api.fish.audio/v1/tts/live"
        self._websocket = None
        self._receive_task = None
        self._request_id = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Fish Audio service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if needed.

        Any change to voice or model triggers a WebSocket reconnect.

        Args:
            delta: A :class:`TTSSettings` (or ``FishAudioTTSSettings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    async def start(self, frame: StartFrame):
        """Start the Fish Audio TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings.fish_sample_rate = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Fish Audio TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Fish Audio TTS service.

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

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Fish Audio")
            headers = {"Authorization": f"Bearer {self._api_key}"}
            headers["model"] = self._settings.model
            self._websocket = await websocket_connect(self._base_url, additional_headers=headers)

            # Send initial start message with ormsgpack
            request_settings = {
                "sample_rate": self._settings.fish_sample_rate,
                "latency": self._settings.latency,
                "format": self._settings.format,
                "normalize": self._settings.normalize,
                "prosody": {
                    "speed": self._settings.prosody_speed,
                    "volume": self._settings.prosody_volume,
                },
                "reference_id": self._settings.reference_id,
            }
            start_message = {"event": "start", "request": {"text": "", **request_settings}}
            await self._websocket.send(ormsgpack.packb(start_message))
            logger.debug("Sent start message to Fish Audio")

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Fish Audio")
                # Send stop event with ormsgpack
                stop_message = {"event": "stop"}
                await self._websocket.send(ormsgpack.packb(stop_message))
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._request_id = None
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def flush_audio(self):
        """Flush any buffered audio by sending a flush event to Fish Audio."""
        logger.trace(f"{self}: Flushing audio buffers")
        if not self._websocket or self._websocket.state is State.CLOSED:
            return
        flush_message = {"event": "flush"}
        await self._get_websocket().send(ormsgpack.packb(flush_message))

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        await self.stop_all_metrics()
        self._request_id = None

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                if isinstance(message, bytes):
                    msg = ormsgpack.unpackb(message)
                    if isinstance(msg, dict):
                        event = msg.get("event")
                        if event == "audio":
                            audio_data = msg.get("audio")
                            # Only process larger chunks to remove msgpack overhead
                            if audio_data and len(audio_data) > 1024:
                                frame = TTSAudioRawFrame(audio_data, self.sample_rate, 1)
                                await self.push_frame(frame)
                                await self.stop_ttfb_metrics()
                                continue

            except Exception as e:
                await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Fish Audio's streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames and control frames for the synthesized speech.
        """
        logger.debug(f"{self}: Generating Fish TTS: [{text}]")
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if not self._request_id:
                await self.start_ttfb_metrics()
                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame(context_id=context_id)
                self._request_id = str(uuid.uuid4())

            # Send the text
            text_message = {
                "event": "text",
                "text": text,
            }
            try:
                await self._get_websocket().send(ormsgpack.packb(text_message))
                await self.start_tts_usage_metrics(text)

                # Send flush event to force audio generation
                flush_message = {"event": "flush"}
                await self._get_websocket().send(ormsgpack.packb(flush_message))
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()

            yield None

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
