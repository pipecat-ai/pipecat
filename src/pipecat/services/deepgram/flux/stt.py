#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram Flux speech-to-text service implementation (WebSocket transport)."""

import json
import time
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
)
from pipecat.services.deepgram.flux.base import (
    DeepgramFluxSTTBase,
    DeepgramFluxSTTSettings,
    FluxEventType,
    FluxMessageType,
)
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Deepgram Flux, you need to `pip install pipecat-ai[deepgram]`.")
    raise Exception(f"Missing module: {e}")

# Re-export for backward compatibility
__all__ = [
    "DeepgramFluxSTTService",
    "DeepgramFluxSTTSettings",
    "FluxEventType",
    "FluxMessageType",
]


class DeepgramFluxSTTService(DeepgramFluxSTTBase, WebsocketService):
    """Deepgram Flux speech-to-text service.

    Provides real-time speech recognition using Deepgram's WebSocket API with Flux capabilities.
    Supports configurable models, VAD events, and various audio processing options
    including advanced turn detection and EagerEndOfTurn events for improved conversational AI performance.

    Event handlers available (in addition to base events):

    - on_start_of_turn(service, transcript): Deepgram detected start of speech
    - on_end_of_turn(service, transcript): Deepgram detected end of turn (EOT)
    - on_eager_end_of_turn(service, transcript): Deepgram predicted end of turn (EagerEOT)
    - on_turn_resumed(service): User resumed speaking after EagerEOT

    Example::

        @stt.event_handler("on_end_of_turn")
        async def on_end_of_turn(service, transcript):
            ...
    """

    Settings = DeepgramFluxSTTSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Configuration parameters for Deepgram Flux API.

        .. deprecated:: 0.0.105
            Use ``settings=DeepgramFluxSTTService.Settings(...)`` instead.

        Parameters:
            eager_eot_threshold: Optional. EagerEndOfTurn/TurnResumed are off by default.
                You can turn them on by setting eager_eot_threshold to a valid value.
                Lower values = more aggressive EagerEndOfTurning (faster response, more LLM calls).
                Higher values = more conservative EagerEndOfTurning (slower response, fewer LLM calls).
            eot_threshold: Optional. End-of-turn confidence required to finish a turn (default 0.7).
                Lower values = turns end sooner (more interruptions, faster responses).
                Higher values = turns end later (fewer interruptions, more complete utterances).
            eot_timeout_ms: Optional. Time in milliseconds after speech to finish a turn
                regardless of EOT confidence (default 5000).
            keyterm: List of keyterms to boost recognition accuracy for specialized terminology.
            mip_opt_out: Optional. Opts out requests from the Deepgram Model Improvement Program
                (default False).
            tag: List of tags to label requests for identification during usage reporting.
            min_confidence: Optional. Minimum confidence required confidence to create a TranscriptionFrame
        """

        eager_eot_threshold: Optional[float] = None
        eot_threshold: Optional[float] = None
        eot_timeout_ms: Optional[int] = None
        keyterm: list = []
        mip_opt_out: Optional[bool] = None
        tag: list = []
        min_confidence: Optional[float] = None  # New parameter

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.deepgram.com/v2/listen",
        sample_rate: Optional[int] = None,
        mip_opt_out: Optional[bool] = None,
        model: Optional[str] = None,
        flux_encoding: str = "linear16",
        tag: Optional[list] = None,
        params: Optional[InputParams] = None,
        should_interrupt: bool = True,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the Deepgram Flux STT service.

        Args:
            api_key: Deepgram API key for authentication. Required for API access.
            url: WebSocket URL for the Deepgram Flux API. Defaults to the preview endpoint.
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                sample rate.
            mip_opt_out: Opt out of the Deepgram Model Improvement Program.
            model: Deepgram Flux model to use for transcription.

                .. deprecated:: 0.0.105
                    Use ``settings=DeepgramFluxSTTService.Settings(model=...)`` instead.

            flux_encoding: Audio encoding format required by Flux API. Must be "linear16".
                Raw signed little-endian 16-bit PCM encoding.
            tag: Tags to label requests for identification during usage reporting.
            params: InputParams instance containing detailed API configuration options.

                .. deprecated:: 0.0.105
                    Use ``settings=DeepgramFluxSTTService.Settings(...)`` instead.

            should_interrupt: Determine whether the bot should be interrupted when Flux detects that the user is speaking.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to the parent classes.

        Examples:
            Basic usage with default parameters::

                stt = DeepgramFluxSTTService(api_key="your-api-key")

            Advanced usage with custom parameters::

                stt = DeepgramFluxSTTService(
                    api_key="your-api-key",
                    settings=DeepgramFluxSTTService.Settings(
                        model="flux-general-en",
                        eager_eot_threshold=0.5,
                        eot_threshold=0.8,
                        keyterm=["AI", "machine learning", "neural network"],
                        tag=["production", "voice-agent"],
                    ),
                )
        """
        # Note: For DeepgramFluxSTTService, differently from other processes, we need to create
        # the _receive_task inside _connect_websocket, because the websocket should only be
        # considered connected and ready to send audio once we receive from Flux the message
        # which confirms the connection has been established.
        # If we try to keep the logic reconnect_on_error, when receiving a message, the
        # _receive_task_handler would try to reconnect in case of error, invoking the
        # _connect_websocket again and leading to a case where the first _receive_task_handler
        # was never destroyed.
        # So we can keep it here as false, because inside the method send_with_retry, it will
        # already try to reconnect if needed.

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="flux-general-en",
            language=Language.EN,
            eager_eot_threshold=None,
            eot_threshold=None,
            eot_timeout_ms=None,
            keyterm=[],
            min_confidence=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.eager_eot_threshold = params.eager_eot_threshold
                default_settings.eot_threshold = params.eot_threshold
                default_settings.eot_timeout_ms = params.eot_timeout_ms
                default_settings.keyterm = params.keyterm or []
                if params.tag and tag is None:
                    tag = params.tag
                default_settings.min_confidence = params.min_confidence
                if params.mip_opt_out is not None:
                    mip_opt_out = params.mip_opt_out

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        DeepgramFluxSTTBase.__init__(
            self,
            encoding=flux_encoding,
            mip_opt_out=mip_opt_out,
            tag=tag,
            should_interrupt=should_interrupt,
            settings=default_settings,
            sample_rate=sample_rate,
            **kwargs,
        )
        WebsocketService.__init__(self, reconnect_on_error=False)

        self._api_key = api_key
        self._url = url
        self._websocket_url = None
        self._receive_task = None

    # ------------------------------------------------------------------
    # Transport interface implementation
    # ------------------------------------------------------------------

    async def _transport_send_audio(self, audio: bytes):
        await self._websocket.send(audio)

    async def _transport_send_json(self, message: dict):
        await self._websocket.send(json.dumps(message))

    def _transport_is_active(self) -> bool:
        return self._websocket is not None and self._websocket.state is State.OPEN

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        """Connect to WebSocket and start background tasks.

        Establishes the WebSocket connection to the Deepgram Flux API and starts
        the background task for receiving transcription results.
        """
        await super()._connect()
        self._websocket_url = f"{self._url}?{self._build_query_string()}"
        await self._connect_websocket()

    async def _disconnect(self):
        """Disconnect from WebSocket and clean up tasks.

        Gracefully disconnects from the Deepgram Flux API, cancels background tasks,
        and cleans up resources to prevent memory leaks.
        """
        await super()._disconnect()

        try:
            await self._disconnect_websocket()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            # Reset state only after everything is cleaned up
            self._websocket = None

    async def _connect_websocket(self):
        """Establish WebSocket connection to API.

        Creates a WebSocket connection to the Deepgram Flux API using the configured
        URL and authentication headers. Handles connection errors and reports them
        through the event handler system.
        """
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._connection_established_event.clear()
            self._user_is_speaking = False
            self._websocket = await websocket_connect(
                self._websocket_url,
                additional_headers={"Authorization": f"Token {self._api_key}"},
            )

            headers = {
                k: v for k, v in self._websocket.response.headers.items() if k.startswith("dg-")
            }
            logger.debug(f'{self}: Websocket connection initialized: {{"headers": {headers}}}')

            # Creating the receiver task
            if not self._receive_task:
                self._receive_task = self.create_task(
                    self._receive_task_handler(self._report_error)
                )

            # Creating the watchdog task
            if not self._watchdog_task:
                self._watchdog_task = self.create_task(self._watchdog_task_handler())

            # Now wait for the connection established event
            logger.debug("WebSocket connected, waiting for server confirmation...")
            await self._connection_established_event.wait()
            logger.debug("Connected to Deepgram Flux Websocket")
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state.

        Closes the WebSocket connection to the Deepgram Flux API and stops all
        metrics collection. Handles disconnection errors gracefully.
        """
        try:
            # Cancel background tasks BEFORE closing websocket
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=2.0)
                self._receive_task = None
            if self._watchdog_task:
                await self.cancel_task(self._watchdog_task, timeout=2.0)
                self._watchdog_task = None
                self._last_stt_time = None

            self._connection_established_event.clear()
            await self.stop_all_metrics()

            if self._websocket:
                await self._send_close_stream()
                logger.debug("Disconnecting from Deepgram Flux Websocket")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    # ------------------------------------------------------------------
    # Audio sending and receiving
    # ------------------------------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Deepgram Flux for transcription.

        Transmits raw audio bytes to the Deepgram Flux API for real-time speech
        recognition. Transcription results are received asynchronously through
        WebSocket callbacks and processed in the background.

        Args:
            audio: Raw audio bytes in linear16 format (signed little-endian 16-bit PCM).

        Yields:
            Frame: None (transcription results are delivered via WebSocket callbacks
                rather than as return values from this method).

        Raises:
            Exception: If the WebSocket connection is not established or if there
                are issues sending the audio data.
        """
        if not self._websocket:
            return

        try:
            self._last_stt_time = time.monotonic()
            await self.send_with_retry(audio, self._report_error)
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
            return

        yield None

    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns the active WebSocket connection instance, raising an exception
        if no connection is currently established.

        Returns:
            The active WebSocket connection instance.

        Raises:
            Exception: If no WebSocket connection is currently active.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and process messages from WebSocket.

        Continuously receives messages from the Deepgram Flux WebSocket connection
        and processes various message types including connection status, transcription
        results, turn information, and error conditions. Handles different event types
        such as StartOfTurn, EndOfTurn, EagerEndOfTurn, and Update events.
        """
        async for message in self._get_websocket():
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON message: {e}")
                    # Skip malformed messages
                    continue
                except Exception as e:
                    await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
                    # Error will be handled inside WebsocketService->_receive_task_handler
                    raise
            else:
                logger.warning(f"Received non-string message: {type(message)}")

    async def _report_error(self, error):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error_frame(error)
