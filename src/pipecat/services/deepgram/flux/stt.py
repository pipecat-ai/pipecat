#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram Flux speech-to-text service implementation."""

import json
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Deepgram Flux, you need to `pip install pipecat-ai[deepgram]`.")
    raise Exception(f"Missing module: {e}")


class FluxMessageType(str, Enum):
    """Deepgram Flux WebSocket message types.

    These are the top-level message types that can be received from the
    Deepgram Flux WebSocket connection.
    """

    RECEIVE_CONNECTED = "Connected"
    RECEIVE_FATAL_ERROR = "Error"
    TURN_INFO = "TurnInfo"


class FluxEventType(str, Enum):
    """Deepgram Flux TurnInfo event types.

    These events are contained within TurnInfo messages and indicate
    different stages of speech processing and turn detection.
    """

    START_OF_TURN = "StartOfTurn"
    TURN_RESUMED = "TurnResumed"
    END_OF_TURN = "EndOfTurn"
    EAGER_END_OF_TURN = "EagerEndOfTurn"
    UPDATE = "Update"


class DeepgramFluxSTTService(WebsocketSTTService):
    """Deepgram Flux speech-to-text service.

    Provides real-time speech recognition using Deepgram's WebSocket API with Flux capabilities.
    Supports configurable models, VAD events, and various audio processing options
    including advanced turn detection and EagerEndOfTurn events for improved conversational AI performance.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Deepgram Flux API.

        This class defines all available connection parameters for the Deepgram Flux API
        based on the official documentation.

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
        """

        eager_eot_threshold: Optional[float] = None
        eot_threshold: Optional[float] = None
        eot_timeout_ms: Optional[int] = None
        keyterm: list = []
        mip_opt_out: Optional[bool] = None
        tag: list = []

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.deepgram.com/v2/listen",
        sample_rate: Optional[int] = None,
        model: str = "flux-general-en",
        flux_encoding: str = "linear16",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Deepgram Flux STT service.

        Args:
            api_key: Deepgram API key for authentication. Required for API access.
            url: WebSocket URL for the Deepgram Flux API. Defaults to the preview endpoint.
            sample_rate: Audio sample rate in Hz. If None, uses the rate from params or 16000.
            model: Deepgram Flux model to use for transcription. Currently only supports "flux-general-en".
            flux_encoding: Audio encoding format required by Flux API. Must be "linear16".
                Raw signed little-endian 16-bit PCM encoding.
            params: InputParams instance containing detailed API configuration options.
                If None, default parameters will be used.
            **kwargs: Additional arguments passed to the parent WebsocketSTTService class.

        Examples:
            Basic usage with default parameters::

                stt = DeepgramFluxSTTService(api_key="your-api-key")

            Advanced usage with custom parameters::

                params = DeepgramFluxSTTService.InputParams(
                    eager_eot_threshold=0.5,
                    eot_threshold=0.8,
                    keyterm=["AI", "machine learning", "neural network"],
                    tag=["production", "voice-agent"]
                )
                stt = DeepgramFluxSTTService(
                    api_key="your-api-key",
                    model="flux-general-en",
                    params=params
                )
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._url = url
        self._model = model
        self._params = params or DeepgramFluxSTTService.InputParams()
        self._flux_encoding = flux_encoding
        # This is the currently only supported language
        self._language = Language.EN
        self._websocket_url = None
        self._receive_task = None

    async def _connect(self):
        """Connect to WebSocket and start background tasks.

        Establishes the WebSocket connection to the Deepgram Flux API and starts
        the background task for receiving transcription results.
        """
        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect from WebSocket and clean up tasks.

        Gracefully disconnects from the Deepgram Flux API, cancels background tasks,
        and cleans up resources to prevent memory leaks.
        """
        try:
            # Cancel background tasks BEFORE closing websocket
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=2.0)
                self._receive_task = None

            # Now close the websocket
            await self._disconnect_websocket()

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
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

            self._websocket = await websocket_connect(
                self._websocket_url,
                additional_headers={"Authorization": f"Token {self._api_key}"},
            )
            logger.debug("Connected to Deepgram Flux Websocket")
            await self._call_event_handler("on_connected")
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state.

        Closes the WebSocket connection to the Deepgram Flux API and stops all
        metrics collection. Handles disconnection errors gracefully.
        """
        try:
            await self.stop_all_metrics()

            if self._websocket:
                await self._send_close_stream()
                logger.debug("Disconnecting from Deepgram Flux Websocket")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _send_close_stream(self) -> None:
        """Sends a CloseStream control message to the Deepgram Flux WebSocket API.

        This signals to the server that no more audio data will be sent.
        """
        if self._websocket:
            logger.debug("Sending CloseStream message to Deepgram Flux")
            message = {"type": "CloseStream"}
            await self._websocket.send(json.dumps(message))

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Deepgram service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Deepgram Flux STT service.

        Initializes the service by constructing the WebSocket URL with all configured
        parameters and establishing the connection to begin transcription processing.

        Args:
            frame: The start frame containing initialization parameters and metadata.
        """
        await super().start(frame)

        url_params = [
            f"model={self._model}",
            f"sample_rate={self.sample_rate}",
            f"encoding={self._flux_encoding}",
        ]

        if self._params.eager_eot_threshold is not None:
            url_params.append(f"eager_eot_threshold={self._params.eager_eot_threshold}")

        if self._params.eot_threshold is not None:
            url_params.append(f"eot_threshold={self._params.eot_threshold}")

        if self._params.eot_timeout_ms is not None:
            url_params.append(f"eot_timeout_ms={self._params.eot_timeout_ms}")

        if self._params.mip_opt_out is not None:
            url_params.append(f"mip_opt_out={str(self._params.mip_opt_out).lower()}")

        # Add keyterm parameters (can have multiple)
        for keyterm in self._params.keyterm:
            url_params.append(f"keyterm={keyterm}")

        # Add tag parameters (can have multiple)
        for tag_value in self._params.tag:
            url_params.append(f"tag={tag_value}")

        self._websocket_url = f"{self._url}?{'&'.join(url_params)}"
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Deepgram Flux STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Deepgram Flux STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

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
            logger.error("Not connected to Deepgram Flux.")
            yield ErrorFrame("Not connected to Deepgram Flux.", fatal=True)
            return

        try:
            await self._websocket.send(audio)
        except Exception as e:
            logger.error(f"Failed to send audio to Flux: {e}")
            yield ErrorFrame(f"Failed to send audio to Flux:  {e}")
            return

        yield None

    async def start_metrics(self):
        """Start TTFB and processing metrics collection."""
        # TTFB (Time To First Byte) metrics are currently disabled for Deepgram Flux.
        # Ideally, TTFB should measure the time from when a user starts speaking
        # until we receive the first transcript. However, Deepgram Flux delivers
        # both the "user started speaking" event and the first transcript simultaneously,
        # making this timing measurement meaningless in this context.
        # await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

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

    def _validate_message(self, data: Dict[str, Any]) -> bool:
        """Validate basic message structure from Deepgram Flux.

        Ensures the received message has the expected structure before processing.

        Args:
            data: The parsed JSON message data to validate.

        Returns:
            True if the message structure is valid, False otherwise.
        """
        if not isinstance(data, dict):
            logger.warning("Message is not a dictionary")
            return False

        if "type" not in data:
            logger.warning("Message missing 'type' field")
            return False

        return True

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
                    logger.error(f"Error processing message: {e}")
                    # Error will be handled inside WebsocketService->_receive_task_handler
                    raise
            else:
                logger.warning(f"Received non-string message: {type(message)}")

    async def _handle_message(self, data: Dict[str, Any]):
        """Handle a parsed WebSocket message from Deepgram Flux.

        Routes messages to appropriate handlers based on their type. Validates
        message structure before processing.

        Args:
            data: The parsed JSON message data from the WebSocket.
        """
        if not self._validate_message(data):
            return

        message_type = data.get("type")

        try:
            flux_message_type = FluxMessageType(message_type)
        except ValueError:
            logger.debug(f"Unhandled message type: {message_type or 'unknown'}")
            return

        match flux_message_type:
            case FluxMessageType.RECEIVE_CONNECTED:
                await self._handle_connection_established()
            case FluxMessageType.RECEIVE_FATAL_ERROR:
                await self._handle_fatal_error(data)
            case FluxMessageType.TURN_INFO:
                await self._handle_turn_info(data)

    async def _handle_connection_established(self):
        """Handle successful connection establishment to Deepgram Flux.

        This event is fired when the WebSocket connection to Deepgram Flux
        is successfully established and ready to receive audio data for
        transcription processing.
        """
        logger.info("Connected to Flux - ready to stream audio")

    async def _handle_fatal_error(self, data: Dict[str, Any]):
        """Handle fatal error messages from Deepgram Flux.

        Fatal errors indicate unrecoverable issues with the connection or
        configuration that require intervention. These errors will cause
        the connection to be terminated.

        Args:
            data: The error message data containing error details.

        Raises:
            Exception: Always raises to trigger error handling in the parent service.
        """
        error_msg = data.get("error", "Unknown error")
        deepgram_error = f"Fatal error: {error_msg}"
        logger.error(deepgram_error)
        # Error will be handled inside WebsocketService->_receive_task_handler
        raise Exception(deepgram_error)

    async def _handle_turn_info(self, data: Dict[str, Any]):
        """Handle TurnInfo events from Deepgram Flux.

        TurnInfo messages contain various turn-based events that indicate
        the state of speech processing, including turn boundaries, interim
        results, and turn finalization events.

        Args:
            data: The TurnInfo message data containing event type, transcript and some extra metadata.
        """
        event = data.get("event")
        transcript = data.get("transcript", "")

        try:
            flux_event_type = FluxEventType(event)
        except ValueError:
            logger.debug(f"Unhandled TurnInfo event: {event}")
            return

        match flux_event_type:
            case FluxEventType.START_OF_TURN:
                await self._handle_start_of_turn(transcript)
            case FluxEventType.TURN_RESUMED:
                await self._handle_turn_resumed(event)
            case FluxEventType.END_OF_TURN:
                await self._handle_end_of_turn(transcript, data)
            case FluxEventType.EAGER_END_OF_TURN:
                await self._handle_eager_end_of_turn(transcript, data)
            case FluxEventType.UPDATE:
                await self._handle_update(transcript)

    async def _handle_start_of_turn(self, transcript: str):
        """Handle StartOfTurn events from Deepgram Flux.

        StartOfTurn events are fired when Deepgram Flux detects the beginning
        of a new speaking turn. This triggers bot interruption to stop any
        ongoing speech synthesis and signals the start of user speech detection.

        The service will:
        - Send a BotInterruptionFrame upstream to stop bot speech
        - Send a UserStartedSpeakingFrame downstream to notify other components
        - Start metrics collection for measuring response times

        Args:
            transcript: maybe the first few words of the turn.
        """
        logger.debug("User started speaking")
        await self.push_interruption_task_frame_and_wait()
        await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
        await self.start_metrics()
        if transcript:
            logger.trace(f"Start of turn transcript: {transcript}")

    async def _handle_turn_resumed(self, event: str):
        """Handle TurnResumed events from Deepgram Flux.

        TurnResumed events indicate that speech has resumed after a brief pause
        within the same turn. This is primarily used for logging and debugging
        purposes and doesn't trigger any significant processing changes.

        Args:
            event: The event type string for logging purposes.
        """
        logger.trace(f"Received event TurnResumed: {event}")

    async def _handle_end_of_turn(self, transcript: str, data: Dict[str, Any]):
        """Handle EndOfTurn events from Deepgram Flux.

        EndOfTurn events are fired when Deepgram Flux determines that a speaking
        turn has concluded, either due to sufficient silence or end-of-turn
        confidence thresholds being met. This provides the final transcript
        for the completed turn.

        The service will:
        - Create and send a final TranscriptionFrame with the complete transcript
        - Trigger transcription handling with tracing for metrics
        - Stop processing metrics collection
        - Send a UserStoppedSpeakingFrame to signal turn completion

        Args:
            transcript: The final transcript text for the completed turn.
            data: The TurnInfo message data containing event type, transcript and some extra metadata.
        """
        logger.debug("User stopped speaking")

        await self.push_frame(
            TranscriptionFrame(
                transcript,
                self._user_id,
                time_now_iso8601(),
                self._language,
                result=data,
            )
        )
        await self._handle_transcription(transcript, True, self._language)
        await self.stop_processing_metrics()
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def _handle_eager_end_of_turn(self, transcript: str, data: Dict[str, Any]):
        """Handle EagerEndOfTurn events from Deepgram Flux.

        EagerEndOfTurn events are fired when the end-of-turn confidence reaches the
        EagerEndOfTurn threshold but hasn't yet reached the full end-of-turn threshold.
        These provide interim transcripts that can be used for faster response
        generation while still allowing the user to continue speaking.

        EagerEndOfTurn events enable more responsive conversational AI by allowing
        the LLM to start processing likely final transcripts before the turn
        is definitively ended.

        Args:
            transcript: The interim transcript text that triggered the EagerEndOfTurn event.
            data: The TurnInfo message data containing event type, transcript and some extra metadata.
        """
        logger.trace(f"EagerEndOfTurn - {transcript}")
        # Deepgram's EagerEndOfTurn feature enables lower-latency voice agents by sending
        # medium-confidence transcripts before EndOfTurn certainty, allowing LLM processing to
        # begin early.
        #
        # However, if speech resumes or the transcripts differ from the final EndOfTurn, the
        # EagerEndOfTurn response should be cancelled to avoid incorrect or partial responses.
        #
        # Pipecat doesn't yet provide built-in Gate/control mechanisms to:
        # 1. Start LLM/TTS processing early on EagerEndOfTurn events
        # 2. Cancel in-flight processing when TurnResumed occurs
        #
        # By pushing EagerEndOfTurn transcripts as InterimTranscriptionFrame, we enable
        # developers to implement custom EagerEndOfTurn handling in their applications while
        # maintaining compatibility with existing interim transcription workflows.
        #
        # TODO: Implement proper EagerEndOfTurn support with cancellable processing pipeline
        # that can start response generation on EagerEndOfTurn and cancel or confirm it.
        await self.push_frame(
            InterimTranscriptionFrame(
                transcript,
                self._user_id,
                time_now_iso8601(),
                self._language,
                result=data,
            )
        )

    async def _handle_update(self, transcript: str):
        """Handle Update events from Deepgram Flux.

        Update events provide incremental transcript updates during an ongoing
        turn. These events allow for real-time display of transcription progress
        and can be used to provide visual feedback to users about what's being
        recognized.

        The service stops TTFB (Time To First Byte) metrics when the first
        substantial update is received, indicating successful processing start.

        Args:
            transcript: The current partial transcript text for the ongoing turn.
        """
        if transcript:
            logger.trace(f"Update event: {transcript}")
            # TTFB (Time To First Byte) metrics are currently disabled for Deepgram Flux.
            # Ideally, TTFB should measure the time from when a user starts speaking
            # until we receive the first transcript. However, Deepgram Flux delivers
            # both the "user started speaking" event and the first transcript simultaneously,
            # making this timing measurement meaningless in this context.
            # await self.stop_ttfb_metrics()
