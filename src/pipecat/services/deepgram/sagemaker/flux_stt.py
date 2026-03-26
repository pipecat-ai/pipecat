#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram Flux speech-to-text service for AWS SageMaker.

This module provides a Pipecat STT service that connects to Deepgram Flux models
deployed on AWS SageMaker endpoints. Uses HTTP/2 bidirectional streaming for
low-latency real-time transcription with advanced turn detection (StartOfTurn,
EndOfTurn, EagerEndOfTurn, TurnResumed).
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional
from urllib.parse import urlencode

from loguru import logger

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
from pipecat.services.aws.sagemaker.bidi_client import SageMakerBidiClient
from pipecat.services.deepgram.flux.stt import (
    DeepgramFluxSTTSettings,
    FluxEventType,
    FluxMessageType,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


@dataclass
class DeepgramFluxSageMakerSTTSettings(DeepgramFluxSTTSettings):
    """Settings for the Deepgram Flux SageMaker STT service.

    Inherits all fields from :class:`DeepgramFluxSTTSettings`.
    """

    pass


class DeepgramFluxSageMakerSTTService(STTService):
    """Deepgram Flux speech-to-text service for AWS SageMaker.

    Provides real-time speech recognition using Deepgram Flux models deployed on
    AWS SageMaker endpoints. Uses HTTP/2 bidirectional streaming for low-latency
    transcription with advanced turn detection (StartOfTurn, EndOfTurn,
    EagerEndOfTurn, TurnResumed).

    Unlike the Nova-based SageMaker STT service, Flux handles turn detection
    natively, so no external VAD is needed for turn boundaries. Use
    ``ExternalUserTurnStrategies`` in your pipeline.

    Requirements:

    - AWS credentials configured (via environment variables, AWS CLI, or instance metadata)
    - A deployed SageMaker endpoint with Deepgram Flux model

    Event handlers available:

    - on_connected: Called when the SageMaker session is established
    - on_disconnected: Called when the session is closed
    - on_connection_error: Called on connection failure
    - on_start_of_turn: Deepgram Flux detected start of speech
    - on_end_of_turn: Deepgram Flux detected end of turn
    - on_eager_end_of_turn: Deepgram Flux predicted end of turn
    - on_turn_resumed: User resumed speaking after EagerEndOfTurn
    - on_update: Interim transcript update during a turn

    Example::

        stt = DeepgramFluxSageMakerSTTService(
            endpoint_name="my-deepgram-flux-endpoint",
            region="us-east-2",
            settings=DeepgramFluxSageMakerSTTService.Settings(
                model="flux-general-en",
                eot_threshold=0.7,
                eager_eot_threshold=0.5,
            ),
        )
    """

    Settings = DeepgramFluxSageMakerSTTSettings
    _settings: Settings
    _CONFIGURE_FIELDS = {"keyterm", "eot_threshold", "eager_eot_threshold", "eot_timeout_ms"}

    def __init__(
        self,
        *,
        endpoint_name: str,
        region: str,
        encoding: str = "linear16",
        sample_rate: Optional[int] = None,
        mip_opt_out: Optional[bool] = None,
        tag: Optional[list] = None,
        should_interrupt: bool = True,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the Deepgram Flux SageMaker STT service.

        Args:
            endpoint_name: Name of the SageMaker endpoint with Deepgram Flux model
                deployed (e.g., "my-deepgram-flux-endpoint").
            region: AWS region where the endpoint is deployed (e.g., "us-east-2").
            encoding: Audio encoding format. Defaults to "linear16".
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                sample rate.
            mip_opt_out: Opt out of Deepgram model improvement program.
            tag: Tags to label requests for identification during usage reporting.
            should_interrupt: Whether to interrupt the bot when Flux detects that
                the user is speaking. Defaults to True.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent STTService.
        """
        # Initialize default settings
        default_settings = self.Settings(
            model="flux-general-en",
            language=Language.EN,
            eager_eot_threshold=None,
            eot_threshold=None,
            eot_timeout_ms=None,
            keyterm=[],
            min_confidence=None,
        )

        # Apply settings delta
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._endpoint_name = endpoint_name
        self._region = region
        self._encoding = encoding
        self._mip_opt_out = mip_opt_out
        self._tag = tag or []
        self._should_interrupt = should_interrupt

        self._client: Optional[SageMakerBidiClient] = None
        self._response_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None

        # Watchdog state
        self._last_stt_time: Optional[float] = None
        self._user_is_speaking = False

        # Connection readiness: Flux sends a "Connected" message when ready
        self._connection_established_event = asyncio.Event()

        # Flux event handlers
        self._register_event_handler("on_start_of_turn")
        self._register_event_handler("on_turn_resumed")
        self._register_event_handler("on_end_of_turn")
        self._register_event_handler("on_eager_end_of_turn")
        self._register_event_handler("on_update")

    def _build_query_string(self) -> str:
        """Build query string from current settings and init-only connection config."""
        params = []

        s = self._settings

        params.append(f"model={s.model}")
        params.append(f"sample_rate={self.sample_rate}")
        params.append(f"encoding={self._encoding}")

        if s.eager_eot_threshold is not None:
            params.append(f"eager_eot_threshold={s.eager_eot_threshold}")

        if s.eot_threshold is not None:
            params.append(f"eot_threshold={s.eot_threshold}")

        if s.eot_timeout_ms is not None:
            params.append(f"eot_timeout_ms={s.eot_timeout_ms}")

        if self._mip_opt_out is not None:
            params.append(f"mip_opt_out={str(self._mip_opt_out).lower()}")

        # Add keyterm parameters (can have multiple)
        for keyterm in s.keyterm:
            params.append(urlencode({"keyterm": keyterm}))

        # Add tag parameters (can have multiple)
        for tag_value in self._tag:
            params.append(urlencode({"tag": tag_value}))

        return "&".join(params)

    async def _connect(self):
        """Connect to the SageMaker endpoint and start the BiDi session.

        Starts the HTTP/2 session and waits for the Flux ``Connected`` message
        before returning, ensuring audio is not sent before the model is ready.
        """
        logger.debug("Connecting to Deepgram Flux on SageMaker...")

        query_string = self._build_query_string()

        self._connection_established_event.clear()

        self._client = SageMakerBidiClient(
            endpoint_name=self._endpoint_name,
            region=self._region,
            model_invocation_path="v2/listen",
            model_query_string=query_string,
        )

        try:
            await self._client.start_session()

            # Start response processor first so we can receive the Connected message
            self._response_task = self.create_task(self._process_responses())

            # Wait for Flux to confirm the connection is ready
            logger.debug("SageMaker session started, waiting for Flux connection confirmation...")
            await self._connection_established_event.wait()

            # Note: Flux does not support KeepAlive messages (only CloseStream and
            # Configure are valid). The watchdog task handles keeping the connection
            # alive by sending silence when needed.
            self._watchdog_task = self.create_task(self._watchdog_task_handler())

            logger.debug("Connected to Deepgram Flux on SageMaker")
            await self._call_event_handler("on_connected")

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect(self):
        """Disconnect from the SageMaker endpoint."""
        self._connection_established_event.clear()

        if self._client and self._client.is_active:
            logger.debug("Disconnecting from Deepgram Flux on SageMaker...")

            await self._send_close_stream()

            if self._watchdog_task and not self._watchdog_task.done():
                await self.cancel_task(self._watchdog_task)
                self._watchdog_task = None
                self._last_stt_time = None

            if self._response_task and not self._response_task.done():
                await self.cancel_task(self._response_task)

            await self._client.close_session()

            logger.debug("Disconnected from Deepgram Flux on SageMaker")
            await self._call_event_handler("on_disconnected")

    async def _send_silence(self, duration_secs: float = 0.5):
        """Send a block of silence of the specified duration (default 500 ms)."""
        sample_width = 2  # bytes per sample for 16-bit PCM
        num_channels = 1  # mono
        num_samples = int(self.sample_rate * duration_secs)
        silence = b"\x00" * (num_samples * sample_width * num_channels)
        await self._client.send_audio_chunk(silence)

    async def _watchdog_task_handler(self):
        """Prevent dangling turns by sending silence when audio stops flowing.

        If we stop sending audio to Flux after receiving a StartOfTurn,
        we never receive the UserStoppedSpeaking event unless we resume
        sending audio.
        """
        while self._client and self._client.is_active:
            now = time.monotonic()
            # More than 500 ms without sending new audio to Flux
            if self._user_is_speaking and self._last_stt_time and now - self._last_stt_time > 0.5:
                logger.warning("Sending silence to Flux to prevent dangling task")
                try:
                    await self._send_silence()
                except Exception as e:
                    logger.warning(f"Failed to send silence: {e}")
                self._last_stt_time = time.monotonic()
            # check every 100ms
            await asyncio.sleep(0.1)

    async def _send_close_stream(self) -> None:
        """Sends a CloseStream control message to the Deepgram Flux SageMaker endpoint.

        This signals to the server that no more audio data will be sent.
        """
        try:
            if self._client and self._client.is_active:
                logger.debug("Sending CloseStream message to Deepgram Flux on SageMaker")
                await self._client.send_json({"type": "CloseStream"})
        except Exception as e:
            await self.push_error(error_msg=f"Error sending CloseStream: {e}", exception=e)

    async def _send_configure(self, fields: set[str]):
        """Send a Configure control message to update settings mid-stream.

        Args:
            fields: Set of changed field names to include in the message.
        """
        message: dict[str, Any] = {"type": "Configure"}

        if "keyterm" in fields:
            message["keyterms"] = self._settings.keyterm

        thresholds: dict[str, Any] = {}
        if "eot_threshold" in fields:
            thresholds["eot_threshold"] = self._settings.eot_threshold
        if "eager_eot_threshold" in fields:
            thresholds["eager_eot_threshold"] = self._settings.eager_eot_threshold
        if "eot_timeout_ms" in fields:
            thresholds["eot_timeout_ms"] = self._settings.eot_timeout_ms
        if thresholds:
            message["thresholds"] = thresholds

        logger.debug(f"{self}: sending Configure message: {message}")
        await self._client.send_json(message)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Deepgram Flux SageMaker service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Configure-able fields (keyterm, eot_threshold, eager_eot_threshold,
        eot_timeout_ms) are sent to Deepgram via a Configure message.
        Other fields are stored but cannot be applied to the active connection.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        configure_fields = changed.keys() & self._CONFIGURE_FIELDS
        if configure_fields and self._client and self._client.is_active:
            await self._send_configure(configure_fields)

        self._warn_unhandled_updated_settings(changed.keys() - self._CONFIGURE_FIELDS)

        return changed

    async def start(self, frame: StartFrame):
        """Start the Deepgram Flux SageMaker STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Deepgram Flux SageMaker STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Deepgram Flux SageMaker STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Deepgram Flux for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via BiDi stream callbacks).
        """
        if not self._connection_established_event.is_set():
            return

        if self._client and self._client.is_active:
            try:
                self._last_stt_time = time.monotonic()
                await self._client.send_audio_chunk(audio)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
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

    def _validate_message(self, data: Dict[str, Any]) -> bool:
        """Validate basic message structure from Deepgram Flux.

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

    async def _process_responses(self):
        """Process streaming responses from Deepgram Flux on SageMaker."""
        try:
            while self._client and self._client.is_active:
                result = await self._client.receive_response()

                if result is None:
                    break

                if hasattr(result, "value") and hasattr(result.value, "bytes_"):
                    if result.value.bytes_:
                        response_data = result.value.bytes_.decode("utf-8")

                        try:
                            parsed = json.loads(response_data)
                            await self._handle_message(parsed)
                        except json.JSONDecodeError:
                            logger.warning(f"Non-JSON response: {response_data}")

        except asyncio.CancelledError:
            logger.debug("Response processor cancelled")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            logger.debug("Response processor stopped")

    async def _handle_message(self, data: Dict[str, Any]):
        """Handle a parsed message from Deepgram Flux.

        Routes messages to appropriate handlers based on their type.

        Args:
            data: The parsed JSON message data.
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
            case FluxMessageType.CONFIGURE_SUCCESS:
                logger.info(f"{self}: Configure accepted: {data}")
            case FluxMessageType.CONFIGURE_FAILURE:
                error_code = data.get("error_code", "unknown")
                description = data.get("description", "no description")
                error_msg = f"Configure rejected: [{error_code}] {description}"
                logger.warning(f"{self}: {error_msg}")
                await self.push_error(error_msg=error_msg)

    async def _handle_connection_established(self):
        """Handle successful connection establishment to Deepgram Flux.

        This event is fired when the WebSocket connection to Deepgram Flux
        is successfully established and ready to receive audio data for
        transcription processing.
        """
        logger.info("Connected to Flux - ready to stream audio")
        # Notify connection is established
        self._connection_established_event.set()

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
        self._user_is_speaking = True
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.broadcast_interruption()
        await self.start_metrics()
        await self._call_event_handler("on_start_of_turn", transcript)
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
        await self._call_event_handler("on_turn_resumed")

    def _calculate_average_confidence(self, transcript_data) -> Optional[float]:
        """Calculate the average confidence from transcript data.

        Return None if the data is missing or invalid.
        """
        # Example: Assume transcript_data has a list of words with confidence
        words = transcript_data.get("words")
        if not words or not isinstance(words, list):
            return None
        confidences = [
            w.get("confidence") for w in words if isinstance(w.get("confidence"), (float, int))
        ]
        if not confidences:
            return None
        return sum(confidences) / len(confidences)

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
        self._user_is_speaking = False

        # Compute the average confidence
        average_confidence = self._calculate_average_confidence(data)

        if not self._settings.min_confidence or average_confidence > self._settings.min_confidence:
            # EndOfTurn means Flux has determined the turn is complete,
            # so this TranscriptionFrame is always finalized
            await self.push_frame(
                TranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    self._settings.language,
                    result=data,
                    finalized=True,
                )
            )
        else:
            logger.warning(
                f"Transcription confidence below min_confidence threshold: {average_confidence}"
            )

        await self._handle_transcription(transcript, True, self._settings.language)
        await self.stop_processing_metrics()
        await self.broadcast_frame(UserStoppedSpeakingFrame)
        await self._call_event_handler("on_end_of_turn", transcript)

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
                self._settings.language,
                result=data,
            )
        )
        await self._call_event_handler("on_eager_end_of_turn", transcript)

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
            await self._call_event_handler("on_update", transcript)
