#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram Flux STT base class shared across transports (WebSocket, SageMaker, etc.)."""

import asyncio
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from urllib.parse import urlencode

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


def language_to_deepgram_flux_language(language: Language) -> str | None:
    """Convert a Pipecat Language to a Deepgram Flux language code.

    Only honored by the ``flux-general-multi`` model. Locale variants
    (e.g. ``Language.EN_GB``) fall back to the base code.
    """
    LANGUAGE_MAP = {
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.IT: "it",
        Language.JA: "ja",
        Language.NL: "nl",
        Language.PT: "pt",
        Language.RU: "ru",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


def _prepare_language_hints(hints: list[Language] | None) -> list[str]:
    """Convert a list of Pipecat Languages to Deepgram Flux codes.

    Drops entries that can't be mapped and deduplicates while preserving order.
    """
    if not hints:
        return []
    seen: set[str] = set()
    prepared: list[str] = []
    for hint in hints:
        code = language_to_deepgram_flux_language(hint)
        if code is None or code in seen:
            continue
        seen.add(code)
        prepared.append(code)
    return prepared


def _code_to_pipecat_language(code: str) -> Language | None:
    """Convert a Deepgram-returned language code to a Pipecat Language."""
    try:
        return Language(code)
    except ValueError:
        logger.debug(f"Unmapped Deepgram Flux detected language code: {code}")
        return None


class FluxMessageType(StrEnum):
    """Deepgram Flux WebSocket message types.

    These are the top-level message types that can be received from the
    Deepgram Flux WebSocket connection.
    """

    RECEIVE_CONNECTED = "Connected"
    RECEIVE_FATAL_ERROR = "Error"
    TURN_INFO = "TurnInfo"
    CONFIGURE_SUCCESS = "ConfigureSuccess"
    CONFIGURE_FAILURE = "ConfigureFailure"


class FluxEventType(StrEnum):
    """Deepgram Flux TurnInfo event types.

    These events are contained within TurnInfo messages and indicate
    different stages of speech processing and turn detection.
    """

    START_OF_TURN = "StartOfTurn"
    TURN_RESUMED = "TurnResumed"
    END_OF_TURN = "EndOfTurn"
    EAGER_END_OF_TURN = "EagerEndOfTurn"
    UPDATE = "Update"


@dataclass
class DeepgramFluxSTTSettings(STTSettings):
    """Settings for DeepgramFluxSTTService.

    Parameters:
        eager_eot_threshold: EagerEndOfTurn/TurnResumed threshold. Off by default.
            Lower values = more aggressive (faster response, more LLM calls).
            Higher values = more conservative (slower response, fewer LLM calls).
        eot_threshold: End-of-turn confidence required to finish a turn (default 0.7).
        eot_timeout_ms: Time in ms after speech to finish a turn regardless of EOT
            confidence (default 5000).
        keyterm: Keyterms to boost recognition accuracy for specialized terminology.
        min_confidence: Minimum confidence required to create a TranscriptionFrame.
        language_hints: Languages to bias transcription toward. Only honored by the
            ``flux-general-multi`` model. An empty list clears any active hints;
            ``None``/``NOT_GIVEN`` means no hints (auto-detect). Can be updated
            mid-stream via ``STTUpdateSettingsFrame``.
    """

    eager_eot_threshold: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    eot_threshold: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    eot_timeout_ms: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    keyterm: list | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_confidence: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    language_hints: list[Language] | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class DeepgramFluxSTTBase(STTService):
    """Base class for Deepgram Flux STT services across transports.

    Contains all shared Flux protocol logic (message handling, turn detection,
    metrics, settings). Concrete subclasses implement the transport layer by
    providing three abstract primitives: ``_transport_send_audio``,
    ``_transport_send_json``, and ``_transport_is_active``.
    """

    Settings = DeepgramFluxSTTSettings
    _settings: Settings
    _CONFIGURE_FIELDS = {
        "keyterm",
        "eot_threshold",
        "eager_eot_threshold",
        "eot_timeout_ms",
        "language_hints",
    }
    _MULTILINGUAL_MODEL = "flux-general-multi"

    def __init__(
        self,
        *,
        encoding: str = "linear16",
        mip_opt_out: bool | None = None,
        tag: list | None = None,
        should_interrupt: bool = True,
        settings: Settings,
        **kwargs,
    ):
        """Initialize the Deepgram Flux STT base service.

        Args:
            encoding: Audio encoding format. Must be "linear16".
            mip_opt_out: Opt out of the Deepgram Model Improvement Program.
            tag: Tags to label requests for identification during usage reporting.
            should_interrupt: Whether to interrupt the bot when Flux detects that
                the user is speaking.
            settings: Fully resolved settings instance (built by concrete subclass).
            **kwargs: Additional arguments passed to the parent STTService (e.g.
                ``sample_rate``, ``reconnect_on_error``).
        """
        super().__init__(settings=settings, **kwargs)

        self._encoding = encoding
        self._mip_opt_out = mip_opt_out
        self._tag = tag or []
        self._should_interrupt = should_interrupt

        # Connection readiness: Flux sends a "Connected" message when ready
        self._connection_established_event = asyncio.Event()

        # Watchdog state — see _watchdog_task_handler for details
        self._last_stt_time: float | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._user_is_speaking = False

        # Flux event handlers
        self._register_event_handler("on_start_of_turn")
        self._register_event_handler("on_turn_resumed")
        self._register_event_handler("on_end_of_turn")
        self._register_event_handler("on_eager_end_of_turn")
        self._register_event_handler("on_update")

    # ------------------------------------------------------------------
    # Abstract transport interface — implemented by each concrete subclass
    # ------------------------------------------------------------------

    @abstractmethod
    async def _transport_send_audio(self, audio: bytes):
        """Send raw audio bytes over the transport."""
        pass

    @abstractmethod
    async def _transport_send_json(self, message: dict):
        """Serialize and send a JSON control message over the transport."""
        pass

    @abstractmethod
    def _transport_is_active(self) -> bool:
        """Return True if the transport connection is currently active."""
        pass

    @abstractmethod
    async def _connect(self):
        """Establish the transport connection."""
        pass

    @abstractmethod
    async def _disconnect(self):
        """Tear down the transport connection."""
        pass

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _build_query_string(self) -> str:
        """Build query string from current settings and init-only connection config."""
        params = [
            f"model={self._settings.model}",
            f"sample_rate={self.sample_rate}",
            f"encoding={self._encoding}",
        ]

        if self._settings.eager_eot_threshold is not None:
            params.append(f"eager_eot_threshold={self._settings.eager_eot_threshold}")

        if self._settings.eot_threshold is not None:
            params.append(f"eot_threshold={self._settings.eot_threshold}")

        if self._settings.eot_timeout_ms is not None:
            params.append(f"eot_timeout_ms={self._settings.eot_timeout_ms}")

        if self._mip_opt_out is not None:
            params.append(f"mip_opt_out={str(self._mip_opt_out).lower()}")

        # Add keyterm parameters (can have multiple)
        for keyterm in self._settings.keyterm:
            params.append(urlencode({"keyterm": keyterm}))

        # Add tag parameters (can have multiple)
        for tag_value in self._tag:
            params.append(urlencode({"tag": tag_value}))

        # Add language_hint parameters (only valid on flux-general-multi)
        hints = self._settings.language_hints
        if hints and not isinstance(hints, _NotGiven):
            if self._settings.model == self._MULTILINGUAL_MODEL:
                for code in _prepare_language_hints(hints):
                    params.append(urlencode({"language_hint": code}))
            else:
                logger.warning(
                    f"language_hints only supported on {self._MULTILINGUAL_MODEL}; "
                    f"ignoring hints for model {self._settings.model!r}"
                )

        return "&".join(params)

    async def _send_silence(self, duration_secs: float = 0.5):
        """Send a block of silence of the specified duration (default 500 ms)."""
        sample_width = 2  # bytes per sample for 16-bit PCM
        num_channels = 1  # mono
        num_samples = int(self.sample_rate * duration_secs)
        silence = b"\x00" * (num_samples * sample_width * num_channels)
        await self._transport_send_audio(silence)

    async def _watchdog_task_handler(self):
        """Prevent dangling turns by sending silence when audio stops flowing.

        If we stop sending audio to Flux after receiving a StartOfTurn,
        we never receive the UserStoppedSpeaking event unless we resume
        sending audio.
        """
        while self._transport_is_active():
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
        """Sends a CloseStream control message to Deepgram Flux.

        This signals to the server that no more audio data will be sent.
        """
        try:
            if self._transport_is_active():
                logger.debug("Sending CloseStream message to Deepgram Flux")
                await self._transport_send_json({"type": "CloseStream"})
        except Exception as e:
            await self.push_error(error_msg=f"Error sending CloseStream: {e}", exception=e)

    async def _send_configure(self, fields: set[str]):
        """Send a Configure control message to update settings mid-stream.

        Builds a Configure JSON message containing only the fields that changed
        and sends it over the existing connection.

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

        if "language_hints" in fields:
            if self._settings.model != self._MULTILINGUAL_MODEL:
                logger.warning(
                    f"language_hints only supported on {self._MULTILINGUAL_MODEL}; "
                    f"skipping Configure update for model {self._settings.model!r}"
                )
            else:
                hints = self._settings.language_hints
                # Empty list clears hints; NOT_GIVEN/None also treated as clear
                # since we only reach this branch when the user set the field.
                if hints is None or isinstance(hints, _NotGiven):
                    message["language_hints"] = []
                else:
                    message["language_hints"] = _prepare_language_hints(hints)

        logger.debug(f"{self}: sending Configure message: {message}")
        await self._transport_send_json(message)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Deepgram Flux service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta.

        Configure-able fields (keyterm, eot_threshold, eager_eot_threshold,
        eot_timeout_ms, language_hints) are sent to Deepgram via a Configure
        message. Other fields are stored but cannot be applied to the active
        connection.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        configure_fields = changed.keys() & self._CONFIGURE_FIELDS
        if configure_fields and self._transport_is_active():
            await self._send_configure(configure_fields)

        self._warn_unhandled_updated_settings(changed.keys() - self._CONFIGURE_FIELDS)

        return changed

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        """Start the Deepgram Flux STT service.

        Args:
            frame: The start frame containing initialization parameters and metadata.
        """
        await super().start(frame)
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
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _validate_message(self, data: dict[str, Any]) -> bool:
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

    async def _handle_message(self, data: dict[str, Any]):
        """Handle a parsed message from Deepgram Flux.

        Routes messages to appropriate handlers based on their type. Validates
        message structure before processing.

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

        This event is fired when the connection to Deepgram Flux is successfully
        established and ready to receive audio data for transcription processing.
        """
        logger.info("Connected to Flux - ready to stream audio")
        # Notify connection is established
        self._connection_established_event.set()

    async def _handle_fatal_error(self, data: dict[str, Any]):
        """Handle fatal error messages from Deepgram Flux.

        Fatal errors indicate unrecoverable issues with the connection or
        configuration that require intervention. These errors will cause
        the connection to be terminated.

        Args:
            data: The error message data containing error details.

        Raises:
            Exception: Always raises to trigger error handling in the transport layer.
        """
        error_msg = data.get("error", "Unknown error")
        deepgram_error = f"Fatal error: {error_msg}"
        logger.error(deepgram_error)
        # Error will be handled by the transport's receive loop error handler
        raise Exception(deepgram_error)

    async def _handle_turn_info(self, data: dict[str, Any]):
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

    def _calculate_average_confidence(self, transcript_data) -> float | None:
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

    def _primary_detected_language(self, data: dict[str, Any]) -> Language | None:
        """Extract the primary detected language from a TurnInfo payload.

        Only populated by ``flux-general-multi``; returns ``None`` otherwise.
        """
        codes = data.get("languages") or []
        if not codes:
            return None
        return _code_to_pipecat_language(codes[0])

    async def _handle_end_of_turn(self, transcript: str, data: dict[str, Any]):
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
        detected_language = self._primary_detected_language(data)

        if not self._settings.min_confidence or average_confidence > self._settings.min_confidence:
            # EndOfTurn means Flux has determined the turn is complete,
            # so this TranscriptionFrame is always finalized
            await self.push_frame(
                TranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    detected_language,
                    result=data,
                    finalized=True,
                )
            )
        else:
            logger.warning(
                f"Transcription confidence below min_confidence threshold: {average_confidence}"
            )

        await self._handle_transcription(transcript, True, detected_language)
        await self.stop_processing_metrics()
        await self.broadcast_frame(UserStoppedSpeakingFrame)
        await self._call_event_handler("on_end_of_turn", transcript)

    async def _handle_eager_end_of_turn(self, transcript: str, data: dict[str, Any]):
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
                self._primary_detected_language(data),
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
