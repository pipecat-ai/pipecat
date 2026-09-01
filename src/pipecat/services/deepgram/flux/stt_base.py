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
from typing_extensions import override

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import DEEPGRAM_FLUX_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given


class FluxConnectionNotConfirmedError(Exception):
    """Flux accepted the connection but never confirmed it was ready."""


class FluxFatalError(Exception):
    """Flux reported a fatal error and terminated the connection.

    Attributes:
        code: The error code Flux sent, e.g. ``UNPARSABLE_CLIENT_MESSAGE``.
    """

    def __init__(self, message: str, code: str):
        """Initialize the error.

        Args:
            message: The formatted error message.
            code: The error code Flux sent.
        """
        super().__init__(message)
        self.code = code


def language_to_deepgram_flux_language(language: Language) -> str:
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
    WARNING = "Warning"


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


class FluxTurnDetection(StrEnum):
    """Who decides when a user turn ends.

    Parameters:
        AUTOMATIC: Flux's end-of-turn model drives the conversation. The service
            recommends
            :class:`~pipecat.turns.user_turn_strategies.ExternalUserTurnStrategies`
            and proposes turn boundaries from Flux's own ``StartOfTurn`` and
            ``EndOfTurn`` events.
        MANUAL: The service transcribes only. Local VAD asks Flux to finalize the
            audio sent so far (via ``ForceEndTurn``) and the resulting
            transcripts flow to the aggregator, but the turn itself is decided by
            whichever user turn strategies the pipeline is configured with — a
            smart-turn analyzer, an LLM completion gate, or anything else. No
            turn boundaries are proposed.
    """

    AUTOMATIC = "automatic"
    MANUAL = "manual"


@dataclass
class DeepgramFluxSTTSettings(STTSettings):
    """Settings for DeepgramFluxSTTService.

    Parameters:
        eager_eot_threshold: EagerEndOfTurn/TurnResumed threshold, 0.3 to 0.9.
            Off by default. Lower values = more aggressive (faster response,
            more LLM calls). Higher values = more conservative (slower response,
            fewer LLM calls).
        eot_threshold: End-of-turn confidence required to finish a turn, 0.5 to
            1.0 (default 0.7). 1.0 suppresses Flux's natural end-of-turn
            detection entirely, leaving ``eot_timeout_ms`` and any explicit
            ``ForceEndTurn`` as the only ways a turn ends — most useful with
            ``turn_detection=FluxTurnDetection.MANUAL``.
        eot_timeout_ms: Maximum silence in ms before finishing a turn regardless
            of EOT confidence, 500 to 60000 (default 5000). The timer resets when
            new speech is detected, and applies even when ``eot_threshold`` is
            1.0.
        keyterm: Keyterms to boost recognition accuracy for specialized terminology.
        min_confidence: Minimum confidence required to create a TranscriptionFrame.
            Unset by default, which accepts every transcript. Under
            ``turn_detection=FluxTurnDetection.MANUAL`` a dropped transcript is
            the only signal the turn would have produced, so setting this can
            lose the user's words outright.
        numerals: Convert spoken numbers to numeral form (e.g. "twenty three" → "23").
            Read only from the connection URL, so an update is applied by
            reconnecting.
        language_hints: Languages to bias transcription toward. Only honored by the
            ``flux-general-multi`` model. An empty list clears any active hints;
            ``None``/``NOT_GIVEN`` means no hints (auto-detect). Can be updated
            mid-stream via ``STTUpdateSettingsFrame``.
    """

    eager_eot_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    eot_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    eot_timeout_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    keyterm: list | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_confidence: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    numerals: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    language_hints: list[Language] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


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
    # Fields Flux only accepts in the connection URL, so changing them reconnects.
    _CONNECTION_FIELDS = {"model", "numerals"}
    # Fields applied to results as they arrive, so no connection change is needed.
    _LOCAL_FIELDS = {"min_confidence"}
    _MULTILINGUAL_MODEL = "flux-general-multi"
    # How long to wait for Flux to confirm a new connection. An endpoint that
    # rejects a connection parameter sends neither a Connected message nor an
    # error, so the wait needs a bound.
    _CONNECTION_TIMEOUT = 10.0
    # Flux error codes whose cause a retry cannot clear. Rejected credentials
    # don't appear here: those fail the HTTP handshake and are classified from
    # its status code, never reaching a Flux error message. Anything not listed
    # falls back to the default classification, leaving recovery to the
    # service's own reconnect handling.
    _ERROR_CODE_CATEGORIES = {
        "UNPARSABLE_CLIENT_MESSAGE": ErrorCategory.INVALID_REQUEST,
    }
    # How long an in-flight Configure is trusted before a new update supersedes
    # it outright. Flux caps the number of un-acked Configure messages, so at
    # most one is ever in flight; this bounds how long a missing ack can block
    # later updates from ever being sent.
    _CONFIGURE_ACK_TIMEOUT = 5.0

    def __init__(
        self,
        *,
        encoding: str = "linear16",
        mip_opt_out: bool | None = None,
        tag: list | None = None,
        should_interrupt: bool = True,
        turn_detection: FluxTurnDetection = FluxTurnDetection.AUTOMATIC,
        watchdog_min_timeout: float = 0.5,
        settings: Settings,
        ttfs_p99_latency: float | None = DEEPGRAM_FLUX_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Deepgram Flux STT base service.

        Args:
            encoding: Audio encoding format. Must be "linear16".
            mip_opt_out: Opt out of the Deepgram Model Improvement Program.
            tag: Tags to label requests for identification during usage reporting.
            should_interrupt: Whether to interrupt the bot when Flux detects that
                the user is speaking. Passed along to the user turn strategies
                this service recommends, which own the interruption; a
                user-supplied ``user_turn_strategies`` overrides the
                recommendation and this setting with it. Ignored under
                ``FluxTurnDetection.MANUAL``, where the turn start strategy owns
                interruptions.
            turn_detection: Who decides when a user turn ends. Defaults to
                :attr:`FluxTurnDetection.AUTOMATIC` (Flux decides). See
                :class:`FluxTurnDetection`.
            watchdog_min_timeout: minimum idle timeout before sending silence to
                prevent dangling turns. The actual threshold is
                ``max(chunk_duration * 2, watchdog_min_timeout)``. Defaults to 0.5.
            settings: Fully resolved settings instance (built by concrete subclass).
            ttfs_p99_latency: P99 latency from speech end to final transcript in
                seconds, reported only under :attr:`FluxTurnDetection.MANUAL`.
                Override for your deployment. See
                https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the parent STTService (e.g.
                ``sample_rate``, ``reconnect_on_error``).
        """
        super().__init__(settings=settings, ttfs_p99_latency=ttfs_p99_latency, **kwargs)

        self._encoding = encoding
        self._mip_opt_out = mip_opt_out
        self._tag = tag or []
        self._should_interrupt = should_interrupt
        self._turn_detection = turn_detection
        self._watchdog_min_timeout = watchdog_min_timeout

        if turn_detection is FluxTurnDetection.MANUAL and not should_interrupt:
            logger.warning(
                f"{self}: should_interrupt is ignored under FluxTurnDetection.MANUAL; "
                "configure interruptions on the user turn start strategy instead"
            )

        # Connection readiness: Flux sends a "Connected" message when ready
        self._connection_established_event = asyncio.Event()

        # Configure serialization: Flux caps the number of un-acked Configure
        # messages, so we only allow one in flight at a time. A Configure sent
        # while one is already in flight is coalesced into
        # `_configure_pending_fields` instead, and sent once the in-flight one
        # is acked (see `_on_configure_acked`) — only the latest settings
        # matter, so there's no need to replay every intermediate update.
        self._configure_in_flight = False
        self._configure_sent_at: float | None = None
        self._configure_pending_fields: set[str] | None = None

        # Watchdog state — see _watchdog_task_handler for details
        self._last_stt_time: float | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._user_is_speaking = False
        self._last_audio_chunk_duration: float = 0.0

        # Flux event handlers
        self._register_event_handler("on_start_of_turn")
        self._register_event_handler("on_turn_resumed")
        self._register_event_handler("on_end_of_turn")
        self._register_event_handler("on_eager_end_of_turn")
        self._register_event_handler("on_update")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Deepgram Flux service supports metrics generation.
        """
        return True

    @property
    def supports_ttfs(self) -> bool:
        """Whether a speech-end to final-transcript interval exists to measure.

        Only under :attr:`FluxTurnDetection.MANUAL`, where VAD marks the end of
        speech and the final transcript arrives a ``ForceEndTurn`` round trip
        later. Under :attr:`FluxTurnDetection.AUTOMATIC` Flux defines the turn
        boundary itself, so the two coincide and there is no interval.
        """
        return self._turn_detection is FluxTurnDetection.MANUAL

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend external turn strategies when Flux detects turns server-side.

        Under :attr:`FluxTurnDetection.AUTOMATIC`, Flux emits its own
        start-of-turn and end-of-turn events (as
        ``ProposedUserStarted/StoppedSpeakingFrame``), so the user aggregator
        resolves those rather than running local VAD/smart-turn. Applied unless
        the user passed their own ``user_turn_strategies``.

        Under :attr:`FluxTurnDetection.MANUAL` no recommendation is made: the
        turn belongs to whichever strategies the pipeline already has.
        """
        frame = super().service_metadata_frame()
        if self._turn_detection is FluxTurnDetection.AUTOMATIC:
            frame.user_turn_strategies = ExternalUserTurnStrategies(
                enable_interruptions=self._should_interrupt,
            )
        return frame

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

    @override
    async def _do_reconnect(self):
        """Tear down the transport connection and re-establish it.

        Called by ``STTService._reconnect()`` inside the reconnecting guard.
        """
        await self._disconnect()
        await self._connect()

    def _classify_error(self, exception: Exception) -> ErrorCategory | None:
        """Classify the failures Flux signals in its own protocol.

        Flux reports these over the connection rather than as an HTTP status,
        so they carry nothing the default classification can read.

        Args:
            exception: The exception to classify.

        Returns:
            The category, or None to fall back to the default classification.
        """
        if isinstance(exception, FluxConnectionNotConfirmedError):
            # Flux stays silent rather than refusing the connection when a
            # setting is unsupported, so an unconfirmed connection means the
            # request was rejected, not that the network was slow.
            return ErrorCategory.INVALID_REQUEST
        if isinstance(exception, FluxFatalError):
            return self._ERROR_CODE_CATEGORIES.get(exception.code)
        return None

    async def _await_connection_established(self):
        """Wait for Flux to confirm the connection is ready.

        Raises:
            FluxConnectionNotConfirmedError: If no confirmation arrives within
                ``_CONNECTION_TIMEOUT``.
        """
        try:
            await asyncio.wait_for(
                self._connection_established_event.wait(), timeout=self._CONNECTION_TIMEOUT
            )
        except TimeoutError:
            raise FluxConnectionNotConfirmedError(
                f"Flux did not confirm the connection within {self._CONNECTION_TIMEOUT}s; "
                "the endpoint may not accept the current connection settings"
            ) from None

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

        if self._settings.numerals is not None:
            params.append(f"numerals={str(self._settings.numerals).lower()}")

        if self._mip_opt_out is not None:
            params.append(f"mip_opt_out={str(self._mip_opt_out).lower()}")

        # Add keyterm parameters (can have multiple)
        for keyterm in assert_given(self._settings.keyterm):
            params.append(urlencode({"keyterm": keyterm}))

        # Add tag parameters (can have multiple)
        for tag_value in self._tag:
            params.append(urlencode({"tag": tag_value}))

        # Add language_hint parameters (only valid on flux-general-multi)
        hints = self._settings.language_hints
        if hints and is_given(hints):
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
        # Watchdog silence is real audio submitted to the service, so it
        # counts toward usage.
        self._record_stt_audio_usage(silence)

    async def _watchdog_task_handler(self):
        """Prevent dangling turns by sending silence when audio stops flowing.

        If we stop sending audio to Flux after receiving a StartOfTurn,
        we never receive the UserStoppedSpeaking event unless we resume
        sending audio.
        """
        while self._transport_is_active():
            now = time.monotonic()
            # Send silence if we go more than 500 ms or twice the chunk size
            # without sending new audio to Flux.
            threshold = max(self._last_audio_chunk_duration * 2, self._watchdog_min_timeout)
            if (
                self._user_is_speaking
                and self._last_stt_time
                and now - self._last_stt_time > threshold
            ):
                logger.warning(
                    f"No audio received for {threshold * 1000:.0f} ms. Sending silence to Flux to prevent a dangling task"
                )
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

    # ------------------------------------------------------------------
    # Turn control
    # ------------------------------------------------------------------

    async def force_end_turn(self) -> None:
        """Ask Flux to finalize the audio sent so far.

        Flux answers with an ``EndOfTurn`` carrying ``trigger: "manual"`` and the
        transcript for everything transcribed before the request arrived, which
        this service pushes as a finalized ``TranscriptionFrame``. Whether that
        also ends the *Pipecat* turn depends on ``turn_detection``: under
        :attr:`FluxTurnDetection.AUTOMATIC` it does, under
        :attr:`FluxTurnDetection.MANUAL` the configured turn strategies decide.

        Under :attr:`FluxTurnDetection.MANUAL` this is called automatically on
        every ``VADUserStoppedSpeakingFrame``. Call it directly to finalize on
        some other signal, such as a push-to-talk release.

        Does nothing when no turn is in progress, since there would be nothing
        to finalize.
        """
        if not self._transport_is_active() or not self._user_is_speaking:
            return

        self.request_finalize()
        logger.trace(f"{self}: sending ForceEndTurn")
        await self._transport_send_json({"type": "ForceEndTurn"})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, finalizing turns on the VAD signal in manual mode.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if (
            isinstance(frame, VADUserStoppedSpeakingFrame)
            and self._turn_detection is FluxTurnDetection.MANUAL
        ):
            await self.force_end_turn()

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
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

    async def cleanup(self):
        """Release Deepgram Flux STT resources at teardown."""
        await super().cleanup()
        await self._disconnect()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _send_configure(self, fields: set[str]):
        """Send a Configure control message to update settings mid-stream.

        Builds a Configure JSON message containing only the fields that changed
        and sends it over the existing connection.

        At most one Configure is ever in flight, since Flux caps the number of
        un-acked Configure messages. If one is already in flight, ``fields`` is
        merged into the pending set and sent once the in-flight one is acked
        (see ``_on_configure_acked``) instead of being sent now — the message is
        always built from the current settings, so only the latest values ever
        need to go out. An in-flight Configure older than
        ``_CONFIGURE_ACK_TIMEOUT`` is treated as lost so a missing ack can't
        permanently block later updates.

        Args:
            fields: Set of changed field names to include in the message.
        """
        if self._configure_in_flight:
            assert self._configure_sent_at is not None
            if time.monotonic() - self._configure_sent_at < self._CONFIGURE_ACK_TIMEOUT:
                self._configure_pending_fields = (self._configure_pending_fields or set()) | fields
                return
            logger.warning(
                f"{self}: timed out after {self._CONFIGURE_ACK_TIMEOUT}s waiting for "
                "Configure ack; sending the next Configure anyway"
            )

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
                if hints is None or not is_given(hints):
                    message["language_hints"] = []
                else:
                    message["language_hints"] = _prepare_language_hints(hints)

        self._configure_in_flight = True
        self._configure_sent_at = time.monotonic()
        logger.debug(f"{self}: sending Configure message: {message}")
        await self._transport_send_json(message)

    async def _on_configure_acked(self):
        """Mark the in-flight Configure as acked and flush any pending update.

        Called when a ConfigureSuccess/ConfigureFailure arrives. If fields were
        coalesced into ``_configure_pending_fields`` while this Configure was in
        flight, immediately sends a follow-up Configure covering all of them —
        unless the transport has since gone inactive, in which case the pending
        fields are simply dropped, since a reconnect re-applies current settings
        via the connection URL anyway. Safe to call with nothing in flight (e.g.
        a stray/duplicate ack), which is a no-op.
        """
        self._configure_in_flight = False
        self._configure_sent_at = None
        if self._configure_pending_fields is not None:
            fields = self._configure_pending_fields
            self._configure_pending_fields = None
            if self._transport_is_active():
                await self._send_configure(fields)

    def _reset_configure_state(self):
        """Clear Configure-serialization state during teardown.

        Called when the connection is torn down (including ahead of a
        reconnect), since any in-flight or pending Configure can no longer be
        acked or sent on a dead connection. A reconnect re-applies the current
        settings via the connection URL, so nothing needs to be replayed.
        """
        self._configure_in_flight = False
        self._configure_sent_at = None
        self._configure_pending_fields = None

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta.

        Configure-able fields (keyterm, eot_threshold, eager_eot_threshold,
        eot_timeout_ms, language_hints) are sent to Deepgram via a Configure
        message. Fields Flux only reads from the connection URL trigger a
        reconnect, which waits until the user stops speaking.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        configure_fields = changed.keys() & self._CONFIGURE_FIELDS
        if configure_fields and self._transport_is_active():
            await self._send_configure(configure_fields)

        if changed.keys() & self._CONNECTION_FIELDS:
            await self._request_reconnect()

        self._warn_unhandled_updated_settings(
            changed.keys() - self._CONFIGURE_FIELDS - self._CONNECTION_FIELDS - self._LOCAL_FIELDS
        )

        return changed

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
                await self._handle_connection_established(data)
            case FluxMessageType.RECEIVE_FATAL_ERROR:
                await self._handle_fatal_error(data)
            case FluxMessageType.TURN_INFO:
                await self._handle_turn_info(data)
            case FluxMessageType.CONFIGURE_SUCCESS:
                logger.info(f"{self}: Configure accepted: {data}")
                await self._on_configure_acked()
            case FluxMessageType.CONFIGURE_FAILURE:
                error_code = data.get("error_code", "unknown")
                description = data.get("description", "no description")
                error_msg = f"Configure rejected: [{error_code}] {description}"
                logger.warning(f"{self}: {error_msg}")
                await self._on_configure_acked()
                await self.push_error(error_msg=error_msg)
            case FluxMessageType.WARNING:
                await self._handle_warning(data)

    async def _handle_connection_established(self, data: dict[str, Any]):
        """Handle successful connection establishment to Deepgram Flux.

        This event is fired when the connection to Deepgram Flux is successfully
        established and ready to receive audio data for transcription processing.
        """
        request_id = data.get("request_id")
        logger.info(f"{self}: Connected to Flux - ready to stream audio ({request_id=})")
        # Notify connection is established
        self._connection_established_event.set()

    async def _handle_warning(self, data: dict[str, Any]):
        """Handle non-fatal Warning messages from Deepgram Flux.

        Warnings never interrupt the stream, so none of them push an error.
        ``FORCE_END_TURN_NO_ACTIVE_TURN`` in particular is routine rather than
        exceptional: a ``ForceEndTurn`` races the ``EndOfTurn`` that Flux may
        have already sent, and losing that race just means the turn we wanted to
        finalize is already finalized.

        Args:
            data: The Warning message data.
        """
        code = data.get("code", "unknown")
        description = data.get("description", "no description")
        if code == "FORCE_END_TURN_NO_ACTIVE_TURN":
            logger.trace(f"{self}: ForceEndTurn arrived with no active turn")
        else:
            logger.warning(f"{self}: Flux warning: [{code}] {description}")

    async def _handle_fatal_error(self, data: dict[str, Any]):
        """Handle fatal error messages from Deepgram Flux.

        Fatal errors indicate unrecoverable issues with the connection or
        configuration that require intervention. These errors will cause
        the connection to be terminated.

        Args:
            data: The error message data containing error details.

        Raises:
            FluxFatalError: Always raises to trigger error handling in the transport layer.
        """
        error_code = data.get("code", "unknown")
        description = data.get("description", "no description")
        deepgram_error = f"{self}: Fatal error [{error_code}] {description}"
        logger.error(deepgram_error)
        # Error will be handled by the transport's receive loop error handler
        raise FluxFatalError(deepgram_error, code=error_code)

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

        if not isinstance(event, str):
            logger.debug(f"Unhandled TurnInfo event (not a string): {event}")
            return

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
        of a new speaking turn.

        Under :attr:`FluxTurnDetection.AUTOMATIC` the service proposes a turn
        start, which the user turn strategies resolve into a
        UserStartedSpeakingFrame and an interruption. Under
        :attr:`FluxTurnDetection.MANUAL` the configured start strategies open the
        turn on their own signals, so nothing is proposed here.

        Args:
            transcript: maybe the first few words of the turn.
        """
        logger.debug("User started speaking")
        self._user_is_speaking = True
        if self._turn_detection is FluxTurnDetection.AUTOMATIC:
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
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

        On ``flux-general-multi`` the language is read from TurnInfo's
        ``languages`` field. On ``flux-general-en`` the field is absent, so we
        fall back to ``Language.EN`` to match the model's fixed language.
        """
        codes = data.get("languages") or []
        if codes:
            return _code_to_pipecat_language(codes[0])
        if self._settings.model == "flux-general-en":
            return Language.EN
        return None

    async def _handle_end_of_turn(self, transcript: str, data: dict[str, Any]):
        """Handle EndOfTurn events from Deepgram Flux.

        EndOfTurn events are fired when Deepgram Flux determines that a speaking
        turn has concluded, either due to sufficient silence or end-of-turn
        confidence thresholds being met. This provides the final transcript
        for the completed turn.

        The service will:
        - Create and send a final TranscriptionFrame with the complete transcript
        - Trigger transcription handling with tracing for metrics
        - Under :attr:`FluxTurnDetection.AUTOMATIC`, propose a turn stop, which
          the user turn strategies resolve into a UserStoppedSpeakingFrame

        Under :attr:`FluxTurnDetection.MANUAL` the event means only "here is the
        transcript for the audio so far" — several can arrive within one user
        turn, and the aggregator concatenates them until the configured stop
        strategy ends the turn.

        Args:
            transcript: The final transcript text for the completed turn.
            data: The TurnInfo message data containing event type, transcript and some extra metadata.
        """
        trigger = data.get("trigger")
        logger.debug(f"User stopped speaking ({trigger=})")
        self._user_is_speaking = False

        # Only a manual trigger answers a ForceEndTurn we sent, so only it can
        # settle the finalize request that went out with it.
        if trigger == "manual":
            self.confirm_finalize()

        # Compute the average confidence
        average_confidence = self._calculate_average_confidence(data)
        detected_language = self._primary_detected_language(data)

        min_confidence = assert_given(self._settings.min_confidence)
        # No threshold (None or 0.0) → accept. Otherwise require confidence
        # data and compare; drop if data is missing.
        if not min_confidence or (
            average_confidence is not None and average_confidence > min_confidence
        ):
            # Report usage before the transcription frame so tracing can
            # attach it to the STT span the frame closes.
            await self.emit_stt_usage_metrics()
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
        if self._turn_detection is FluxTurnDetection.AUTOMATIC:
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
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
