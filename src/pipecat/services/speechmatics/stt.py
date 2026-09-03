#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechmatics STT service integration."""

import asyncio
import os
import warnings
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from loguru import logger
from pydantic import BaseModel

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, assert_given, is_given
from pipecat.services.stt_latency import SPEECHMATICS_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from speechmatics.agent_stt import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_MODEL,
        AdditionalVocabEntry,
        AgentSttAsyncClient,
        AudioFormat,
        AudioEncoding,
        Model,
        Segment,
        SpeakerDiarizationConfig,
        SpeakerIdentifier,
        TranscriptionConfig,
        TurnConfig,
    )
    from speechmatics.agent_stt import ClientMessageType as AgentClientMessageType
    from speechmatics.agent_stt import ServerMessageType as AgentServerMessageType
    from speechmatics.agent_stt import TurnDetectionMode as AgentTurnDetectionMode
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Speechmatics, you need to `uv add "pipecat-ai[speechmatics]"`.')
    raise ImportError(f"Missing module: {e}") from e


def _resolve_model(model: Model | str | None, operating_point: Model | str | None) -> str:
    """Resolve the transcription model, preferring `model` over the deprecated
    `operating_point` alias.

    Both accept a `Model` enum member or its wire string. If both are given they must
    match; if only one is given it wins; if neither, the default model is used.
    (`Model` is a `str` enum, so string/enum values compare equal.)
    """
    if model is not None and operating_point is not None and model != operating_point:
        raise ValueError(
            f"`model` ({model!r}) and `operating_point` ({operating_point!r}) differ. "
            "Pass only `model` (`operating_point` is deprecated)."
        )
    if model is None and operating_point is not None:
        warnings.warn(
            "`operating_point` is deprecated; use `model` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    resolved = model or operating_point or DEFAULT_MODEL
    return resolved.value if isinstance(resolved, Model) else resolved


class TurnDetectionMode(str, Enum):
    """How turn boundaries (end of speech) are detected.

    `VAD`: the STT service runs its own VAD and closes turns itself.

    `EXTERNAL`: turn boundaries are controlled by the caller — the service does not
    endpoint on its own, and the caller drives turns by calling `finalize()` (for
    example from Pipecat's own VAD).

    The values mirror the Agent STT SDK's own turn-detection modes so the two never
    drift.
    """

    VAD = AgentTurnDetectionMode.VAD.value
    EXTERNAL = AgentTurnDetectionMode.EXTERNAL.value


def _handle_turn_detection_mode(mode: TurnDetectionMode) -> AgentTurnDetectionMode:
    """Map the service's turn detection mode onto the SDK's.

    The values match, so this is a direct lookup — but it's still required:
    ``TranscriptionConfig.to_dict()`` compares by identity and lifts the mode into the
    top-level ``turn_config``, so the config must carry the SDK's own enum member.
    """
    return AgentTurnDetectionMode(mode.value)


@dataclass
class SpeechmaticsSTTSettings(STTSettings):
    """Settings for SpeechmaticsSTTService.

    See ``SpeechmaticsSTTService.InputParams`` for detailed descriptions of each field.

    Parameters:
        domain: Domain for Speechmatics API.
        turn_detection_mode: Endpoint handling mode.
        speaker_active_format: Formatter for speaker ID.
        known_speakers: List of known speaker labels and identifiers.
        additional_vocab: List of additional vocabulary entries.
        model: Resolved transcription model (operating point). See ``_resolve_model``.
        operating_point: Deprecated alias for ``model``.
        include_partials: Include partial segment fragments.
        enable_diarization: Enable speaker diarization.
        speaker_sensitivity: Diarization sensitivity.
        max_speakers: Maximum number of speakers to detect.
        prefer_current_speaker: Prefer current speaker ID.
    """

    domain: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_detection_mode: TurnDetectionMode | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speaker_active_format: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    known_speakers: list[SpeakerIdentifier] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    additional_vocab: list[AdditionalVocabEntry] | _NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    operating_point: Model | str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    include_partials: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_diarization: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speaker_sensitivity: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_speakers: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prefer_current_speaker: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    #: Fields that are purely local (formatting templates) — no reconnect
    #: and no API call needed.
    LOCAL_FIELDS: ClassVar[frozenset[str]] = frozenset({"speaker_active_format"})


def _build_diarization_config(s: SpeechmaticsSTTSettings) -> SpeakerDiarizationConfig | None:
    """Build the wire ``speaker_diarization_config`` from the diarization settings.

    Returns ``None`` when diarization is off or no diarization knob is set, so an empty
    config is never sent. Only the fields that were actually set are included.
    """
    if not s.enable_diarization:
        return None

    fields: dict[str, Any] = {}
    if s.max_speakers is not None:
        fields["max_speakers"] = s.max_speakers
    if s.speaker_sensitivity is not None:
        fields["speaker_sensitivity"] = s.speaker_sensitivity
    if s.prefer_current_speaker is not None:
        fields["prefer_current_speaker"] = s.prefer_current_speaker
    if s.known_speakers:
        fields["speakers"] = s.known_speakers

    return SpeakerDiarizationConfig(**fields) if fields else None


class SpeechmaticsSTTService(STTService):
    """Speechmatics STT service implementation.

    This service provides real-time speech-to-text transcription using the Speechmatics API.
    It supports partial and final transcriptions, multiple languages, various audio formats,
    and speaker diarization.

    Event handlers available (in addition to STTService events):

    - on_speakers_result(service, speakers): Speaker diarization results received

    Example::

        @stt.event_handler("on_speakers_result")
        async def on_speakers_result(service, speakers):
            ...
    """

    Settings = SpeechmaticsSTTSettings
    _settings: Settings

    # Export related classes as class attributes
    TurnDetectionMode = TurnDetectionMode
    AudioEncoding = AudioEncoding
    Model = Model
    SpeakerIdentifier = SpeakerIdentifier
    AdditionalVocabEntry = AdditionalVocabEntry

    # Reconnect backoff: first retry after this many seconds, doubling each attempt up
    # to the cap. Retries continue for the life of the session so a transient drop can
    # never permanently deafen the pipeline.
    RECONNECT_INITIAL_DELAY: ClassVar[float] = 1.0
    RECONNECT_MAX_DELAY: ClassVar[float] = 30.0

    class InputParams(BaseModel):
        """Configuration parameters for Speechmatics STT service.

        Parameters:
            domain: Domain for Speechmatics API. Defaults to None.

            language: Language code for transcription. Defaults to `Language.EN`.

            turn_detection_mode: How turns are closed. `TurnDetectionMode.VAD` lets the
                STT service run its own VAD and close turns itself; `TurnDetectionMode.EXTERNAL`
                has the caller drive turns via `finalize()` (e.g. Pipecat's own VAD).
                Defaults to `TurnDetectionMode.VAD`.

            speaker_active_format: Formatter for the speaker ID. This formatter is used to format
                the text output for individual speakers and ensures that the context is clear for
                language models further down the pipeline. The attributes `text` and `speaker_id` are
                available. The system instructions for the language model may need to include any
                necessary instructions to handle the formatting.
                Example: `@{speaker_id}: {text}`. Defaults to None.

            known_speakers: List of known speaker labels and identifiers. If you supply a list of
                labels and identifiers for speakers, then the STT engine will use them to attribute
                any spoken words to that speaker. This is useful when you want to attribute words
                to a specific speaker, such as the assistant or a specific user. Labels and identifiers
                can be obtained from a running STT session and then used in subsequent sessions.
                Identifiers are unique to each Speechmatics account and cannot be used across accounts.
                Refer to our examples on the format of the known_speakers parameter.
                Defaults to [].

            additional_vocab: List of additional vocabulary entries. If you supply a list of
                additional vocabulary entries, the this will increase the weight of the words in the
                vocabulary and help the STT engine to better transcribe the words.
                Defaults to [].

            audio_encoding: Audio encoding format. Defaults to AudioEncoding.PCM_S16LE.

            model: The transcription model (operating point) to use, e.g. `"linden-1"`.
                Defaults to `Model.LINDEN_1`, the SDK's default model. Preferred over
                `operating_point`.

            operating_point: Deprecated alias for `model`. If both are given they must name the
                same value, otherwise a `ValueError` is raised. Optional.

            include_partials: Include partial segment fragments (words) in the output of
                AddPartialSegment messages. Partial fragments from the STT will always be used for
                speaker activity detection. This setting is used only for the formatted text output
                of individual segments.

            enable_diarization: Enable speaker diarization. When enabled, the STT engine will
                determine and attribute words to unique speakers. The speaker_sensitivity
                parameter can be used to adjust the sensitivity of diarization.

            speaker_sensitivity: Diarization sensitivity. A higher value increases the sensitivity
                of diarization and helps when two or more speakers have similar voices.

            max_speakers: Maximum number of speakers to detect. This forces the STT engine to cluster
                words into a fixed number of speakers. It should not be used to limit the number of
                speakers, unless it is clear that there will only be a known number of speakers.

            prefer_current_speaker: Prefer current speaker ID. When set to true, groups of words close
                together are given extra weight to be identified as the same speaker.

        """

        # Service configuration
        domain: str | None = None
        language: Language | str = Language.EN

        # Endpointing mode
        turn_detection_mode: TurnDetectionMode = TurnDetectionMode.VAD

        # Output formatting
        speaker_active_format: str | None = None

        # Speakers
        known_speakers: list[SpeakerIdentifier] = []

        # Custom dictionary
        additional_vocab: list[AdditionalVocabEntry] = []

        # Audio
        audio_encoding: AudioEncoding = AudioEncoding.PCM_S16LE

        # -------------------
        # Advanced features
        # -------------------

        # Features
        model: Model | str | None = None
        operating_point: Model | str | None = None
        include_partials: bool | None = None

        # Diarization
        enable_diarization: bool | None = None
        speaker_sensitivity: float | None = None
        max_speakers: int | None = None
        prefer_current_speaker: bool | None = None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        sample_rate: int | None = None,
        encoding: AudioEncoding = AudioEncoding.PCM_S16LE,
        params: InputParams | None = None,
        should_interrupt: bool = True,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = SPEECHMATICS_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Speechmatics STT service.

        Args:
            api_key: Speechmatics API key for authentication. Uses environment variable
                `SPEECHMATICS_API_KEY` if not provided.
            base_url: Base URL for Speechmatics API. Uses environment variable `SPEECHMATICS_RT_URL`
                or defaults to `wss://eu2.rt.speechmatics.com/v2`.
            sample_rate: Optional audio sample rate in Hz.
            encoding: Audio encoding format. Defaults to ``AudioEncoding.PCM_S16LE``.
            params: Input parameters for the service.

                .. deprecated:: 0.0.105
                    Use ``settings=SpeechmaticsSTTService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            should_interrupt: Determine whether the bot should be interrupted when Speechmatics turn_detection_mode is configured to detect user speech.
            settings: Runtime-updatable settings. When provided alongside deprecated
                ``params``, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to STTService.
        """
        # Service parameters
        self._api_key: str = api_key or os.getenv("SPEECHMATICS_API_KEY")
        self._base_url: str = (
            base_url or os.getenv("SPEECHMATICS_RT_URL") or "wss://eu2.rt.speechmatics.com/v2"
        )

        # Check we have required attributes
        if not self._api_key:
            raise ValueError("Missing Speechmatics API key")
        if not self._base_url:
            raise ValueError("Missing Speechmatics base URL")

        self._should_interrupt = should_interrupt

        # Deprecation check (mutates params in-place for legacy kwargs migration)
        _params = params or SpeechmaticsSTTService.InputParams()
        _legacy_kwargs = self._check_deprecated_args(kwargs, _params)

        # --- 1. Hardcoded defaults ---
        default_settings = self.Settings(
            model=None,  # Resolved from model / operating_point below
            language=Language.EN,
            domain=None,
            turn_detection_mode=TurnDetectionMode.VAD,
            speaker_active_format="{text}",
            known_speakers=[],
            additional_vocab=[],
            operating_point=None,
            include_partials=None,
            enable_diarization=None,
            speaker_sensitivity=None,
            max_speakers=None,
            prefer_current_speaker=None,
        )

        # --- 2. No direct init arg overrides ---

        # --- 3. Deprecated params overrides ---
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
        # Apply the migrated params whenever the legacy path was used — either an
        # explicit `params=` or deprecated kwargs migrated into `_params`.
        if (params is not None or _legacy_kwargs) and not settings:
            default_settings.language = _params.language
            default_settings.domain = _params.domain
            default_settings.turn_detection_mode = _params.turn_detection_mode
            # Output formatting default — prefix the speaker when diarizing.
            speaker_active_format = _params.speaker_active_format
            if speaker_active_format is None:
                speaker_active_format = (
                    "@{speaker_id}: {text}" if _params.enable_diarization else "{text}"
                )
            default_settings.speaker_active_format = speaker_active_format
            default_settings.known_speakers = _params.known_speakers
            default_settings.additional_vocab = _params.additional_vocab
            encoding = _params.audio_encoding
            default_settings.model = _params.model
            default_settings.operating_point = _params.operating_point
            default_settings.include_partials = _params.include_partials
            default_settings.enable_diarization = _params.enable_diarization
            default_settings.speaker_sensitivity = _params.speaker_sensitivity
            default_settings.max_speakers = _params.max_speakers
            default_settings.prefer_current_speaker = _params.prefer_current_speaker

        # --- 4. Settings delta (canonical API, always wins) ---
        if settings is not None:
            default_settings.apply_update(settings)

        # Reconcile the preferred `model` with the deprecated `operating_point` alias
        # (model preferred, both-differ raises, default = DEFAULT_MODEL) before building
        # the SDK config from settings.
        default_settings.model = _resolve_model(
            default_settings.model, default_settings.operating_point
        )

        # Build SDK config from settings before calling super.
        self._client: AgentSttAsyncClient | None = None
        self._audio_encoding = encoding
        self._config: TranscriptionConfig = self._build_config(default_settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        # Message queue
        self._stt_msg_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._stt_msg_task: asyncio.Task | None = None

        # Reconnect state. `_closed` gates the reconnect loop off once the session is
        # torn down (stop/cancel/cleanup); `_reconnect_task` holds the in-flight retry.
        self._closed: bool = False
        self._reconnect_task: asyncio.Task | None = None

        # Event handlers
        if default_settings.enable_diarization:
            self._register_event_handler("on_speakers_result")

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Request external turn strategies when Speechmatics endpoints server-side.

        Every mode other than ``EXTERNAL`` (which uses Pipecat's own endpointing) has
        Speechmatics detect turns and emit the turn frames, so the user aggregator
        defers to those. Applied unless the user passed their own
        ``user_turn_strategies``.
        """
        frame = super().service_metadata_frame()
        mode = self._settings.turn_detection_mode
        if is_given(mode) and mode != TurnDetectionMode.EXTERNAL:
            frame.user_turn_strategies = ExternalUserTurnStrategies()
        return frame

    @property
    def session_id(self) -> str | None:
        """The Agent STT session id, set once ``RecognitionStarted`` arrives (else None)."""
        info = self._client.session_info if self._client else None
        return getattr(info, "session_id", None) if info is not None else None

    # ============================================================================
    # LIFE-CYCLE / SESSION MANAGEMENT
    # ============================================================================

    async def start(self, frame: StartFrame):
        """Called when the new session starts."""
        await super().start(frame)
        self._closed = False
        await self._connect()

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta, reconnecting only when necessary.

        LOCAL_FIELDS (formatting templates) take effect immediately with no reconnect.
        Every other field is baked into the ``TranscriptionConfig`` at connect time, so
        changing one requires a full disconnect / reconnect.

        Args:
            delta: A settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        if changed.keys() - self.Settings.LOCAL_FIELDS:
            logger.debug(f"{self} settings update requires reconnect: {changed.keys()}")
            # Connection-level fields changed — rebuild the config, then reconnect.
            self._config = self._build_config(self._settings)
            await self._disconnect()
            await self._connect()
        else:
            # Only local (formatting) fields changed — effective immediately.
            logger.debug(f"{self} local settings update, no reconnect: {changed.keys()}")

        return changed

    async def stop(self, frame: EndFrame):
        """Called when the session ends."""
        await super().stop(frame)
        self._closed = True
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Called when the session is cancelled."""
        await super().cancel(frame)
        self._closed = True
        await self._disconnect()

    async def cleanup(self):
        """Release Speechmatics resources at pipeline teardown."""
        await super().cleanup()
        self._closed = True
        await self._disconnect()

    async def _connect(self) -> None:
        """Connect to the STT service, scheduling a retry if the attempt fails."""
        if not await self._open_connection():
            self._schedule_reconnect()

    async def _open_connection(self, *, report_error: bool = True) -> bool:
        """Build the client, register handlers, and open the connection.

        - Create STT client
        - Register handlers for messages
        - Connect to the client
        - Start message processing task

        Args:
            report_error: Whether to surface a connect failure via ``push_error``. The
                reconnect loop passes False so retries only log instead of spamming the
                pipeline with an error per attempt.

        Returns:
            True if the connection is live, False if the attempt failed (the caller
            decides whether to retry).
        """
        # Log the event
        logger.debug(f"{self} connecting to Speechmatics STT service")

        # Agent STT client. Turn detection is a top-level turn_config (sibling of the
        # transcription config); audio encoding / sample rate go via AudioFormat.
        self._client = AgentSttAsyncClient(
            api_key=self._api_key,
            url=self._base_url,
            app=f"pipecat/{pipecat_version()}",
            config=self._config,
            turn_config=TurnConfig(
                turn_detection_mode=_handle_turn_detection_mode(
                    assert_given(self._settings.turn_detection_mode)
                )
            ),
            audio_format=AudioFormat(
                encoding=self._audio_encoding,
                sample_rate=self.sample_rate,
                chunk_size=DEFAULT_CHUNK_SIZE,
            ),
        )

        # Message pump — feeds handler callbacks into the ordered processing queue.
        def add_message(message: dict[str, Any]):
            self._stt_msg_queue.put_nowait(message)

        # Segment + status listeners.
        self._client.on(AgentServerMessageType.ADD_PARTIAL_SEGMENT, add_message)
        self._client.on(AgentServerMessageType.ADD_SEGMENT, add_message)
        self._client.on(AgentServerMessageType.ERROR, add_message)
        self._client.on(AgentServerMessageType.WARNING, add_message)

        # Service-side turn events (only emitted when the service closes turns).
        if self._settings.turn_detection_mode != TurnDetectionMode.EXTERNAL:
            self._client.on(AgentServerMessageType.START_OF_TURN, add_message)
            self._client.on(AgentServerMessageType.END_OF_TURN, add_message)

        # Speaker diarization results.
        if self._settings.enable_diarization:
            self._client.on(AgentServerMessageType.SPEAKERS_RESULT, add_message)

        # Connect. A rejected session (e.g. invalid config) or transport failure surfaces
        # via push_error so it reaches the pipeline instead of dying silently.
        try:
            await self._client.connect()
            logger.debug(f"{self} connected")
        except Exception as e:
            self._client = None
            if report_error:
                await self.push_error(
                    error_msg=f"Error connecting to STT service: {e}", exception=e
                )
            else:
                logger.warning(f"{self} reconnect attempt failed: {e}")
            return False

        # Start message processing task
        if not self._stt_msg_task:
            self._stt_msg_task = self.create_task(self._process_stt_messages())
        return True

    def _schedule_reconnect(self) -> None:
        """Start the background reconnect loop, unless one is already running or the
        session has been torn down."""
        if self._closed or self._reconnect_task is not None:
            return
        self._reconnect_task = self.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Retry the connection with exponential backoff until it comes back.

        Runs for the life of the session: a transient drop is retried indefinitely (with
        a capped, doubling delay) so audio is never permanently dropped in silence. The
        loop exits on the first successful reconnect or once the session is closed. The
        initial failure was already surfaced via ``push_error``; retries only log.
        """
        delay = self.RECONNECT_INITIAL_DELAY
        try:
            while not self._closed and self._client is None:
                logger.warning(f"{self} reconnecting to Speechmatics STT in {delay:.0f}s")
                await asyncio.sleep(delay)
                if self._closed:
                    return
                if await self._open_connection(report_error=False):
                    logger.debug(f"{self} reconnected to Speechmatics STT")
                    return
                delay = min(delay * 2, self.RECONNECT_MAX_DELAY)
        finally:
            self._reconnect_task = None

    async def _disconnect(self) -> None:
        """Disconnect from the STT service.

        - Cancel any in-flight reconnect attempt
        - Cancel message processing task
        - Disconnect the client
        - Emit on_disconnected event handler for clients
        """
        # Cancel a pending reconnect so it doesn't race this teardown.
        if self._reconnect_task:
            await self.cancel_task(self._reconnect_task)
            self._reconnect_task = None

        # Cancel the message processing task
        if self._stt_msg_task:
            await self.cancel_task(self._stt_msg_task)
            self._stt_msg_task = None

        # Disconnect the client
        logger.debug(f"{self} disconnecting from Speechmatics STT service")
        try:
            if self._client:
                await self._client.disconnect()
        except TimeoutError:
            logger.warning(f"{self} timeout while closing Speechmatics client connection")
        except Exception as e:
            await self.push_error(error_msg=f"Error closing Speechmatics client: {e}", exception=e)
        finally:
            self._client = None
            await self._call_event_handler("on_disconnected")

    async def _process_stt_messages(self) -> None:
        """Process messages from the STT client.

        Messages from the STT client are processed in a separate task to avoid blocking the main
        thread. They are handled in strict order in which they are received.
        """
        try:
            while True:
                message = await self._stt_msg_queue.get()
                await self._handle_message(message)
        except asyncio.CancelledError:
            pass

    # ============================================================================
    # CONFIGURATION
    # ============================================================================

    def _build_config(self, settings: Settings) -> TranscriptionConfig:
        """Build an Agent STT ``TranscriptionConfig`` from the given settings.

        Only fields Agent STT accepts on the wire are set. Audio encoding / sample rate are
        passed to the client via ``AudioFormat``; turn detection is passed to the client via
        ``TurnConfig`` (a top-level ``turn_config`` sibling of ``transcription_config``).
        """
        s = settings
        language = assert_given(s.language)
        sm_language = self._language_to_speechmatics_language(language)

        return TranscriptionConfig(
            language=sm_language,
            model=assert_given(s.model),
            diarization="speaker" if s.enable_diarization else None,
            speaker_diarization_config=_build_diarization_config(s),
            additional_vocab=s.additional_vocab or None,
            output_locale=self._locale_to_speechmatics_locale(sm_language, language),
            domain=s.domain or None,
            enable_partials=s.include_partials,
        )

    # ============================================================================
    # HANDLE ENGINE MESSAGES
    # ============================================================================

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Handle a message from the STT client."""
        event = message.get("message", "")

        # Handle events
        match event:
            case AgentServerMessageType.ADD_PARTIAL_SEGMENT:
                await self._handle_partial_segment(message)
            case AgentServerMessageType.ADD_SEGMENT:
                await self._handle_segment(message)
            case AgentServerMessageType.START_OF_TURN:
                await self._handle_start_of_turn(message)
            case AgentServerMessageType.END_OF_TURN:
                await self._handle_end_of_turn(message)
            case AgentServerMessageType.SPEAKERS_RESULT:
                await self._handle_speakers_result(message)
            case AgentServerMessageType.ERROR:
                await self._handle_error(message)
            case AgentServerMessageType.WARNING:
                self._handle_warning(message)
            case _:
                logger.debug(f"{self} {event} -> {message}")

    async def _handle_partial_segment(self, message: dict[str, Any]) -> None:
        """Handle AddPartialSegment events.

        Agent STT sends a single ``segment`` object (``transcript``/``speaker``) plus
        message-level ``metadata``; ``Segment.from_message`` reads that singular shape.

        Args:
            message: the message payload.
        """
        segment = Segment.from_message(message)
        if segment.transcript:
            await self._send_frame(segment, finalized=False)

    async def _handle_segment(self, message: dict[str, Any]) -> None:
        """Handle AddSegment events.

        Agent STT sends a single final ``segment`` object plus message-level ``metadata``.

        Args:
            message: the message payload.
        """
        segment = Segment.from_message(message)
        if not segment.transcript:
            return

        # If a finalize() was requested, confirm it before pushing so this final frame is
        # tagged as the one that was asked for.
        if self._finalize_requested:
            self.confirm_finalize()

        await self._send_frame(segment, finalized=True)

    async def _handle_start_of_turn(self, message: dict[str, Any]) -> None:
        """Handle StartOfTurn events.

        When Speechmatics STT detects the start of a new speaking turn, a StartOfTurn
        event is triggered. This triggers bot interruption to stop any ongoing speech
        synthesis and signals the start of user speech detection.

        The service will:
        - Send a BotInterruptionFrame upstream to stop bot speech
        - Send a UserStartedSpeakingFrame downstream to notify other components
        - Start metrics collection for measuring response times

        Args:
            message: the message payload.
        """
        logger.debug(f"{self} StartOfTurn received")
        # await self.start_processing_metrics()
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.broadcast_interruption()

    async def _handle_end_of_turn(self, message: dict[str, Any]) -> None:
        """Handle EndOfTurn events.

        EndOfTurn events are triggered by Speechmatics STT when it concludes a
        speaking turn. This occurs either due to silence or reaching the
        end-of-turn confidence thresholds. These events provide the final
        transcript for the completed turn.

        The service will:
        - Stop processing metrics collection
        - Send a UserStoppedSpeakingFrame to signal turn completion

        Args:
            message: the message payload.
        """
        logger.debug(f"{self} EndOfTurn received")
        # await self.stop_processing_metrics()
        await self.broadcast_frame(UserStoppedSpeakingFrame)

    async def _handle_speakers_result(self, message: dict[str, Any]) -> None:
        """Handle SpeakersResult events.

        SpeakersResult events are triggered by Speechmatics STT when it provides
        speaker information for the current speaking turn.

        Args:
            message: the message payload.
        """
        logger.debug(f"{self} speakers result received from STT")
        await self._call_event_handler("on_speakers_result", message)

    @staticmethod
    def _describe_status(message: dict[str, Any]) -> str:
        """Build a human-readable string from a server status message.

        Error/Warning/Info messages carry a ``type`` and ``reason`` (and sometimes a
        numeric ``code``); any may be absent, so fall back to the raw payload.
        """
        parts = [str(message[k]) for k in ("type", "code", "reason") if message.get(k) is not None]
        return " ".join(parts) if parts else str(message)

    async def _handle_error(self, message: dict[str, Any]) -> None:
        """Handle Error events.

        An Error ends the session server-side, so surface it upstream via
        ``push_error`` instead of letting the session die silently.
        """
        await self.push_error(f"Speechmatics STT error: {self._describe_status(message)}")

    def _handle_warning(self, message: dict[str, Any]) -> None:
        """Handle Warning events.

        The session continues (possibly with adjusted config), so log without
        interrupting the pipeline.
        """
        logger.warning(f"{self} Speechmatics STT warning: {self._describe_status(message)}")

    # ============================================================================
    # SEND FRAMES TO PIPELINE
    # ============================================================================

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for VAD and metrics handling.

        Args:
            frame: Frame to process.
            direction: Direction of frame processing.
        """
        # Forward to parent
        await super().process_frame(frame, direction)

        # Force finalization — only when the caller drives turns (EXTERNAL).
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._settings.turn_detection_mode != TurnDetectionMode.EXTERNAL:
                logger.warning(
                    f"{self} VADUserStoppedSpeakingFrame received but the service VAD is in use"
                )
            elif self._client is not None:
                self.request_finalize()
                self._client.finalize()

    def _segment_to_frame(
        self, segment: Segment, *, finalized: bool
    ) -> TranscriptionFrame | InterimTranscriptionFrame:
        """Transform an Agent STT ``Segment`` into a Pipecat transcription frame.

        Pure mapping (the Gap 1 seam) — no side effects. ``finalized`` picks the frame
        type. ``language`` has no wire field, so it comes from the configured setting;
        ``result`` has no wire field and is left unset.
        """
        language = assert_given(self._settings.language)
        active_format = assert_given(self._settings.speaker_active_format)
        text = active_format.format(
            speaker_id=segment.speaker or "UU",
            text=segment.transcript,
            ts=segment.start_time,
            lang=language,
        )

        frame_cls = TranscriptionFrame if finalized else InterimTranscriptionFrame
        return frame_cls(
            text=text,
            user_id=segment.speaker or "",
            timestamp=time_now_iso8601(),
            language=language,
        )

    async def _send_frame(self, segment: Segment, *, finalized: bool) -> None:
        """Emit one transcription frame for a segment, with final-only metrics.

        Args:
            segment: The segment to emit.
            finalized: Whether this is a final (True) or interim (False) transcript.
        """
        frame = self._segment_to_frame(segment, finalized=finalized)

        if finalized:
            await self._handle_transcription(
                segment.transcript, is_final=True, language=assert_given(self._settings.language)
            )
            # Report usage before the transcription frame so tracing can attach it to the
            # STT span the frame closes.
            await self.emit_stt_usage_metrics()
            logger.debug(f"{self} finalized transcript: {frame.text!r}")
        else:
            logger.debug(f"{self} interim transcript: {frame.text!r}")

        await self.push_frame(frame)

    # ============================================================================
    # PUBLIC FUNCTIONS
    # ============================================================================

    async def send_message(self, message: AgentClientMessageType | str, **kwargs: Any) -> None:
        """Send a message to the STT service.

        This sends a message to the STT service via the underlying transport. If the session
        is not running, this will raise an exception. Messages in the wrong format will also
        cause an error.

        Args:
            message: Message to send to the STT service.
            **kwargs: Additional arguments passed to the underlying transport.
        """
        try:
            payload = {"message": message}
            payload.update(kwargs)
            logger.debug(f"{self} sending message to STT: {payload}")
            self.create_task(self._client.send_message(payload))
        except Exception as e:
            raise RuntimeError(f"{self} error sending message to STT: {e}")

    # ============================================================================
    # METRICS
    # ============================================================================

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Speechmatics STT supports generation of metrics.
        """
        return True

    @traced_stt
    async def _handle_transcription(self, transcript: str, is_final: bool, language: Language):
        """Record transcription event for tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Adds audio to the audio buffer and yields None."""
        try:
            if self._client:
                await self._client.send_audio(audio)
            yield None
        except Exception as e:
            yield ErrorFrame(f"Speechmatics error: {e}")
            # The stream is broken; tear down and retry rather than silently dropping
            # all further audio for the rest of the session.
            await self._disconnect()
            self._schedule_reconnect()

    # ============================================================================
    # HELPERS
    # ============================================================================

    def _language_to_speechmatics_language(self, language: Language) -> str:
        """Convert a Language enum to a Speechmatics language code.

        Args:
            language: The Language enum to convert.

        Returns:
            str: The Speechmatics language code, if found.
        """
        # List of supported input languages
        BASE_LANGUAGES = {
            Language.AR: "ar",
            Language.BA: "ba",
            Language.EU: "eu",
            Language.BE: "be",
            Language.BG: "bg",
            Language.BN: "bn",
            Language.YUE: "yue",
            Language.CA: "ca",
            Language.HR: "hr",
            Language.CS: "cs",
            Language.DA: "da",
            Language.NL: "nl",
            Language.EN: "en",
            Language.EO: "eo",
            Language.ET: "et",
            Language.FA: "fa",
            Language.FI: "fi",
            Language.FR: "fr",
            Language.GL: "gl",
            Language.DE: "de",
            Language.EL: "el",
            Language.HE: "he",
            Language.HI: "hi",
            Language.HU: "hu",
            Language.IT: "it",
            Language.ID: "id",
            Language.GA: "ga",
            Language.JA: "ja",
            Language.KO: "ko",
            Language.LV: "lv",
            Language.LT: "lt",
            Language.MS: "ms",
            Language.MT: "mt",
            Language.CMN: "cmn",
            Language.MR: "mr",
            Language.MN: "mn",
            Language.NO: "no",
            Language.PL: "pl",
            Language.PT: "pt",
            Language.RO: "ro",
            Language.RU: "ru",
            Language.SK: "sk",
            Language.SL: "sl",
            Language.ES: "es",
            Language.SV: "sv",
            Language.SW: "sw",
            Language.TA: "ta",
            Language.TH: "th",
            Language.TR: "tr",
            Language.UG: "ug",
            Language.UK: "uk",
            Language.UR: "ur",
            Language.VI: "vi",
            Language.CY: "cy",
        }

        # Get the language code
        result = resolve_language(language, BASE_LANGUAGES, use_base_code=True)

        # Fail if language is not supported
        if not result:
            raise ValueError(f"Unsupported language: {language}")

        # Return the language code
        return result

    def _locale_to_speechmatics_locale(self, base_code: str, locale: Language) -> str | None:
        """Convert a Language enum to a Speechmatics language / locale code.

        Args:
            base_code: The language code.
            locale: The Language enum to convert.

        Returns:
            str: The Speechmatics language code, if found.
        """
        # Languages and output locales
        LOCALES = {
            "en": {
                Language.EN_GB: "en-GB",
                Language.EN_US: "en-US",
                Language.EN_AU: "en-AU",
            },
        }

        # Ensure language code is in the map
        if "-" not in str(locale) or base_code not in LOCALES:
            return None

        # Get the locale code
        result = LOCALES.get(base_code).get(locale, None)

        # Fail if locale is not supported
        if not result:
            logger.warning(f"{self} Unsupported output locale: {locale}, defaulting to {base_code}")

        # Return the locale code
        return result

    def _check_deprecated_args(self, kwargs: dict, params: InputParams) -> bool:
        """Check arguments for deprecation and update params if necessary.

        This function will show deprecation warnings for deprecated arguments and
        migrate them to the new location in the params object. If the new location
        is None, the argument is not used. Recognized deprecated arguments are
        popped from ``kwargs`` so they are not forwarded to the parent constructor.

        Args:
            kwargs: Keyword arguments passed to the constructor.
            params: Input parameters for the service.

        Returns:
            True if any deprecated argument was present, so the caller knows to
            apply the migrated ``params`` to its settings.
        """

        # Show deprecation warnings
        def _deprecation_warning(old: str, new: str | None = None) -> None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                if new:
                    message = f"`{old}` is deprecated, use `InputParams.{new}`"
                else:
                    message = f"`{old}` is deprecated and not used"
                warnings.warn(message, DeprecationWarning)

        # List of deprecated arguments and their new location
        deprecated_args = [
            ("language", "language"),
            ("language_code", "language"),
            ("domain", "domain"),
            ("output_locale", None),
            ("output_locale_code", None),
            ("enable_partials", None),
            ("max_delay", None),
            ("chunk_size", None),
            ("audio_encoding", "audio_encoding"),
            ("end_of_utterance_silence_trigger", None),
            ("enable_speaker_diarization", "enable_diarization"),
            ("text_format", "speaker_active_format"),
            ("max_speakers", "max_speakers"),
            ("transcription_config", None),
            ("enable_vad", None),
            ("end_of_utterance_mode", None),
        ]

        # Show warnings + migrate the arguments. Recognized deprecated kwargs are
        # popped so they are not forwarded to the parent constructor.
        found = False
        for old, new in deprecated_args:
            if old in kwargs:
                found = True
                value = kwargs.pop(old)
                _deprecation_warning(old, new)
                if new is not None and value is not None:
                    setattr(params, new, value)
        return found
