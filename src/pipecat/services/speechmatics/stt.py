#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechmatics STT service integration."""

import asyncio
import os
import warnings
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, cast

from loguru import logger
from pydantic import BaseModel

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import SPEECHMATICS_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.network import exponential_backoff_time
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given

try:
    from speechmatics.agent_stt import (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_MODEL,
        AdditionalVocabEntry,
        AgentSttAsyncClient,
        AudioEncoding,
        AudioFormat,
        AuthenticationError,
        ConfigurationError,
        Model,
        Segment,
        SessionError,
        SpeakerDiarizationConfig,
        SpeakerIdentifier,
        TranscriptionConfig,
        TranscriptionError,
        TurnConfig,
    )
    from speechmatics.agent_stt import ClientMessageType as AgentClientMessageType
    from speechmatics.agent_stt import ServerMessageType as AgentServerMessageType
    from speechmatics.agent_stt import TurnDetectionMode as AgentTurnDetectionMode
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Speechmatics, you need to `uv add "pipecat-ai[speechmatics]"`.')
    raise ImportError(f"Missing module: {e}") from e


# Connect-time failures that will never clear on retry (auth, bad config, rejected session).
# These are reported as permanent, leaving the service unusable, and never trigger a
# reconnect; every other connect exception is treated as a transient drop and retried with
# backoff.
_PERMANENT_CONNECT_ERRORS = (
    AuthenticationError,
    ConfigurationError,
    TranscriptionError,
    SessionError,
)

# HTTP statuses a WebSocket handshake returns when the credentials are rejected.
_AUTH_REJECTION_STATUSES = frozenset({401, 403})

# Server ``Error`` types that reject the request or the account, so a fresh session with
# the same configuration would fail the same way. Every other type (idle/session timeouts,
# buffer, data, and internal errors) ends the session, but a new one can succeed.
_PERMANENT_SERVER_ERROR_TYPES = frozenset(
    {
        "invalid_message",
        "invalid_model",
        "invalid_config",
        "invalid_audio_type",
        "invalid_output_format",
        "not_authorised",
        "insufficient_funds",
        "not_allowed",
        "protocol_error",
        "quota_exceeded",
    }
)


def _is_auth_rejection(exc: BaseException) -> bool:
    """Whether ``exc`` is a WebSocket handshake rejected for authentication (HTTP 401/403).

    A rejected credential surfaces as a ``ConnectionError`` carrying the underlying
    ``websockets`` handshake error in its exception chain. The status is read from that error
    (``InvalidStatus.response.status_code``, or the legacy ``InvalidStatusCode.status_code``),
    with the status in the message text as a fallback.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None)
        if status is None:
            status = getattr(current, "status_code", None)
        if status in _AUTH_REJECTION_STATUSES:
            return True
        current = current.__cause__ or current.__context__
    text = str(exc)
    return any(f"HTTP {status}" in text for status in _AUTH_REJECTION_STATUSES)


def _resolve_model(
    model: Model | str | None | NotGiven, operating_point: Model | str | None | NotGiven
) -> str:
    """Resolve the transcription model from `model` and the deprecated `operating_point`.

    Both accept a `Model` enum member or its wire string; an unset (`NOT_GIVEN`) value
    counts as `None`. If both are given they must match; if only one is given it wins;
    if neither, the default model is used. (`Model` is a `str` enum, so string/enum
    values compare equal.)
    """
    model = model if is_given(model) else None
    operating_point = operating_point if is_given(operating_point) else None
    if model is not None and operating_point is not None and model != operating_point:
        raise ValueError(
            f"`model` ({model!r}) and `operating_point` ({operating_point!r}) differ. "
            "Pass only `model` (`operating_point` is deprecated)."
        )
    if model is None and operating_point is not None:
        warnings.warn(
            "`operating_point` is deprecated since 1.10.0 and will be removed in 2.0.0. "
            "Use `model` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    resolved = model or operating_point or DEFAULT_MODEL
    return resolved.value if isinstance(resolved, Model) else resolved


class TurnDetectionMode(StrEnum):
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
        operating_point: Alias for ``model``.

            .. deprecated:: 1.10.0
                Use ``model`` instead. Will be removed in 2.0.0.

        enable_partials: Include partial segment fragments.
        enable_diarization: Enable speaker diarization.
        speaker_sensitivity: Diarization sensitivity.
        max_speakers: Maximum number of speakers to detect.
        prefer_current_speaker: Prefer current speaker ID.
    """

    domain: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    turn_detection_mode: TurnDetectionMode | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speaker_active_format: str | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    known_speakers: list[SpeakerIdentifier] | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    additional_vocab: list[AdditionalVocabEntry] | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    operating_point: Model | str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_partials: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_diarization: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speaker_sensitivity: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_speakers: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prefer_current_speaker: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

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

    # Attempts a single reconnect makes before giving up. Audio is buffered for the
    # whole sequence, so the bound also caps how much is held and replayed at once.
    RECONNECT_MAX_ATTEMPTS: ClassVar[int] = 3

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

            operating_point: Alias for `model`. If both are given they must name the same
                value, otherwise a `ValueError` is raised. Optional.

                .. deprecated:: 1.10.0
                    Use ``model`` instead. Will be removed in 2.0.0.

            enable_partials: Include partial segment fragments (words) in the output of
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
        enable_partials: bool | None = None

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
                or defaults to `wss://eu2.rt.speechmatics.com/v2/agent`.
            sample_rate: Optional audio sample rate in Hz.
            encoding: Audio encoding format. Defaults to ``AudioEncoding.PCM_S16LE``.
            params: Input parameters for the service.

                .. deprecated:: 0.0.105
                    Use ``settings=SpeechmaticsSTTService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            should_interrupt: Determine whether the bot should be interrupted when
                Speechmatics turn_detection_mode is configured to detect user speech.
                Passed along to the user turn strategies this service recommends,
                which own the interruption; a user-supplied ``user_turn_strategies``
                overrides the recommendation and this setting with it.
            settings: Runtime-updatable settings. When provided alongside deprecated
                ``params``, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to STTService.
        """
        # Service parameters
        api_key = api_key or os.getenv("SPEECHMATICS_API_KEY")
        base_url = (
            base_url or os.getenv("SPEECHMATICS_RT_URL") or "wss://eu2.rt.speechmatics.com/v2/agent"
        )

        # Check we have required attributes
        if not api_key:
            raise ValueError("Missing Speechmatics API key")
        if not base_url:
            raise ValueError("Missing Speechmatics base URL")

        self._api_key: str = api_key
        self._base_url: str = base_url

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
            enable_partials=None,
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
            legacy_encoding = self._apply_legacy_params(default_settings, _params)
            if legacy_encoding is not None:
                encoding = legacy_encoding

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

        # Stops reconnect attempts once the session is torn down (stop/cancel/cleanup)
        # or rejected outright.
        self._closed: bool = False

        # A reconnect running in the background (see _schedule_reconnect).
        self._reconnect_task: asyncio.Task | None = None

        # Registered unconditionally: diarization can be turned on at runtime, and a
        # handler added while it was off would otherwise be dropped.
        self._register_event_handler("on_speakers_result")

    @staticmethod
    def _apply_legacy_params(settings: Settings, params: InputParams) -> AudioEncoding | None:
        """Fold the deprecated ``InputParams`` into canonical ``Settings``.

        Every field the two shapes share by name is copied straight across. Two are
        special-cased and excluded from the generic copy: ``speaker_active_format`` has a
        diarization-aware default, and ``audio_encoding`` has no ``Settings`` field (it
        reaches the client via the separate ``encoding`` argument).

        Args:
            settings: The canonical settings to populate in place.
            params: The deprecated input params to migrate from.

        Returns:
            The audio encoding the caller set on ``params``, or None if they left it
            unset — the field carries a default that would otherwise silently override
            the ``encoding`` argument.
        """
        shared = type(settings).__dataclass_fields__.keys() & type(params).model_fields.keys()
        for name in shared - {"speaker_active_format"}:
            setattr(settings, name, getattr(params, name))

        # Output formatting default — prefix the speaker when diarizing.
        fmt = params.speaker_active_format
        if fmt is None:
            fmt = "@{speaker_id}: {text}" if params.enable_diarization else "{text}"
        settings.speaker_active_format = fmt

        return params.audio_encoding if "audio_encoding" in params.model_fields_set else None

    @property
    def _service_closes_turns(self) -> bool:
        """True when Speechmatics detects turns itself and emits Start/EndOfTurn.

        Every turn-detection mode except ``EXTERNAL`` (where the caller drives turns via
        ``finalize()``) has the service close turns. This gates the turn-scoped behavior
        — turn frames, turn-event subscriptions, and processing metrics — off in EXTERNAL
        mode, where Pipecat owns endpointing.
        """
        mode = self._settings.turn_detection_mode
        return is_given(mode) and mode != TurnDetectionMode.EXTERNAL

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Request external turn strategies when Speechmatics endpoints server-side.

        Every mode other than ``EXTERNAL`` (which uses Pipecat's own endpointing) has
        Speechmatics detect turns and propose the boundaries, so the user aggregator
        resolves those. Applied unless the user passed their own
        ``user_turn_strategies``.
        """
        frame = super().service_metadata_frame()
        if self._service_closes_turns:
            frame.user_turn_strategies = ExternalUserTurnStrategies(
                enable_interruptions=self._should_interrupt,
            )
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

        # A runtime change to `model` or the deprecated `operating_point` alias must be
        # re-reconciled into `model` (the only field `_build_config` reads); resolve from
        # just the fields that actually changed so a new `operating_point` wins on its own
        # instead of clashing with the already-resolved `model` (which would raise).
        if "model" in changed or "operating_point" in changed:
            self._settings.model = _resolve_model(
                self._settings.model if "model" in changed else NOT_GIVEN,
                self._settings.operating_point if "operating_point" in changed else NOT_GIVEN,
            )

        if changed.keys() - self.Settings.LOCAL_FIELDS:
            logger.debug(f"{self} settings update requires reconnect: {changed.keys()}")
            # Connection-level fields changed — rebuild the config, then reconnect. The
            # new settings may be what a rejected session needed (a supported language,
            # say), so a closed service gets another chance, matching the usability the
            # base class just restored.
            self._config = self._build_config(self._settings)
            self._closed = False
            await self._cancel_reconnect_task()
            await self._request_reconnect()
        else:
            # Only local (formatting) fields changed — effective immediately.
            logger.debug(f"{self} local settings update, no reconnect: {changed.keys()}")

        return changed

    async def stop(self, frame: EndFrame):
        """Called when the session ends."""
        await super().stop(frame)
        self._closed = True
        await self._cancel_reconnect_task()
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Called when the session is cancelled."""
        await super().cancel(frame)
        self._closed = True
        await self._cancel_reconnect_task()
        await self._disconnect()

    async def cleanup(self):
        """Release Speechmatics resources at pipeline teardown."""
        await super().cleanup()
        self._closed = True
        await self._cancel_reconnect_task()
        await self._disconnect()

    async def _connect(self) -> None:
        """Connect to the STT service, retrying in the background if the attempt fails.

        Runs from ``start()``, ahead of the ``StartFrame`` reaching the rest of the
        pipeline, so the backoff loop must not hold that up: audio is buffered while the
        retry runs. A rejected session marks the service closed, and retrying cannot
        clear that.
        """
        if not await self._open_connection() and not self._closed:
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
            transcription_config=self._config,
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

        # Casting to broaden what message types `on` accepts (the SDK annotates it
        # with the RT message enum, not the Agent STT one it also dispatches).
        on = cast(Callable[[AgentServerMessageType, Callable], Any], self._client.on)

        # Segment + status listeners.
        on(AgentServerMessageType.ADD_PARTIAL_SEGMENT, add_message)
        on(AgentServerMessageType.ADD_SEGMENT, add_message)
        on(AgentServerMessageType.ERROR, add_message)
        on(AgentServerMessageType.WARNING, add_message)

        # Service-side turn events (only emitted when the service closes turns).
        if self._service_closes_turns:
            on(AgentServerMessageType.START_OF_TURN, add_message)
            on(AgentServerMessageType.END_OF_TURN, add_message)

        # Speaker diarization results.
        if self._settings.enable_diarization:
            on(AgentServerMessageType.SPEAKERS_RESULT, add_message)

        # Connect. Errors reach the pipeline via push_error instead of dying silently, and are
        # split by recoverability: an unrecoverable rejection (auth / bad config / rejected
        # session) is permanent and stops the session, while any other failure is a transient
        # drop the caller retries with backoff.
        try:
            await self._client.connect()
            logger.debug(f"{self} connected")
        except _PERMANENT_CONNECT_ERRORS as e:
            self._client = None
            await self._fail_permanently(
                error_msg=f"Speechmatics STT rejected the session: {e}", exception=e
            )
            return False
        except Exception as e:
            self._client = None
            # A rejected credential arrives as a ConnectionError; report it as permanent
            # (like the other unrecoverable rejections) instead of retrying.
            if _is_auth_rejection(e):
                await self._fail_permanently(
                    error_msg=f"Speechmatics STT rejected the credentials: {e}", exception=e
                )
                return False
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

    async def _fail_permanently(self, error_msg: str, exception: Exception | None = None) -> None:
        """Report an error that will not clear on retry and stop the session reconnecting.

        The error is pushed as permanent, which marks the service unusable so it is given
        no more audio and the pipeline worker applies its ``ProcessorUnusablePolicy``. The
        session is also marked closed, which stops ``_do_reconnect`` retrying against it.
        A later settings update reopens the service (see ``_update_settings``).
        """
        self._closed = True
        await self.push_error(
            error_msg=error_msg, exception=exception, force_treat_as_permanent=True
        )

    async def _do_reconnect(self) -> None:
        """Re-establish the session, retrying with exponential backoff.

        Called by ``STTService._reconnect()`` inside the reconnecting guard, which holds
        for the whole call — so audio arriving during the retries is buffered and replayed
        rather than dropped. Exhausting the attempts is reported as a permanent error.
        """
        await self._disconnect()
        for attempt in range(1, self.RECONNECT_MAX_ATTEMPTS + 1):
            # A rejected session, or a stop/cancel that landed during the backoff,
            # marks the service closed: nothing may reopen it.
            if self._closed:
                return
            if await self._open_connection(report_error=False):
                logger.debug(f"{self} reconnected to Speechmatics STT")
                return
            if attempt < self.RECONNECT_MAX_ATTEMPTS and not self._closed:
                await asyncio.sleep(exponential_backoff_time(attempt))
        if self._closed:
            return
        await self._fail_permanently(
            f"Speechmatics STT failed to reconnect after {self.RECONNECT_MAX_ATTEMPTS} attempts"
        )

    def _schedule_reconnect(self) -> None:
        """Run a reconnect in the background, unless one is already in flight.

        For paths that cannot run the reconnect inline: the message pump, which the
        reconnect tears down, and the initial connect, where the backoff loop would
        otherwise hold up the rest of the pipeline's start.
        """
        if self._reconnect_task and not self._reconnect_task.done():
            return
        self._reconnect_task = self.create_task(self._request_reconnect(), name="reconnect")

    async def _cancel_reconnect_task(self) -> None:
        """Cancel a background reconnect, if one is still running."""
        task, self._reconnect_task = self._reconnect_task, None
        if task and not task.done() and task is not asyncio.current_task():
            await self.cancel_task(task)

    async def _disconnect(self) -> None:
        """Disconnect from the STT service.

        - Cancel message processing task
        - Disconnect the client
        - Emit on_disconnected event handler for clients
        """
        # Cancel the message processing task
        if self._stt_msg_task:
            await self.cancel_task(self._stt_msg_task)
            self._stt_msg_task = None

        # Drain any messages buffered from this session. The consumer task is cancelled
        # above, so anything still queued would otherwise be replayed into the next
        # session by the fresh consumer started on reconnect (the queue is reused).
        while not self._stt_msg_queue.empty():
            self._stt_msg_queue.get_nowait()

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

        # The stored language may be a plain code rather than a Language, but the
        # mapping keys compare equal either way.
        language = cast(Language, assert_given(s.language))
        sm_language = self._language_to_speechmatics_language(language)

        return TranscriptionConfig(
            language=sm_language,
            # The SDK annotates `model` with its enum but forwards any name; the
            # service resolves names the SDK has no member for.
            model=cast(Model, assert_given(s.model)),
            diarization="speaker" if s.enable_diarization else None,
            speaker_diarization_config=_build_diarization_config(s),
            additional_vocab=[*s.additional_vocab] if s.additional_vocab else None,
            output_locale=self._locale_to_speechmatics_locale(sm_language, language),
            domain=s.domain or None,
            enable_partials=assert_given(s.enable_partials),
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
        event is triggered. The service opens the turn's processing-metrics span and
        proposes a turn start, which the user turn strategies resolve into a
        UserStartedSpeakingFrame and, when ``should_interrupt`` is set, an interruption.

        Only reached when the service closes turns; EXTERNAL mode never subscribes to
        StartOfTurn, so the span is never opened there.

        Args:
            message: the message payload.
        """
        logger.debug(f"{self} StartOfTurn received")
        await self.start_processing_metrics()
        await self.broadcast_frame(ProposedUserStartedSpeakingFrame)

    async def _handle_end_of_turn(self, message: dict[str, Any]) -> None:
        """Handle EndOfTurn events.

        EndOfTurn events are triggered by Speechmatics STT when it concludes a
        speaking turn. This occurs either due to silence or reaching the
        end-of-turn confidence thresholds. These events provide the final
        transcript for the completed turn. The service proposes a turn stop, which
        the user turn strategies resolve into a UserStoppedSpeakingFrame.

        Args:
            message: the message payload.
        """
        logger.debug(f"{self} EndOfTurn received")
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)

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

        A server Error always ends the session. One that rejects the request or account
        is permanent: it is surfaced upstream and the service stops reconnecting. Any
        other (a timeout, a buffer or internal error) is reported and a new session is
        opened in the background, since this handler runs on the message pump the
        reconnect tears down.
        """
        error_msg = f"Speechmatics STT error: {self._describe_status(message)}"
        if message.get("type") in _PERMANENT_SERVER_ERROR_TYPES:
            await self._fail_permanently(error_msg)
            return
        await self.push_error(error_msg=error_msg)
        self._schedule_reconnect()

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
            if self._service_closes_turns:
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
        # The stored language may be a plain code rather than a Language; the frame
        # carries it as-is.
        language = cast(Language, assert_given(self._settings.language))
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

        # Close the turn's processing-metrics span on the final transcript. Gated so
        # EXTERNAL mode — which never opens the span (no StartOfTurn) — emits nothing.
        if finalized and self._service_closes_turns:
            await self.stop_processing_metrics()

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

        Raises:
            RuntimeError: If the session is not connected, or the message could not be
                sent (e.g. a malformed payload).
        """
        if self._client is None:
            raise RuntimeError(f"{self} cannot send message: STT session is not connected")

        payload = {"message": message, **kwargs}
        logger.debug(f"{self} sending message to STT: {payload}")
        try:
            await self._client.send_message(payload)
        except Exception as e:
            raise RuntimeError(f"{self} error sending message to STT: {e}") from e

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
                # send_audio swallows transport errors and shuts its own audio gate, so a
                # dropped socket is only visible as the gate being closed. A gate closed
                # with no session_error is a broken stream; when the service ended the
                # session itself, _handle_error has already reported it.
                if not self._client.is_ready_for_audio and self._client.session_error is None:
                    logger.warning(f"{self} audio stream closed, reconnecting")
                    await self._request_reconnect()
            yield None
        except Exception as e:
            yield ErrorFrame(f"Speechmatics error: {e}")
            await self._request_reconnect()

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
        result = LOCALES[base_code].get(locale, None)

        # Fail if locale is not supported. No `{self}` prefix here: `_build_config`
        # also runs from __init__, before the base class has named this processor.
        if not result:
            logger.warning(
                f"Unsupported Speechmatics output locale: {locale}, defaulting to {base_code}"
            )

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
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                if new:
                    message = f"`{old}` is deprecated, use `InputParams.{new}`"
                else:
                    message = f"`{old}` is deprecated and not used"
                # 3 frames out of this nested helper is the caller constructing
                # the service, which is the code that has to change.
                warnings.warn(message, DeprecationWarning, stacklevel=3)

        # List of deprecated arguments and their new location
        deprecated_args = [
            ("language", "language"),
            ("language_code", "language"),
            ("domain", "domain"),
            ("output_locale", None),
            ("output_locale_code", None),
            ("include_partials", "enable_partials"),
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
