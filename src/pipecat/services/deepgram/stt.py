#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram speech-to-text service implementation."""

import inspect
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Mapping, Optional, Type

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import _S, NOT_GIVEN, STTSettings, _NotGiven, is_given
from pipecat.services.stt_latency import DEEPGRAM_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from deepgram import (
        AsyncListenWebSocketClient,
        DeepgramClient,
        DeepgramClientOptions,
        ErrorResponse,
        LiveOptions,
        LiveResultResponse,
        LiveTranscriptionEvents,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Deepgram, you need to `pip install pipecat-ai[deepgram]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class _DeepgramSTTSettingsBase(STTSettings):
    """Base settings for Deepgram STT services that use ``LiveOptions``.

    Shared by ``DeepgramSTTSettings`` and ``DeepgramSageMakerSTTSettings``.
    Not intended for other Deepgram services that don't use ``LiveOptions``.

    Wraps the Deepgram SDK's ``LiveOptions`` in a single ``live_options``
    field and provides delta-merge semantics: when used as a delta (e.g.
    via ``STTUpdateSettingsFrame``), only the non-None fields of
    ``live_options`` are merged into the stored options rather than
    replacing them wholesale.

    ``model`` and ``language`` are kept in sync bidirectionally between
    the top-level settings fields and the nested ``live_options``.

    Parameters:
        live_options: Deepgram ``LiveOptions`` for STT configuration.
            In delta mode only its non-None fields are merged into the
            stored options.
    """

    live_options: LiveOptions | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    # Valid LiveOptions __init__ parameter names (cached at class level).
    _live_options_params: set[str] | None = field(default=None, init=False, repr=False)

    @classmethod
    def _get_live_options_params(cls) -> set[str]:
        """Return the set of valid ``LiveOptions.__init__`` parameter names."""
        if cls._live_options_params is None:
            cls._live_options_params = set(inspect.signature(LiveOptions.__init__).parameters) - {
                "self"
            }
        return cls._live_options_params

    def _merge_live_options_delta(self, delta: LiveOptions) -> Dict[str, Any]:
        """Merge a ``LiveOptions`` delta into the stored ``live_options``.

        Non-None fields from *delta* overwrite corresponding fields in the
        stored ``LiveOptions``.  ``model`` and ``language`` are synced to
        the top-level settings fields when they change.

        Args:
            delta: A ``LiveOptions`` whose non-None fields are the desired
                overrides.

        Returns:
            Dict mapping each changed key to its **previous** value (same
            contract as ``apply_update``).
        """
        old_dict = self.live_options.to_dict()  # type: ignore[union-attr]
        delta_dict = delta.to_dict()

        # Deepgram SDK bug: model initialised to the *string* "None".
        if delta_dict.get("model") == "None":
            del delta_dict["model"]

        if not delta_dict:
            return {}

        merged = {**old_dict, **delta_dict}
        self.live_options = LiveOptions(**merged)

        # Track what changed.
        changed: Dict[str, Any] = {}
        for key in delta_dict:
            old_val = old_dict.get(key, NOT_GIVEN)
            if old_val != delta_dict[key]:
                changed[key] = old_val

        # Sync model/language from live_options delta to top-level fields.
        if "model" in delta_dict and delta_dict["model"] != self.model:
            changed.setdefault("model", self.model)
            self.model = delta_dict["model"]
        if "language" in delta_dict and delta_dict["language"] != self.language:
            changed.setdefault("language", self.language)
            self.language = delta_dict["language"]

        return changed

    def apply_update(self: _S, delta: _S) -> Dict[str, Any]:
        """Merge a delta into this store, with delta-merge for ``live_options``.

        ``live_options`` is merged field-by-field via
        ``_merge_live_options_delta`` rather than being replaced wholesale.

        ``model`` and ``language`` are kept in sync bidirectionally between
        the top-level settings fields and ``live_options``.
        """
        # Pull live_options out of the delta so super() doesn't replace it.
        delta_lo = getattr(delta, "live_options", NOT_GIVEN)
        if is_given(delta_lo):
            delta.live_options = NOT_GIVEN  # type: ignore[assignment]

        # Let the base class handle model, language, extra.
        changed = super().apply_update(delta)

        # Sync top-level model/language changes into stored live_options.
        if "model" in changed:
            self.live_options.model = self.model  # type: ignore[union-attr]
        if "language" in changed:
            self.live_options.language = self.language  # type: ignore[union-attr]

        # Merge live_options delta.  Top-level model/language take precedence
        # over conflicting values in live_options, so write them into the
        # delta before merging.
        if is_given(delta_lo):
            if "model" in changed:
                delta_lo.model = self.model
            if "language" in changed:
                delta_lo.language = self.language

            for key, old_val in self._merge_live_options_delta(delta_lo).items():
                changed.setdefault(key, old_val)

        return changed

    @classmethod
    def from_mapping(cls: Type[_S], settings: Mapping[str, Any]) -> _S:
        """Build a delta from a plain dict, routing LiveOptions keys correctly.

        Keys that are valid ``LiveOptions.__init__`` parameters (and not
        top-level ``STTSettings`` fields like ``model`` / ``language``) are
        collected into a ``LiveOptions`` object.  ``model`` and ``language``
        are routed to the top-level settings fields.  Truly unknown keys go
        to ``extra``.
        """
        lo_params = cls._get_live_options_params()
        stt_field_names = {"model", "language"}

        kwargs: Dict[str, Any] = {}
        lo_kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}

        for key, value in settings.items():
            canonical = cls._aliases.get(key, key)
            if canonical in stt_field_names:
                kwargs[canonical] = value
            elif canonical in lo_params:
                lo_kwargs[canonical] = value
            else:
                extra[key] = value

        if lo_kwargs:
            kwargs["live_options"] = LiveOptions(**lo_kwargs)

        instance = cls(**kwargs)
        instance.extra = extra
        return instance


@dataclass
class DeepgramSTTSettings(_DeepgramSTTSettingsBase):
    """Settings for the Deepgram STT service.

    See ``_DeepgramSTTSettingsBase`` for full documentation.
    """

    pass


class DeepgramSTTService(STTService):
    """Deepgram speech-to-text service.

    Provides real-time speech recognition using Deepgram's WebSocket API.
    Supports configurable models, languages, and various audio processing options.

    Event handlers available (in addition to STTService events):

    - on_speech_started(service): Deepgram detected start of speech
    - on_utterance_end(service): Deepgram detected end of utterance

    Example::

        @stt.event_handler("on_speech_started")
        async def on_speech_started(service):
            ...
    """

    _settings: DeepgramSTTSettings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "",
        base_url: str = "",
        sample_rate: Optional[int] = None,
        live_options: Optional[LiveOptions] = None,
        addons: Optional[Dict] = None,
        should_interrupt: bool = True,
        ttfs_p99_latency: Optional[float] = DEEPGRAM_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Deepgram STT service.

        Args:
            api_key: Deepgram API key for authentication.
            url: Custom Deepgram API base URL.

                .. deprecated:: 0.0.64
                    Parameter `url` is deprecated, use `base_url` instead.

            base_url: Custom Deepgram API base URL.
            sample_rate: Audio sample rate. If None, uses default or live_options value.
            live_options: Deepgram LiveOptions configuration. Treated as a
                delta from a set of sensible defaults — only the fields you
                set are overridden; all others keep their default values.
            addons: Additional Deepgram features to enable.
            should_interrupt: Determine whether the bot should be interrupted when Deepgram VAD events are enabled and the system detects that the user is speaking.

                .. deprecated:: 0.0.99
                    This parameter will be removed along with `vad_events` support.

            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the parent STTService.

        Note:
            The `vad_events` option in LiveOptions is deprecated as of version 0.0.99 and will be removed in a future version. Please use the Silero VAD instead.
        """
        sample_rate = sample_rate or (live_options.sample_rate if live_options else None)

        if url:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'url' is deprecated, use 'base_url' instead.",
                    DeprecationWarning,
                )
            base_url = url

        default_options = LiveOptions(
            encoding="linear16",
            language=Language.EN,
            model="nova-3-general",
            channels=1,
            interim_results=True,
            smart_format=False,
            punctuate=True,
            profanity_filter=True,
            vad_events=False,
        )

        settings = DeepgramSTTSettings(
            model=default_options.model,
            language=default_options.language,
            live_options=default_options,
        )
        if live_options:
            settings._merge_live_options_delta(live_options)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=settings,
            **kwargs,
        )

        self._addons = addons
        self._should_interrupt = should_interrupt

        if self._settings.live_options.vad_events:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The 'vad_events' parameter is deprecated and will be removed in a future version. "
                    "Please use the Silero VAD instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        self._client = DeepgramClient(
            api_key,
            config=DeepgramClientOptions(
                url=base_url,
                options={"keepalive": "true"},  # verbose=logging.DEBUG
            ),
        )

        if self.vad_enabled:
            self._register_event_handler("on_speech_started")
            self._register_event_handler("on_utterance_end")

    @property
    def vad_enabled(self):
        """Check if Deepgram VAD events are enabled.

        Returns:
            True if VAD events are enabled in the current settings.
        """
        return self._settings.live_options.vad_events

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Deepgram service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if anything changed."""
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        await self._disconnect()
        await self._connect()

        return changed

    async def start(self, frame: StartFrame):
        """Start the Deepgram STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Deepgram STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Deepgram STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Deepgram for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via WebSocket callbacks).
        """
        await self._connection.send(audio)
        yield None

    async def _connect(self):
        logger.debug("Connecting to Deepgram")

        self._connection: AsyncListenWebSocketClient = self._client.listen.asyncwebsocket.v("1")

        self._connection.on(
            LiveTranscriptionEvents(LiveTranscriptionEvents.Transcript), self._on_message
        )
        self._connection.on(LiveTranscriptionEvents(LiveTranscriptionEvents.Error), self._on_error)

        if self.vad_enabled:
            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.SpeechStarted),
                self._on_speech_started,
            )
            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.UtteranceEnd),
                self._on_utterance_end,
            )

        live_options = LiveOptions(
            **{**self._settings.live_options.to_dict(), "sample_rate": self.sample_rate}
        )

        if not await self._connection.start(options=live_options, addons=self._addons):
            await self.push_error(error_msg=f"Unable to connect to Deepgram")
        else:
            headers = {
                k: v
                for k, v in self._connection._socket.response.headers.items()
                if k.startswith("dg-")
            }
            logger.debug(f'{self}: Websocket connection initialized: {{"headers": {headers}}}')

    async def _disconnect(self):
        if await self._connection.is_connected():
            logger.debug("Disconnecting from Deepgram")
            # Deepgram swallows asyncio.CancelledError internally which prevents
            # proper cancellation propagation. This issue was found with
            # parallel pipelines where `CancelFrame` was not awaited for to
            # finish in all branches and it was pushed downstream reaching the
            # end of the pipeline, which caused `cleanup()` to be called while
            # Deepgram disconnection was still finishing and therefore
            # preventing the task cancellation that occurs during `cleanup()`.
            # GH issue: https://github.com/deepgram/deepgram-python-sdk/issues/570
            await self._connection.finish()

    async def _start_metrics(self):
        """Start processing metrics collection for this utterance."""
        await self.start_processing_metrics()

    async def _on_error(self, *args, **kwargs):
        error: ErrorResponse = kwargs["error"]
        logger.warning(f"{self} connection error, will retry: {error}")
        await self.push_error(error_msg=f"{error}")
        await self.stop_all_metrics()
        # NOTE(aleix): we don't disconnect (i.e. call finish on the connection)
        # because this triggers more errors internally in the Deepgram SDK. So,
        # we just forget about the previous connection and create a new one.
        await self._connect()

    async def _on_speech_started(self, *args, **kwargs):
        await self._start_metrics()
        await self._call_event_handler("on_speech_started", *args, **kwargs)
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.push_interruption_task_frame_and_wait()

    async def _on_utterance_end(self, *args, **kwargs):
        await self._call_event_handler("on_utterance_end", *args, **kwargs)
        await self.broadcast_frame(UserStoppedSpeakingFrame)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _on_message(self, *args, **kwargs):
        result: LiveResultResponse = kwargs["result"]
        if len(result.channel.alternatives) == 0:
            return
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        language = None
        if result.channel.alternatives[0].languages:
            language = result.channel.alternatives[0].languages[0]
            language = Language(language)
        if len(transcript) > 0:
            if is_final:
                # Check if this response is from a finalize() call.
                # Only mark as finalized when both we requested it AND Deepgram confirms it.
                from_finalize = getattr(result, "from_finalize", False)
                if from_finalize:
                    self.confirm_finalize()
                await self.push_frame(
                    TranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        result=result,
                    )
                )
                await self._handle_transcription(transcript, is_final, language)
                await self.stop_processing_metrics()
            else:
                # For interim transcriptions, just push the frame without tracing
                await self.push_frame(
                    InterimTranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        result=result,
                    )
                )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with Deepgram-specific handling.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame) and not self.vad_enabled:
            # Start metrics if Deepgram VAD is disabled & pipeline VAD has detected speech
            await self._start_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # https://developers.deepgram.com/docs/finalize
            # Mark that we're awaiting a from_finalize response
            self.request_finalize()
            await self._connection.finalize()
            logger.trace(f"Triggered finalize event on: {frame.name=}, {direction=}")
