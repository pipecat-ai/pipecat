#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram speech-to-text service implementation."""

import asyncio
from dataclasses import dataclass, field, fields
from typing import Any, AsyncGenerator, Optional

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
from pipecat.services.settings import (
    NOT_GIVEN,
    STTSettings,
    _NotGiven,
    _warn_deprecated_param,
    is_given,
)
from pipecat.services.stt_latency import DEEPGRAM_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from deepgram import AsyncDeepgramClient
    from deepgram.core.events import EventType
    from deepgram.listen.v1.types import (
        ListenV1Results,
        ListenV1SpeechStarted,
        ListenV1UtteranceEnd,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Deepgram, you need to `pip install pipecat-ai[deepgram]`.")
    raise Exception(f"Missing module: {e}")


class LiveOptions:
    """Deepgram live transcription options.

    Compatibility wrapper that mirrors the ``LiveOptions`` class removed in
    deepgram-sdk v6.

    .. deprecated:: 0.0.105
        Use ``settings=DeepgramSTTSettings(...)`` for runtime-updatable fields
        and direct ``__init__`` parameters for connection-level config instead.
    """

    def __init__(
        self,
        *,
        callback: Optional[str] = None,
        callback_method: Optional[str] = None,
        channels: Optional[int] = None,
        detect_entities: Optional[bool] = None,
        diarize: Optional[bool] = None,
        dictation: Optional[bool] = None,
        encoding: Optional[str] = None,
        endpointing: Optional[Any] = None,
        extra: Optional[Any] = None,
        interim_results: Optional[bool] = None,
        keyterm: Optional[Any] = None,
        keywords: Optional[Any] = None,
        language: Optional[str] = None,
        mip_opt_out: Optional[bool] = None,
        model: Optional[str] = None,
        multichannel: Optional[bool] = None,
        numerals: Optional[bool] = None,
        profanity_filter: Optional[bool] = None,
        punctuate: Optional[bool] = None,
        redact: Optional[Any] = None,
        replace: Optional[Any] = None,
        sample_rate: Optional[int] = None,
        search: Optional[Any] = None,
        smart_format: Optional[bool] = None,
        tag: Optional[Any] = None,
        utterance_end_ms: Optional[int] = None,
        vad_events: Optional[bool] = None,
        version: Optional[str] = None,
        **kwargs,
    ):
        """Initialize live transcription options.

        Args:
            callback: Callback URL for async transcription delivery.
            callback_method: HTTP method to use for the callback (``"GET"`` or ``"POST"``).
            channels: Number of audio channels.
            detect_entities: Enable named entity detection.
            diarize: Enable speaker diarization.
            dictation: Enable dictation mode (converts commands to punctuation).
            encoding: Audio encoding (e.g. ``"linear16"``).
            endpointing: Endpointing sensitivity in ms, or ``False`` to disable.
            extra: Additional key-value metadata to attach to the transcription (str or list).
            interim_results: Whether to emit interim transcriptions.
            keyterm: Keyterms to boost (str or list of str).
            keywords: Keywords to boost (str or list of str).
            language: BCP-47 language tag (e.g. ``"en-US"``).
            mip_opt_out: Opt out of model improvement program.
            model: Deepgram model name (e.g. ``"nova-3-general"``).
            multichannel: Enable per-channel transcription for multi-channel audio.
            numerals: Convert spoken numbers to numerals.
            profanity_filter: Filter profanity from transcripts.
            punctuate: Add punctuation to transcripts.
            redact: Redact sensitive information (str or list of redaction types).
            replace: Word replacement rules (str or list).
            sample_rate: Audio sample rate in Hz.
            search: Search terms to highlight (str or list of str).
            smart_format: Apply smart formatting to transcripts.
            tag: Custom billing tag (str or list of str).
            utterance_end_ms: Silence duration in ms before an utterance-end event.
            vad_events: Enable Deepgram VAD speech-started / utterance-end events.
            version: Model version (e.g. ``"latest"``).
            **kwargs: Any additional Deepgram query parameters.
        """
        self.callback = callback
        self.callback_method = callback_method
        self.channels = channels
        self.detect_entities = detect_entities
        self.diarize = diarize
        self.dictation = dictation
        self.encoding = encoding
        self.endpointing = endpointing
        self.extra = extra
        self.interim_results = interim_results
        self.keyterm = keyterm
        self.keywords = keywords
        self.language = language
        self.mip_opt_out = mip_opt_out
        self.model = model
        self.multichannel = multichannel
        self.numerals = numerals
        self.profanity_filter = profanity_filter
        self.punctuate = punctuate
        self.redact = redact
        self.replace = replace
        self.sample_rate = sample_rate
        self.search = search
        self.smart_format = smart_format
        self.tag = tag
        self.utterance_end_ms = utterance_end_ms
        self.vad_events = vad_events
        self.version = version
        self._extra = kwargs

    def __getattr__(self, name: str):
        # Fall back to _extra for any params passed as **kwargs.
        # __getattr__ is only called when normal attribute lookup fails.
        extra = self.__dict__.get("_extra", {})
        try:
            return extra[name]
        except KeyError:
            raise AttributeError(f"'LiveOptions' object has no attribute '{name}'")

    def to_dict(self) -> dict:
        """Return a dict of all non-None options."""
        result = {k: v for k, v in vars(self).items() if not k.startswith("_") and v is not None}
        result.update({k: v for k, v in self._extra.items() if v is not None})
        return result


@dataclass
class DeepgramSTTSettings(STTSettings):
    """Settings for Deepgram STT services.

    ``model`` and ``language`` are inherited from ``STTSettings`` /
    ``ServiceSettings``.  Additional Deepgram connection params may
    be passed in through ``extra`` (also inherited).

    Parameters:
        detect_entities: Enable named entity detection.
        diarize: Enable speaker diarization.
        dictation: Enable dictation mode (converts commands to punctuation).
        endpointing: Endpointing sensitivity in ms, or ``False`` to disable.
        interim_results: Whether to emit interim transcriptions.
        keyterm: Keyterms to boost (str or list of str).
        keywords: Keywords to boost (str or list of str).
        numerals: Convert spoken numbers to numerals.
        profanity_filter: Filter profanity from transcripts.
        punctuate: Add punctuation to transcripts.
        redact: Redact sensitive information (str or list of redaction types).
        replace: Word replacement rules (str or list).
        search: Search terms to highlight (str or list of str).
        smart_format: Apply smart formatting to transcripts.
        utterance_end_ms: Silence duration in ms before an utterance-end event.
        vad_events: Enable Deepgram VAD speech-started / utterance-end events.
    """

    detect_entities: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    diarize: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    dictation: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    endpointing: Any | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    interim_results: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    keyterm: Any | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    keywords: Any | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    numerals: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    profanity_filter: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    punctuate: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    redact: Any | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    replace: Any | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    search: Any | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    smart_format: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    utterance_end_ms: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    vad_events: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    def _sync_extra_to_fields(self) -> None:
        """Sync values from extra dict to declared fields.

        If a key in extra matches a field name and the field is NOT_GIVEN,
        promote the extra value to the field. This ensures self._settings
        always reflects the "final truth" of values that will be used.

        Keys in extra that match declared fields are always removed from extra
        to avoid confusion, even if the field was already set.
        """
        if not self.extra:
            return

        field_names = {
            f.name
            for f in fields(self)
            if f.name not in ("extra", "model", "language") and not f.name.startswith("_")
        }

        for key in list(self.extra.keys()):
            if key in field_names:
                current_value = getattr(self, key)
                if not is_given(current_value):
                    # Promote extra value to the field
                    setattr(self, key, self.extra[key])
                # Always remove from extra to avoid ambiguity
                del self.extra[key]


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
        encoding: str = "linear16",
        channels: int = 1,
        multichannel: bool = False,
        sample_rate: Optional[int] = None,
        callback: Optional[str] = None,
        callback_method: Optional[str] = None,
        tag: Optional[Any] = None,
        mip_opt_out: Optional[bool] = None,
        live_options: Optional[LiveOptions] = None,
        addons: Optional[dict] = None,
        should_interrupt: bool = True,
        settings: Optional[DeepgramSTTSettings] = None,
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
            encoding: Audio encoding format. Defaults to "linear16".
            channels: Number of audio channels. Defaults to 1.
            multichannel: Transcribe each audio channel independently.
                Defaults to False.
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                sample rate.
            callback: Callback URL for async transcription delivery.
            callback_method: HTTP method for the callback (``"GET"`` or ``"POST"``).
            tag: Custom billing tag.
            mip_opt_out: Opt out of Deepgram model improvement program.
            live_options: Legacy configuration options.

                .. deprecated:: 0.0.105
                    Use ``settings=DeepgramSTTSettings(...)`` for runtime-updatable
                    fields and direct init parameters for connection-level config.

            addons: Additional Deepgram features to enable.
            should_interrupt: Whether to interrupt the bot when Deepgram VAD
                detects the user is speaking.

                .. deprecated:: 0.0.99
                    This parameter will be removed along with `vad_events` support.

            settings: Runtime-updatable settings. When provided alongside
                ``live_options``, ``settings`` values take precedence (applied
                after the ``live_options`` merge).
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the parent STTService.

        Note:
            The `vad_events` option in LiveOptions is deprecated as of version 0.0.99 and will be removed in a future version. Please use the Silero VAD instead.
        """
        if url:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'url' is deprecated, use 'base_url' instead.",
                    DeprecationWarning,
                )
            base_url = url

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = DeepgramSTTSettings(
            model="nova-3-general",
            language=Language.EN,
            detect_entities=False,
            diarize=False,
            dictation=False,
            endpointing=None,
            interim_results=True,
            keyterm=None,
            keywords=None,
            numerals=False,
            profanity_filter=True,
            punctuate=True,
            redact=None,
            replace=None,
            search=None,
            smart_format=False,
            utterance_end_ms=None,
            vad_events=False,
        )

        # 2. Apply live_options overrides — only if settings not provided
        if live_options is not None:
            _warn_deprecated_param("live_options", DeepgramSTTSettings)
            if not settings:
                # Extract init-only fields from live_options
                if live_options.sample_rate is not None and sample_rate is None:
                    sample_rate = live_options.sample_rate
                if live_options.encoding is not None:
                    encoding = live_options.encoding
                if live_options.channels is not None:
                    channels = live_options.channels
                if live_options.callback is not None:
                    callback = live_options.callback
                if live_options.callback_method is not None:
                    callback_method = live_options.callback_method
                if live_options.tag is not None:
                    tag = live_options.tag
                if live_options.mip_opt_out is not None:
                    mip_opt_out = live_options.mip_opt_out
                if live_options.multichannel is not None:
                    multichannel = live_options.multichannel

                # Build settings delta from remaining fields
                init_only = {
                    "sample_rate",
                    "encoding",
                    "channels",
                    "multichannel",
                    "callback",
                    "callback_method",
                    "tag",
                    "mip_opt_out",
                }
                lo_dict = {k: v for k, v in live_options.to_dict().items() if k not in init_only}
                delta = DeepgramSTTSettings.from_mapping(lo_dict)
                default_settings.apply_update(delta)

        # 3. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        # Sync extra to top-level fields so self._settings is unambiguous
        default_settings._sync_extra_to_fields()

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._addons = addons
        self._should_interrupt = should_interrupt
        self._encoding = encoding
        self._channels = channels
        self._multichannel = multichannel
        self._callback = callback
        self._callback_method = callback_method
        self._tag = tag
        self._mip_opt_out = mip_opt_out

        if self._settings.vad_events:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The 'vad_events' parameter is deprecated and will be removed in a future version. "
                    "Please use the Silero VAD instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        # Build client - support optional custom base URL via DeepgramClientEnvironment
        if base_url:
            try:
                from deepgram import DeepgramClientEnvironment

                ws_url = base_url if base_url.startswith("wss://") else f"wss://{base_url}"
                http_url = base_url if base_url.startswith("https://") else f"https://{base_url}"
                environment = DeepgramClientEnvironment(
                    base=http_url,
                    production=ws_url,
                    agent=ws_url,
                )
                self._client = AsyncDeepgramClient(api_key=api_key, environment=environment)
            except Exception:
                logger.warning(
                    f"{self}: Custom base_url configuration failed, falling back to default"
                )
                self._client = AsyncDeepgramClient(api_key=api_key)
        else:
            self._client = AsyncDeepgramClient(api_key=api_key)

        self._connection = None
        self._connection_task = None

        if self.vad_enabled:
            self._register_event_handler("on_speech_started")
            self._register_event_handler("on_utterance_end")

    @property
    def vad_enabled(self):
        """Check if Deepgram VAD events are enabled.

        Returns:
            True if VAD events are enabled in the current settings.
        """
        return self._settings.vad_events

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

        # Sync extra to fields after the update so self._settings stays unambiguous
        if isinstance(self._settings, DeepgramSTTSettings):
            self._settings._sync_extra_to_fields()

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
        if self._connection:
            await self._connection.send_media(audio)
        yield None

    def _build_connect_kwargs(self) -> dict:
        """Build keyword arguments for ``client.listen.v1.connect()`` from current settings."""
        kwargs = {}
        s = self._settings

        # Declared Deepgram-specific fields
        for f in fields(s):
            if f.name in ("model", "language", "extra") or f.name.startswith("_"):
                continue
            value = getattr(s, f.name)
            if not is_given(value) or value is None:
                continue
            kwargs[f.name] = str(value).lower() if isinstance(value, bool) else str(value)

        # model and language
        if is_given(s.model) and s.model is not None:
            kwargs["model"] = str(s.model)
        if is_given(s.language) and s.language is not None:
            kwargs["language"] = str(s.language)

        # Init-only connection config
        kwargs["encoding"] = self._encoding
        kwargs["channels"] = str(self._channels)
        kwargs["multichannel"] = str(self._multichannel).lower()
        kwargs["sample_rate"] = str(self.sample_rate)

        if self._callback is not None:
            kwargs["callback"] = self._callback
        if self._callback_method is not None:
            kwargs["callback_method"] = self._callback_method
        if self._tag is not None:
            kwargs["tag"] = str(self._tag)
        if self._mip_opt_out is not None:
            kwargs["mip_opt_out"] = str(self._mip_opt_out).lower()

        # Any remaining values in extra (that didn't map to declared fields)
        for key, value in s.extra.items():
            if value is not None:
                kwargs[key] = str(value).lower() if isinstance(value, bool) else str(value)

        if self._addons:
            for key, value in self._addons.items():
                kwargs[key] = str(value)

        return kwargs

    async def _connect(self):
        logger.debug("Connecting to Deepgram")
        self._connection_task = self.create_task(self._connection_handler())

    async def _disconnect(self):
        if not self._connection_task:
            return

        logger.debug("Disconnecting from Deepgram")
        # Ask Deepgram to close the stream gracefully before cancelling the task.
        if self._connection:
            await self._connection.send_close_stream()

        await self.cancel_task(self._connection_task)
        self._connection_task = None
        self._connection = None

    async def _connection_handler(self):
        """Manages the full WebSocket lifecycle inside a single async with block.

        Reconnects automatically after transient errors. Exits cleanly when
        the task is cancelled (i.e. on stop/cancel).
        """
        while True:
            connect_kwargs = self._build_connect_kwargs()
            try:
                async with self._client.listen.v1.connect(**connect_kwargs) as connection:
                    self._connection = connection
                    connection.on(EventType.MESSAGE, self._on_message)
                    connection.on(EventType.ERROR, self._on_error)

                    logger.debug(f"{self}: Websocket connection initialized")

                    keepalive_task = self.create_task(
                        self._keepalive_handler(), f"{self}::keepalive"
                    )
                    try:
                        await connection.start_listening()
                    finally:
                        await self.cancel_task(keepalive_task)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"{self}: Connection lost, will retry: {e}")
            finally:
                self._connection = None

    async def _keepalive_handler(self):
        """Periodically send KeepAlive frames to prevent server-side timeout.

        Deepgram closes inactive connections after 10 seconds (NET-0001 error).
        Sending every 5 seconds stays within the recommended 3-5 second interval.
        """
        while True:
            await asyncio.sleep(5)
            if self._connection:
                try:
                    await self._connection.send_keep_alive()
                    logger.trace(f"{self}: Sent keepalive")
                except Exception as e:
                    logger.warning(f"{self}: Keepalive failed: {e}")

    async def _start_metrics(self):
        """Start processing metrics collection for this utterance."""
        await self.start_processing_metrics()

    async def _on_error(self, error):
        logger.warning(f"{self} connection error, will retry: {error}")
        await self.push_error(error_msg=f"{error}")
        await self.stop_all_metrics()
        # Reconnection is handled automatically by the retry loop in
        # _connection_handler once start_listening() exits after the error.

    async def _on_speech_started(self, message):
        await self._start_metrics()
        await self._call_event_handler("on_speech_started", message)
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.broadcast_interruption()

    async def _on_utterance_end(self, message):
        await self._call_event_handler("on_utterance_end", message)
        await self.broadcast_frame(UserStoppedSpeakingFrame)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _on_message(self, message):
        if isinstance(message, ListenV1SpeechStarted):
            if self.vad_enabled:
                await self._on_speech_started(message)
        elif isinstance(message, ListenV1UtteranceEnd):
            if self.vad_enabled:
                await self._on_utterance_end(message)
        elif isinstance(message, ListenV1Results):
            if not message.channel or len(message.channel.alternatives) == 0:
                return
            is_final = message.is_final
            transcript = message.channel.alternatives[0].transcript
            language = None
            if message.channel.alternatives[0].languages:
                language = message.channel.alternatives[0].languages[0]
                language = Language(language)
            if len(transcript) > 0:
                if is_final:
                    # Check if this response is from a finalize() call.
                    # Only mark as finalized when both we requested it AND Deepgram confirms it.
                    from_finalize = getattr(message, "from_finalize", False) or False
                    if from_finalize:
                        self.confirm_finalize()
                    await self.push_frame(
                        TranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                            language,
                            result=message,
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
                            result=message,
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
            if self._connection:
                self.request_finalize()
                await self._connection.send_finalize()
                logger.trace(f"Triggered finalize event on: {frame.name=}, {direction=}")
