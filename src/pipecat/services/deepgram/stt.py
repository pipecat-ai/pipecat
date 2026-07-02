#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram speech-to-text service implementation."""

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field, fields
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import (
    NOT_GIVEN,
    STTSettings,
    _NotGiven,
    is_given,
)
from pipecat.services.stt_latency import DEEPGRAM_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.deprecation import deprecated
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from deepgram import AsyncDeepgramClient
    from deepgram.core.events import EventType
    from deepgram.core.request_options import RequestOptions
    from deepgram.listen.v1.types import ListenV1Results
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Deepgram, you need to `uv add "pipecat-ai[deepgram]"`.')
    raise ImportError(f"Missing module: {e}") from e


@deprecated(
    "`LiveOptions` is deprecated since 0.0.105 and will be removed in 2.0.0. Use "
    "`DeepgramSTTService.Settings` instead."
)
class LiveOptions:
    """Deepgram live transcription options.

    Compatibility wrapper that mirrors the ``LiveOptions`` class removed in
    deepgram-sdk v6.

    .. deprecated:: 0.0.105
        Use ``settings=DeepgramSTTService.Settings(...)`` for runtime-updatable fields
        and direct ``__init__`` parameters for connection-level config instead.
        Will be removed in 2.0.0.
    """

    def __init__(
        self,
        *,
        callback: str | None = None,
        callback_method: str | None = None,
        channels: int | None = None,
        detect_entities: bool | None = None,
        diarize: bool | None = None,
        dictation: bool | None = None,
        encoding: str | None = None,
        endpointing: Any | None = None,
        extra: Any | None = None,
        interim_results: bool | None = None,
        keyterm: Any | None = None,
        keywords: Any | None = None,
        language: str | None = None,
        mip_opt_out: bool | None = None,
        model: str | None = None,
        multichannel: bool | None = None,
        numerals: bool | None = None,
        profanity_filter: bool | None = None,
        punctuate: bool | None = None,
        redact: Any | None = None,
        replace: Any | None = None,
        sample_rate: int | None = None,
        search: Any | None = None,
        smart_format: bool | None = None,
        tag: Any | None = None,
        utterance_end_ms: int | None = None,
        version: str | None = None,
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
    """Settings for DeepgramSTTService.

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


def _derive_deepgram_urls(base_url: str) -> tuple[str, str]:
    """Derive paired WebSocket and HTTP URLs from a single base URL.

    The Deepgram SDK client requires both a WebSocket URL (for streaming)
    and an HTTP URL (for REST calls). This helper lets developers provide
    a single ``base_url`` and consistently derives both, preserving the
    security level they chose. Useful for air-gapped or private deployments
    where insecure schemes (ws:// / http://) are acceptable.

    Accepted inputs:
        - ``wss://`` or ``https://`` — secure (paired as wss + https)
        - ``ws://`` or ``http://`` — insecure (paired as ws + http)
        - Bare hostname (no scheme) — defaults to secure
        - Unrecognized scheme — logs a warning, defaults to secure

    Args:
        base_url: Host with optional scheme, port, and path.

    Returns:
        A (ws_url, http_url) tuple with consistent schemes.
    """
    known_schemes = ("wss://", "https://", "ws://", "http://")
    if "://" in base_url:
        scheme, host = base_url.split("://", 1)
        scheme += "://"
        if scheme not in known_schemes:
            logger.warning(
                f"Unrecognized scheme in base_url '{base_url}', defaulting to wss:// / https://"
            )
    else:
        scheme = ""
        host = base_url

    insecure = scheme in ("ws://", "http://")
    ws_url = f"{'ws' if insecure else 'wss'}://{host}"
    http_url = f"{'http' if insecure else 'https'}://{host}"
    return ws_url, http_url


class DeepgramSTTService(STTService):
    """Deepgram speech-to-text service.

    Provides real-time speech recognition using Deepgram's WebSocket API.
    Supports configurable models, languages, and various audio processing options.
    """

    Settings = DeepgramSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "",
        encoding: str = "linear16",
        channels: int = 1,
        multichannel: bool = False,
        sample_rate: int | None = None,
        callback: str | None = None,
        callback_method: str | None = None,
        tag: Any | None = None,
        mip_opt_out: bool | None = None,
        live_options: LiveOptions | None = None,
        addons: dict | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = DEEPGRAM_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Deepgram STT service.

        Args:
            api_key: Deepgram API key for authentication.
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
                    Use ``settings=DeepgramSTTService.Settings(...)`` for runtime-updatable
                    fields and direct init parameters for connection-level config.
                    Will be removed in 2.0.0.

            addons: Additional Deepgram features to enable.
            settings: Runtime-updatable settings. When provided alongside
                ``live_options``, ``settings`` values take precedence (applied
                after the ``live_options`` merge).
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the parent STTService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
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
        )

        # 2. (No step 2, as there are no deprecated direct args)

        # 3. Apply live_options overrides — only if settings not provided
        if live_options is not None:
            self._warn_init_param_moved_to_settings("live_options")
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
                delta = self.Settings.from_mapping(lo_dict)
                default_settings.apply_update(delta)

        # 4. Apply settings delta (canonical API, always wins)
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
        self._encoding = encoding
        self._channels = channels
        self._multichannel = multichannel
        self._callback = callback
        self._callback_method = callback_method
        self._tag = tag
        self._mip_opt_out = mip_opt_out

        # Build client - support optional custom base URL via DeepgramClientEnvironment
        if base_url:
            try:
                from deepgram import DeepgramClientEnvironment

                ws_url, http_url = _derive_deepgram_urls(base_url)
                environment = DeepgramClientEnvironment(
                    base=http_url,
                    production=ws_url,
                    agent=ws_url,
                    agent_rest=http_url,
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
        self._audio_sender_task = None
        self._watchdog_task = None
        self._connection_ready = asyncio.Event()
        self._disconnecting = False
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._last_sent_time: float = 0.0
        self._handler_id: int = 0

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Deepgram service supports metrics generation.
        """
        return True

    async def _do_reconnect(self):
        """Disconnect and reconnect to Deepgram, waiting until ready.

        Called by ``STTService._reconnect()`` inside the reconnecting guard.
        Unlike ``WebsocketSTTService``, Deepgram's ``_connect()`` only
        launches a background task — the actual WebSocket handshake happens
        asynchronously. This method waits for ``_connection_ready`` to be set
        before returning so that buffered audio frames are replayed only after
        the new connection can accept them.

        Raises:
            asyncio.TimeoutError: If the connection is not established within
                05 seconds.
        """
        await self._disconnect()
        await self._connect()
        await asyncio.wait_for(self._connection_ready.wait(), timeout=5.0)

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if anything changed."""
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        # Sync extra to fields after the update so self._settings stays unambiguous
        if isinstance(self._settings, self.Settings):
            self._settings._sync_extra_to_fields()

        await self._request_reconnect()

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

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Enqueue audio for sending to Deepgram.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via WebSocket callbacks).
        """
        await self._audio_queue.put(audio)
        yield None

    def _build_connect_kwargs(self) -> tuple[dict, RequestOptions | None]:
        """Build arguments for ``client.listen.v1.connect()`` from current settings.

        Returns a (kwargs, request_options) tuple. ``kwargs`` contains the
        parameters accepted directly by the v7 ``connect()`` method. Any
        unrecognised keys (from ``settings.extra`` or ``addons``) are placed in
        ``request_options["additional_query_parameters"]`` so the SDK still
        forwards them as query-string parameters.
        """
        kwargs = {}
        extra_query_params: dict = {}
        s = self._settings

        # Declared Deepgram-specific fields — all are known v7 connect() params.
        for f in fields(s):
            if f.name in ("model", "language", "extra") or f.name.startswith("_"):
                continue
            value = getattr(s, f.name)
            if not is_given(value) or value is None:
                continue
            # Lists (e.g. keyterm, keywords, search, redact, replace) must be
            # passed through as-is so the SDK's encode_query produces repeated
            # query params (keyterm=a&keyterm=b) instead of a stringified list.
            if isinstance(value, list):
                kwargs[f.name] = value
            elif isinstance(value, bool):
                kwargs[f.name] = str(value).lower()
            else:
                kwargs[f.name] = str(value)

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

        # Any remaining values in settings.extra and addons are not in the v7
        # connect() signature; route them through request_options.
        for key, value in s.extra.items():
            if value is not None:
                if isinstance(value, list):
                    extra_query_params[key] = value
                elif isinstance(value, bool):
                    extra_query_params[key] = str(value).lower()
                else:
                    extra_query_params[key] = str(value)

        if self._addons:
            for key, value in self._addons.items():
                extra_query_params[key] = str(value)

        request_options: RequestOptions | None = (
            RequestOptions(additional_query_parameters=extra_query_params)
            if extra_query_params
            else None
        )
        return kwargs, request_options

    async def _connect(self):
        logger.debug("Connecting to Deepgram")
        self._disconnecting = False
        self._audio_queue = asyncio.Queue()
        self._handler_id += 1
        self._connection_task = self.create_task(self._connection_handler())
        self._audio_sender_task = self.create_task(
            self._audio_sender_handler(), f"{self}::audio_sender"
        )
        self._watchdog_task = self.create_task(self._watchdog_handler(), f"{self}::watchdog")

    async def _disconnect(self):
        if not self._connection_task:
            return

        logger.debug("Disconnecting from Deepgram")
        self._disconnecting = True
        # Clear _connection and _connection_ready first to prevent run_stt
        # from sending audio during the close handshake, and to ensure any
        # concurrent _do_reconnect() waiter sees a clean state before the
        # new connection is established.
        self._connection_ready.clear()
        connection = self._connection
        self._connection = None

        if connection:
            await connection.send_close_stream()

        # Both tasks are cancelled without awaiting for the same reason: if the
        # WebSocket connection is dead, send_media() and websockets' close
        # handshake can block for many seconds and won't respond to CancelledError
        # promptly. Fire-and-forget; each task cleans itself
        # up via the task manager's done callback when it eventually exits.
        if self._audio_sender_task:
            self._audio_sender_task.cancel()
            self._audio_sender_task = None

        if self._watchdog_task:
            await self.cancel_task(self._watchdog_task)
            self._watchdog_task = None

        # _handler_id prevents the stale connection task from touching the new
        # connection's state once it eventually exits.
        if self._connection_task:
            self._connection_task.cancel()
            self._connection_task = None

    async def _connection_handler(self):
        """Manages a single WebSocket connection lifetime.

        Exits cleanly when cancelled (intentional stop/cancel). For unexpected
        drops, reconnection is driven by _on_close.
        """
        logger.info(f"{self}: connecting")
        connect_kwargs, request_options = self._build_connect_kwargs()
        handler_id = self._handler_id
        try:
            async with self._client.listen.v1.connect(
                **connect_kwargs, request_options=request_options
            ) as connection:
                self._connection = connection
                connection.on(EventType.OPEN, self._on_open)
                connection.on(EventType.MESSAGE, self._on_message)
                connection.on(EventType.ERROR, self._on_error)
                connection.on(EventType.CLOSE, self._on_close)

                await connection.start_listening()
        except Exception as e:
            logger.warning(f"{self}: Connection failed: {e}")
        finally:
            # Only touch shared state if we are still the current handler.
            # A stale task can outlive _disconnect() — _handler_id tells us apart.
            if self._handler_id == handler_id:
                self._connection_ready.clear()
                self._connection = None

    async def _audio_sender_handler(self):
        """Drain the audio queue and forward each chunk to Deepgram.

        This is the only coroutine that writes audio to the WebSocket, eliminating
        concurrent-write races between audio and keepalive frames.
        """
        while True:
            audio = await self._audio_queue.get()
            if self._connection and self._connection_ready.is_set():
                try:
                    await self._connection.send_media(audio)
                    self._last_sent_time = time.monotonic()
                except Exception as e:
                    logger.warning(f"{self}: send_media failed: {e}")
                    self._connection = None
            # If not connected, drop the chunk — the base class buffers frames
            # during reconnect and will replay them via run_stt() once ready.
            self._audio_queue.task_done()

    async def _watchdog_handler(self):
        """Detect a stuck send_media or idle connection and act accordingly.

        Checks every 500 ms. If nothing has been sent for more than 1 second:
        - Queue non-empty  → send_media is likely blocked → trigger reconnect.
        - Queue empty      → connection is idle → send a KeepAlive.

        Deepgram closes inactive connections after 10 s (NET-0001). Sending a
        KeepAlive when idle keeps the session alive without racing with audio.
        """
        stuck_threshold = 1.0
        check_interval = 0.5
        while not self._disconnecting:
            await asyncio.sleep(check_interval)
            if not self._connection_ready.is_set():
                continue
            if time.monotonic() - self._last_sent_time <= stuck_threshold:
                continue
            if not self._audio_queue.empty():
                # Items are waiting but nothing has been sent — sender is stuck.
                logger.warning(f"{self}: watchdog detected stuck send_media, reconnecting")
                self._connection = None
                self.create_task(self._reconnect())
                return
            # Idle connection — send a KeepAlive to prevent server-side timeout.
            if self._connection:
                try:
                    await asyncio.wait_for(self._connection.send_keep_alive(), timeout=1.0)
                    self._last_sent_time = time.monotonic()
                except TimeoutError:
                    logger.warning(f"{self}: keepalive timed out, reconnecting")
                    self._connection = None
                    self.create_task(self._reconnect())
                    return
                except Exception as e:
                    logger.warning(f"{self}: keepalive failed: {e}")

    async def _start_metrics(self):
        """Start processing metrics collection for this utterance."""
        await self.start_processing_metrics()

    async def _on_open(self, open):
        logger.debug(f"{self}: connection opened")
        self._last_sent_time = time.monotonic()
        self._connection_ready.set()

    async def _on_error(self, error):
        logger.warning(f"{self} connection error: {error}")
        await self.push_error(error_msg=f"{error}")
        await self.stop_all_metrics()
        # _on_close always follows an error and will drive reconnection.

    async def _on_close(self, close):
        logger.debug(f"{self}: connection closed")
        self._connection = None
        if not self._disconnecting and not self._reconnecting:
            # Schedule reconnect as a separate task and return immediately so
            # _connection_handler can finish __aexit__ cleanly. Awaiting
            # _reconnect() here would call _disconnect() while
            # _connection_handler is still alive, hitting the same
            # self-cancellation/timeout issue as the keepalive handler.
            self.create_task(self._reconnect())

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _on_message(self, message):
        if isinstance(message, ListenV1Results):
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

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._start_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # https://developers.deepgram.com/docs/finalize
            # Mark that we're awaiting a from_finalize response
            if self._connection:
                self.request_finalize()
                await self._connection.send_finalize()
                logger.trace(f"Triggered finalize event on: {frame.name=}, {direction=}")
