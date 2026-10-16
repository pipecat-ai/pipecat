#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime WebSocket v1 text-to-speech service implementation."""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.rime._websocket_v1 import (
    AudioEvent,
    CancelledEvent,
    ConnectionErrorEvent,
    ContextErrorEvent,
    DoneEvent,
    RimeV1ConnectionError,
    RimeV1ProtocolError,
    RimeV1ProviderError,
    RimeWebSocketV1Client,
    StartedEvent,
    SynthesisOptions,
    WebSocketProtocol,
    model_from_websocket_url,
    subprotocol_for_protocol,
)
from pipecat.services.rime.tts import language_to_rime_language
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory, classify_http_exception
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven

_DEFAULT_WEBSOCKET_URL = "wss://api.rime.ai/coda/ws"
_V1_READY_TIMEOUT_S = 10.0
_V1_START_TIMEOUT_S = 10.0
_V1_TERMINAL_TIMEOUT_S = 10.0
_V1_CANCEL_TIMEOUT_S = 1.0


def _resolve_websocket_v1_model(
    websocket_url: str,
    explicit_model: str | None,
    *,
    allow_custom_endpoint: bool,
) -> str:
    """Resolve the model bound to a WebSocket v1 endpoint."""
    endpoint_model = model_from_websocket_url(
        websocket_url, allow_custom_endpoint=allow_custom_endpoint
    )
    if endpoint_model is None:
        if explicit_model is None:
            raise ValueError("Rime WebSocket v1 requires a model for a dedicated endpoint")
        return explicit_model
    if explicit_model is not None and explicit_model != endpoint_model:
        raise ValueError("settings.model does not match the v1 endpoint model")
    return endpoint_model


@dataclass
class RimeV1TTSSettings(TTSSettings):
    """Settings for RimeV1TTSService.

    Each context uses the settings in effect when it starts.

    Parameters:
        time_scale_factor: Audio playback speed factor. Values above 1.0 slow down
            the audio; values below 1.0 speed it up. Not supported by mistv2.
        text_lookahead_tokens: Number of input tokens to collect before audio
            starts (Coda only).
        pause_between_brackets: Whether to add pauses between bracketed content (Mist only).
        phonemize_between_brackets: Whether to phonemize bracketed content (Mist only).
        save_oovs: Whether to save out-of-vocabulary words (Mist only).
    """

    time_scale_factor: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_lookahead_tokens: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pause_between_brackets: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    phonemize_between_brackets: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    save_oovs: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[dict[str, str]] = {"speaker": "voice"}


class RimeV1TTSService(WebsocketTTSService):
    """Text-to-Speech service using Rime's WebSocket v1 API.

    Sends protobuf envelopes, in binary or proto3 JSON encoding, over one
    connection that carries a synthesis context per Pipecat turn. Interruptions
    cancel the turn's context without reconnecting. The model is read from the
    endpoint path (``/coda/ws``, ``/mist/ws``); a dedicated ``/ws`` endpoint
    requires ``settings.model``. WebSocket v1 does not report word timestamps.

    Requires the ``rime`` extra: ``uv add "pipecat-ai[rime]"``.
    """

    Settings = RimeV1TTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        websocket_url: str = _DEFAULT_WEBSOCKET_URL,
        websocket_protocol: Literal["binary", "json"] = "binary",
        allow_custom_endpoint: bool = False,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Rime WebSocket v1 TTS service.

        Args:
            api_key: Rime API key for authentication.
            websocket_url: Rime WebSocket v1 endpoint. Its path selects the model.
            websocket_protocol: Envelope encoding, ``"binary"`` or ``"json"``.
            allow_custom_endpoint: Allow a secure endpoint outside ``rime.ai``.
            sample_rate: Audio sample rate in Hz.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to parent class. Text is always
                aggregated by sentence.
        """
        if websocket_protocol not in ("binary", "json"):
            raise ValueError('websocket_protocol must be "binary" or "json"')
        text_aggregation_mode = kwargs.pop("text_aggregation_mode", None)
        aggregate_sentences = kwargs.pop("aggregate_sentences", None)
        if aggregate_sentences is False or text_aggregation_mode not in (
            None,
            TextAggregationMode.SENTENCE,
        ):
            raise ValueError("Rime WebSocket v1 requires sentence text aggregation")

        default_settings = self.Settings(
            model=None,
            voice=None,
            language=None,
            time_scale_factor=None,
            text_lookahead_tokens=None,
            pause_between_brackets=None,
            phonemize_between_brackets=None,
            save_oovs=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        explicit_model = default_settings.model if isinstance(default_settings.model, str) else None
        endpoint_model = _resolve_websocket_v1_model(
            websocket_url,
            explicit_model,
            allow_custom_endpoint=allow_custom_endpoint,
        )
        default_settings.model = endpoint_model
        self._validate_v1_settings(default_settings)

        super().__init__(
            text_aggregation_mode=TextAggregationMode.SENTENCE,
            push_text_frames=False,
            push_start_frame=True,
            push_stop_frames=False,
            pause_frame_processing=True,
            append_trailing_space=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        # Keep spelled-out text in one aggregate so a sentence split never breaks a tag.
        self._text_aggregator = SkipTagsAggregator(
            [("spell(", ")")], aggregation_type=self._text_aggregation_mode
        )

        self._api_key = api_key
        self._websocket_url = websocket_url
        self._websocket_protocol: WebSocketProtocol = websocket_protocol
        self._v1_endpoint_model = endpoint_model

        self._receive_task = None
        self._v1_client: RimeWebSocketV1Client | None = None
        self._v1_receiving_client: RimeWebSocketV1Client | None = None
        self._v1_options_by_context: dict[str, SynthesisOptions] = {}
        self._v1_keepalive_tasks: dict[str, asyncio.Task[None]] = {}
        self._v1_start_watchdogs: dict[str, asyncio.Task[None]] = {}
        self._v1_terminal_watchdogs: dict[str, asyncio.Task[None]] = {}
        self._v1_cancel_watchdogs: dict[str, asyncio.Task[None]] = {}
        self._v1_failed_contexts: set[str] = set()
        self._v1_closed_contexts: set[str] = set()
        self._v1_invalidating = False
        self._v1_connection_lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Rime service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert pipecat language to Rime language code.

        Args:
            language: The language to convert.

        Returns:
            The Rime-specific language code, or None if not supported.
        """
        return language_to_rime_language(language)

    # A set of Rime-specific helpers for text transformations
    @staticmethod
    def SPELL(text: str) -> str:
        """Wrap text in Rime spell function."""
        return f"spell({text})"

    @staticmethod
    def PAUSE_TAG(seconds: float) -> str:
        """Convenience method to create a pause tag."""
        return f"<{seconds * 1000}>"

    @staticmethod
    def _validate_v1_settings(settings: Settings) -> None:
        """Validate one complete settings snapshot.

        Args:
            settings: Settings to validate.
        """
        if settings.extra:
            names = ", ".join(sorted(settings.extra))
            raise ValueError(f"Rime WebSocket v1 does not support these settings: {names}")

        model = settings.model
        if not isinstance(model, str):
            raise ValueError("Rime WebSocket v1 requires a model")
        mist_fields = (
            settings.pause_between_brackets,
            settings.phonemize_between_brackets,
            settings.save_oovs,
        )
        if not model.startswith("mist") and any(value is not None for value in mist_fields):
            raise ValueError("Rime WebSocket v1 Mist settings require a Mist model")
        if model != "coda" and settings.text_lookahead_tokens is not None:
            raise ValueError("text_lookahead_tokens requires the Coda model")
        if model == "mistv2" and settings.time_scale_factor is not None:
            raise ValueError("time_scale_factor is not supported by the mistv2 model")

    def _build_v1_options(self) -> SynthesisOptions:
        """Build an immutable settings snapshot for a new context."""
        self._validate_v1_settings(self._settings)
        model = self._settings.model
        if not isinstance(model, str):
            raise ValueError("Rime WebSocket v1 requires a model")
        language = self._settings.language
        return SynthesisOptions(
            model=model,
            speaker=self._settings.voice if isinstance(self._settings.voice, str) else None,
            language=language if isinstance(language, str) else None,
            sample_rate=self.sample_rate,
            time_scale_factor=cast(float | None, self._settings.time_scale_factor),
            text_lookahead_tokens=cast(int | None, self._settings.text_lookahead_tokens),
            pause_between_brackets=cast(bool | None, self._settings.pause_between_brackets),
            phonemize_between_brackets=cast(bool | None, self._settings.phonemize_between_brackets),
            save_oovs=cast(bool | None, self._settings.save_oovs),
        )

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta to contexts that start after the update."""
        candidate = self._settings.copy()
        candidate.apply_update(self.Settings.from_mapping(delta.given_fields()))
        if candidate.model != self._v1_endpoint_model:
            raise ValueError("settings.model does not match the v1 endpoint model")
        self._validate_v1_settings(candidate)
        return await super()._update_settings(delta)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def _connect(self):
        """Establish websocket connection and start receive task."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close websocket connection and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self) -> None:
        """Connect to Rime v1 and wait until its engine is ready."""
        async with self._v1_connection_lock:
            await self._connect_websocket_locked()

    async def _connect_websocket_locked(self) -> None:
        """Connect to Rime v1 while holding the connection lifecycle lock."""
        websocket = None
        try:
            if self._websocket and self._websocket.state is State.OPEN and self._v1_client:
                return

            protocol = self._websocket_protocol
            subprotocol = subprotocol_for_protocol(protocol)
            websocket = await self._websocket_connect(
                self._websocket_url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                subprotocols=[subprotocol],
            )
            if websocket.subprotocol != subprotocol:
                raise RimeV1ProtocolError("Rime v1 selected an unexpected WebSocket subprotocol")

            client = RimeWebSocketV1Client(websocket, protocol=protocol)
            await client.wait_ready(_V1_READY_TIMEOUT_S)
            self._websocket = websocket
            self._v1_client = client
            self._v1_invalidating = False
            self._v1_closed_contexts.clear()
            await self._call_event_handler("on_connected")
        except Exception as e:
            if websocket is not None:
                await websocket.close()
            self._websocket = None
            self._v1_client = None
            category = self._v1_exception_category(e)
            message = "Failed to connect to Rime WebSocket v1"
            await self.push_error(error_msg=message, category=category)
            await self._call_event_handler("on_connection_error", message)

    async def _disconnect_websocket(self) -> None:
        """Close Rime v1 and clear all connection-scoped state."""
        async with self._v1_connection_lock:
            await self._disconnect_websocket_locked()

    async def _disconnect_websocket_locked(self) -> None:
        """Close Rime v1 while holding the connection lifecycle lock."""
        try:
            await self.stop_all_metrics()
            await self._cancel_all_v1_tasks()
            client = self._v1_client
            websocket = self._websocket
            self._v1_client = None
            self._v1_receiving_client = None
            self._websocket = None
            if client:
                await client.close()
            elif websocket:
                await websocket.close()
        except Exception:
            await self.push_error(
                error_msg="Failed to disconnect from Rime WebSocket v1",
                category=ErrorCategory.CONNECTIVITY,
            )
        finally:
            for context_id in self.get_audio_contexts():
                if context_id not in self._v1_closed_contexts and self.audio_context_available(
                    context_id
                ):
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                    await self.remove_audio_context(context_id)
            self._v1_options_by_context.clear()
            self._v1_closed_contexts.clear()
            await self._call_event_handler("on_disconnected")

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        """Reconnect the socket without replacing a newer ready connection."""
        logger.warning(f"{self} reconnecting (attempt: {attempt_number})")
        async with self._v1_connection_lock:
            receiving_client = self._v1_receiving_client
            if (
                receiving_client is not None
                and self._v1_client is not receiving_client
                and self._v1_client is not None
                and self._websocket is not None
                and self._websocket.state is State.OPEN
            ):
                return True

            await self._disconnect_websocket_locked()
            await self._close_connection_contexts()
            await self._connect_websocket_locked()
            if not await self._verify_connection():
                raise ConnectionError(f"{self} websocket reconnection failed verification")
            return True

    def _get_v1_client(self) -> RimeWebSocketV1Client:
        """Return the ready v1 client or raise a safe connection error."""
        if self._v1_client:
            return self._v1_client
        raise RimeV1ConnectionError("Rime WebSocket v1 is not connected")

    @staticmethod
    def _v1_error_category(kind: str) -> ErrorCategory:
        categories = {
            "invalid_input": ErrorCategory.INVALID_REQUEST,
            "unauthenticated": ErrorCategory.AUTHENTICATION,
            "permission_denied": ErrorCategory.AUTHORIZATION,
            "not_found": ErrorCategory.INVALID_REQUEST,
            "resource_exhausted": ErrorCategory.RATE_LIMIT,
            "timeout": ErrorCategory.CONNECTIVITY,
            "unavailable": ErrorCategory.CONNECTIVITY,
            "unimplemented": ErrorCategory.SERVER,
            "internal": ErrorCategory.SERVER,
        }
        return categories.get(kind, ErrorCategory.UNKNOWN)

    @classmethod
    def _v1_context_error_category(cls, kind: str) -> ErrorCategory:
        """Classify a context error without changing service usability."""
        category = cls._v1_error_category(kind)
        if category.is_permanent:
            return ErrorCategory.UNKNOWN
        return category

    @classmethod
    def _v1_exception_category(cls, exception: Exception) -> ErrorCategory:
        if isinstance(exception, RimeV1ProviderError):
            return cls._v1_error_category(exception.kind)
        if isinstance(exception, RimeV1ProtocolError):
            return ErrorCategory.SERVER
        if isinstance(exception, (RimeV1ConnectionError, ConnectionError, TimeoutError)):
            return ErrorCategory.CONNECTIVITY
        if isinstance(exception, ValueError):
            return ErrorCategory.INVALID_REQUEST
        return classify_http_exception(exception)

    @staticmethod
    def _v1_error_message(kind: str, request_id: str | None) -> str:
        message = f"Rime WebSocket v1 request failed with {kind}"
        if request_id:
            message += f" (request ID: {request_id})"
        return message

    async def _cancel_v1_context_tasks(
        self,
        context_id: str,
        *,
        include_terminal: bool = True,
        include_cancel: bool = True,
    ) -> None:
        """Cancel tracked tasks for one v1 context."""
        task_maps = [self._v1_keepalive_tasks, self._v1_start_watchdogs]
        if include_terminal:
            task_maps.append(self._v1_terminal_watchdogs)
        if include_cancel:
            task_maps.append(self._v1_cancel_watchdogs)

        current = asyncio.current_task()
        for task_map in task_maps:
            task = task_map.pop(context_id, None)
            if task and task is not current:
                await self.cancel_task(task)

    async def _cancel_all_v1_tasks(self) -> None:
        """Cancel every tracked v1 context task."""
        context_ids = set()
        for task_map in (
            self._v1_keepalive_tasks,
            self._v1_start_watchdogs,
            self._v1_terminal_watchdogs,
            self._v1_cancel_watchdogs,
        ):
            context_ids.update(task_map)
        for context_id in context_ids:
            await self._cancel_v1_context_tasks(context_id)

    @staticmethod
    def _remove_current_v1_task(task_map: dict[str, asyncio.Task[None]], context_id: str) -> None:
        """Remove a context task when that same task finishes."""
        if task_map.get(context_id) is asyncio.current_task():
            task_map.pop(context_id, None)

    async def _finish_v1_context(
        self,
        context_id: str,
        *,
        error: ErrorFrame | None = None,
        emit_stop: bool = True,
    ) -> None:
        """Finish protocol and Pipecat state for one context once."""
        first_close = context_id not in self._v1_closed_contexts
        self._v1_closed_contexts.add(context_id)
        await self.stop_ttfb_metrics()
        if error:
            await self.stop_all_metrics()
        await self._cancel_v1_context_tasks(context_id)
        self._v1_options_by_context.pop(context_id, None)

        defer_audio_context_close = False
        if first_close and self.audio_context_available(context_id):
            if error:
                await self.append_to_audio_context(context_id, error)
                defer_audio_context_close = context_id == self._turn_context_id
                if defer_audio_context_close:
                    self._v1_failed_contexts.add(context_id)
            if not defer_audio_context_close:
                if emit_stop:
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                await self.remove_audio_context(context_id)

        if self._v1_client:
            self._v1_client.discard_context(context_id)
        self._v1_closed_contexts.discard(context_id)

    async def _fail_all_v1_contexts(self, message: str, category: ErrorCategory) -> None:
        """Fail each active context after the connection becomes unsafe."""
        context_ids = set(self._v1_options_by_context)
        if self._v1_client:
            context_ids.update(self._v1_client.context_ids)
        for context_id in context_ids:
            await self._finish_v1_context(
                context_id,
                error=ErrorFrame(error=message, category=category),
            )

    async def _invalidate_v1_connection(
        self,
        message: str,
        category: ErrorCategory,
        client: RimeWebSocketV1Client,
    ) -> None:
        """Close the connection owned by a failed v1 operation."""
        async with self._v1_connection_lock:
            if client is not self._v1_client or self._v1_invalidating:
                return
            self._v1_invalidating = True
            websocket = self._websocket
            client.invalidate()
            await self._fail_all_v1_contexts(message, category)
            if websocket:
                await websocket.close()

    def _start_v1_keepalive(self, context_id: str) -> None:
        client = self._v1_client
        if client and client.has_context(context_id) and context_id not in self._v1_keepalive_tasks:
            self._v1_keepalive_tasks[context_id] = self.create_task(
                self._v1_keepalive_loop(context_id),
                name=f"{self}::{context_id}::keepalive",
            )

    async def _v1_keepalive_loop(self, context_id: str) -> None:
        try:
            interval = max(0.01, min(1.0, self._stop_frame_timeout_s / 3))
            while (
                context_id not in self._v1_closed_contexts
                and self._v1_client
                and self._v1_client.has_context(context_id)
            ):
                await asyncio.sleep(interval)
                self._refresh_audio_context(context_id)
        finally:
            self._remove_current_v1_task(self._v1_keepalive_tasks, context_id)

    def _start_v1_start_watchdog(self, context_id: str) -> None:
        client = self._v1_client
        if client and client.has_context(context_id) and context_id not in self._v1_start_watchdogs:
            self._v1_start_watchdogs[context_id] = self.create_task(
                self._v1_start_watchdog(context_id),
                name=f"{self}::{context_id}::start-watchdog",
            )

    async def _v1_start_watchdog(self, context_id: str) -> None:
        client = self._v1_client
        if not client:
            return
        try:
            await client.wait_started(context_id, _V1_START_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._invalidate_v1_connection(
                "Rime v1 did not start a context", ErrorCategory.CONNECTIVITY, client
            )
        finally:
            self._remove_current_v1_task(self._v1_start_watchdogs, context_id)

    def _start_v1_terminal_watchdog(self, context_id: str) -> None:
        client = self._v1_client
        if (
            client
            and client.has_context(context_id)
            and context_id not in self._v1_terminal_watchdogs
        ):
            self._v1_terminal_watchdogs[context_id] = self.create_task(
                self._v1_terminal_watchdog(context_id),
                name=f"{self}::{context_id}::terminal-watchdog",
            )

    async def _v1_terminal_watchdog(self, context_id: str) -> None:
        client = self._v1_client
        if not client:
            return
        try:
            while client.has_context(context_id):
                await client.wait_activity(context_id, _V1_TERMINAL_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._invalidate_v1_connection(
                "Rime v1 did not finish a context", ErrorCategory.CONNECTIVITY, client
            )
        finally:
            self._remove_current_v1_task(self._v1_terminal_watchdogs, context_id)

    def _start_v1_cancel_watchdog(self, context_id: str) -> None:
        client = self._v1_client
        if (
            client
            and client.has_context(context_id)
            and context_id not in self._v1_cancel_watchdogs
        ):
            self._v1_cancel_watchdogs[context_id] = self.create_task(
                self._v1_cancel_watchdog(context_id),
                name=f"{self}::{context_id}::cancel-watchdog",
            )

    async def _v1_cancel_watchdog(self, context_id: str) -> None:
        client = self._v1_client
        if not client:
            return
        try:
            if not client.has_context(context_id):
                return
            await client.wait_terminal(context_id, _V1_CANCEL_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._invalidate_v1_connection(
                "Rime v1 did not cancel a context", ErrorCategory.CONNECTIVITY, client
            )
        finally:
            self._remove_current_v1_task(self._v1_cancel_watchdogs, context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the Rime context and stop metrics when the bot is interrupted."""
        await self.stop_all_metrics()
        self._v1_failed_contexts.discard(context_id)
        await self._cancel_v1_context_tasks(context_id, include_terminal=True, include_cancel=False)
        client = self._v1_client
        if client and client.has_context(context_id):
            self._v1_closed_contexts.add(context_id)
            try:
                await client.cancel(context_id, timeout_s=_V1_CANCEL_TIMEOUT_S)
                self._start_v1_cancel_watchdog(context_id)
            except Exception:
                await self._invalidate_v1_connection(
                    "Failed to cancel a Rime v1 context", ErrorCategory.CONNECTIVITY, client
                )
        else:
            self._v1_options_by_context.pop(context_id, None)
            self._v1_closed_contexts.discard(context_id)
            if client:
                client.discard_context(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def on_turn_context_completed(self) -> None:
        """Close a failed audio context after its Pipecat turn ends."""
        context_id = self._turn_context_id
        await super().on_turn_context_completed()
        if context_id is not None and context_id in self._v1_failed_contexts:
            if self.audio_context_available(context_id):
                await self.append_to_audio_context(
                    context_id, TTSStoppedFrame(context_id=context_id)
                )
                await self.remove_audio_context(context_id)
            self._v1_failed_contexts.discard(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """End input for the context so Rime synthesizes the rest of its text."""
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket:
            return

        client = self._v1_client
        if not client or not client.has_context(flush_id):
            return
        try:
            await client.end(flush_id, timeout_s=_V1_TERMINAL_TIMEOUT_S)
            self._start_v1_terminal_watchdog(flush_id)
        except Exception:
            await self._invalidate_v1_connection(
                "Failed to end a Rime v1 context", ErrorCategory.CONNECTIVITY, client
            )

    async def _receive_messages(self):
        """Map validated v1 events into Pipecat audio contexts."""
        client = self._get_v1_client()
        self._v1_receiving_client = client
        try:
            async for event in client.events():
                if isinstance(event, StartedEvent):
                    continue
                if isinstance(event, AudioEvent):
                    if event.context_id in self._v1_closed_contexts:
                        continue
                    if event.audio and self.audio_context_available(event.context_id):
                        await self.append_to_audio_context(
                            event.context_id,
                            TTSAudioRawFrame(
                                audio=event.audio,
                                sample_rate=self.sample_rate,
                                num_channels=1,
                                context_id=event.context_id,
                            ),
                        )
                    continue
                if isinstance(event, ConnectionErrorEvent):
                    await self.push_error(
                        self._v1_error_message(event.kind, event.request_id),
                        category=self._v1_error_category(event.kind),
                    )
                    continue
                if isinstance(event, ContextErrorEvent):
                    await self._finish_v1_context(
                        event.context_id,
                        error=ErrorFrame(
                            error=self._v1_error_message(event.kind, event.request_id),
                            category=self._v1_context_error_category(event.kind),
                        ),
                    )
                    continue
                if isinstance(event, DoneEvent):
                    await self._finish_v1_context(event.context_id)
                    continue
                if isinstance(event, CancelledEvent):
                    await self._finish_v1_context(event.context_id, emit_stop=False)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self._invalidate_v1_connection(
                "Rime WebSocket v1 connection failed", self._v1_exception_category(e), client
            )
            raise

    async def _split_v1_sentences(self, text: str) -> list[str]:
        """Split prepared text into the sentence units required by WebSocket v1."""
        aggregator = SkipTagsAggregator(
            [("spell(", ")")], aggregation_type=TextAggregationMode.SENTENCE
        )
        sentences = [
            self._prepare_text_for_tts(aggregate.text)
            async for aggregate in aggregator.aggregate(text)
        ]
        remaining = await aggregator.flush()
        if remaining and remaining.text:
            sentences.append(self._prepare_text_for_tts(remaining.text))
        return sentences

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send sentence units to the turn's Rime context.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Error frames when the request fails; audio arrives through the
            receive task.
        """
        client = None
        try:
            if context_id in self._v1_failed_contexts:
                return
            if not self._websocket or self._websocket.state is State.CLOSED or not self._v1_client:
                await self._connect()
            client = self._get_v1_client()
            first_text = context_id not in self._v1_options_by_context
            if first_text:
                self._v1_closed_contexts.discard(context_id)
                self._v1_options_by_context[context_id] = self._build_v1_options()
            options = self._v1_options_by_context[context_id]
            for sentence in await self._split_v1_sentences(text):
                await client.send_text(context_id, options, sentence, timeout_s=_V1_START_TIMEOUT_S)
                if not client.has_context(context_id):
                    return
                if first_text:
                    self._start_v1_keepalive(context_id)
                    self._start_v1_start_watchdog(context_id)
                    first_text = False
            await self.start_tts_usage_metrics(text)
            yield None
        except Exception as e:
            category = self._v1_exception_category(e)
            self._v1_closed_contexts.add(context_id)
            self._v1_options_by_context.pop(context_id, None)
            yield ErrorFrame(error="Rime WebSocket v1 request failed", category=category)
            yield TTSStoppedFrame(context_id=context_id)
            await self.remove_audio_context(context_id)
            self._v1_closed_contexts.discard(context_id)
            if client and category not in (
                ErrorCategory.AUTHENTICATION,
                ErrorCategory.AUTHORIZATION,
                ErrorCategory.INVALID_REQUEST,
            ):
                await self._invalidate_v1_connection(
                    "Rime WebSocket v1 request failed", category, client
                )
