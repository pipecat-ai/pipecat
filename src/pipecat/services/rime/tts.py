#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime text-to-speech service implementations.

This module provides both WebSocket and HTTP-based text-to-speech services
using Rime's API for streaming and batch audio synthesis.
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

import aiohttp
from loguru import logger
from pydantic import BaseModel
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
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
    TerminalEvent,
    WebSocketProtocol,
    model_from_websocket_url,
    subprotocol_for_protocol,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import (
    InterruptibleTTSService,
    TextAggregationMode,
    TTSService,
    WebsocketTTSService,
)
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.deprecation import deprecated
from pipecat.utils.errors import ErrorCategory, classify_http_exception
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given

_LEGACY_WEBSOCKET_URL = "wss://users-ws.rime.ai/ws3"
_V1_READY_TIMEOUT_S = 10.0
_V1_START_TIMEOUT_S = 10.0
_V1_TERMINAL_TIMEOUT_S = 10.0
_V1_CANCEL_TIMEOUT_S = 1.0


def language_to_rime_language(language: Language) -> str:
    """Convert pipecat Language to Rime language code.

    Args:
        language: The pipecat Language enum value.

    Returns:
        Language code used by Rime (e.g., 'eng' for English). Rime accepts both
        ISO 639-2/3 and ISO 639-1 codes, so a region-qualified language falls
        back to its two-letter base code.
    """
    LANGUAGE_MAP = {
        Language.AR: "ara",
        Language.DE: "ger",
        Language.EN: "eng",
        Language.ES: "spa",
        Language.FR: "fra",
        Language.HI: "hin",
        Language.IT: "ita",
        Language.JA: "jpn",
        Language.PT: "por",
    }
    return resolve_language(language, LANGUAGE_MAP)


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
class RimeTTSSettings(TTSSettings):
    """Settings for RimeTTSService and RimeHttpTTSService.

    Parameters:
        segment: Text segmentation mode ("immediate", "bySentence", "never").
        speedAlpha: Speech speed multiplier (mistv2 only).
        reduceLatency: Whether to reduce latency at potential quality cost (mistv2 only).
        pauseBetweenBrackets: Whether to add pauses between bracketed content (mistv2 only).
        phonemizeBetweenBrackets: Whether to phonemize bracketed content (mistv2 only).
        noTextNormalization: Whether to disable text normalization (mistv2 only).
        saveOovs: Whether to save out-of-vocabulary words (mistv2 only).
        inlineSpeedAlpha: Inline speed control markup.
        repetition_penalty: Token repetition penalty for Coda requests (1.0-2.0).
        temperature: Sampling temperature for Coda requests (0.0-1.0).
        top_p: Cumulative probability threshold for Coda requests (0.0-1.0).
        timeScaleFactor: Audio playback speed factor for Coda requests.
            Values above 1.0 slow down the audio; values below 1.0 speed it up.
        text_lookahead_tokens: Number of Coda v1 input tokens to collect before audio starts.
    """

    segment: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speedAlpha: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    reduceLatency: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pauseBetweenBrackets: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    phonemizeBetweenBrackets: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    noTextNormalization: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    saveOovs: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    inlineSpeedAlpha: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    repetition_penalty: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    timeScaleFactor: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_lookahead_tokens: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[dict[str, str]] = {"speaker": "voice"}


@dataclass
class RimeNonJsonTTSSettings(TTSSettings):
    """Settings for RimeNonJsonTTSService.

    Parameters:
        segment: Text segmentation mode ("immediate", "bySentence", "never").
        repetition_penalty: Token repetition penalty (1.0-2.0).
        temperature: Sampling temperature (0.0-1.0).
        top_p: Cumulative probability threshold (0.0-1.0).
    """

    segment: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    repetition_penalty: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[dict[str, str]] = {"speaker": "voice"}


class RimeTTSService(WebsocketTTSService):
    """Text-to-Speech service using Rime's websocket API.

    Uses the legacy WS3 interface by default. Setting ``websocket_url`` selects
    WebSocket v1 with binary or JSON protobuf envelopes. V1 keeps one synthesis
    context for each Pipecat turn and supports cancellation without reconnecting.
    """

    Settings = RimeTTSSettings
    _settings: Settings

    @deprecated(
        "`RimeTTSService.InputParams` is deprecated since 0.0.105 and will be removed in 2.0.0. "
        "Use `RimeTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Configuration parameters for Rime TTS service.

        .. deprecated:: 0.0.105
            Use ``settings=RimeTTSService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Language for synthesis. Defaults to English.
            segment: Text segmentation mode ("immediate", "bySentence", "never").
            speed_alpha: Speech speed multiplier.
            repetition_penalty: Token repetition penalty for Coda requests.
            temperature: Sampling temperature for Coda requests.
            top_p: Cumulative probability threshold for Coda requests.
            reduce_latency: Whether to reduce latency at potential quality cost (mistv2 only).
            pause_between_brackets: Whether to add pauses between bracketed content (mistv2 only).
            phonemize_between_brackets: Whether to phonemize bracketed content (mistv2 only).
            no_text_normalization: Whether to disable text normalization (mistv2 only).
            save_oovs: Whether to save out-of-vocabulary words (mistv2 only).
        """

        language: Language | None = Language.EN
        segment: str | None = None
        speed_alpha: float | None = None
        repetition_penalty: float | None = None
        temperature: float | None = None
        top_p: float | None = None
        # Mistv2 params
        reduce_latency: bool | None = None
        pause_between_brackets: bool | None = None
        phonemize_between_brackets: bool | None = None
        no_text_normalization: bool | None = None
        save_oovs: bool | None = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str | None = None,
        url: str | NotGiven = NOT_GIVEN,
        websocket_url: str | None = None,
        websocket_protocol: Literal["binary", "json"] | None = None,
        allow_custom_endpoint: bool = False,
        model: str | None = None,
        sample_rate: int | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        aggregate_sentences: bool | None = None,
        **kwargs,
    ):
        """Initialize Rime TTS service.

        Args:
            api_key: Rime API key for authentication.
            voice_id: ID of the voice to use.

                .. deprecated:: 0.0.105
                    Use ``settings=RimeTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            url: Legacy Rime WS3 endpoint.
            websocket_url: Rime WebSocket v1 endpoint. Setting this selects v1.
            websocket_protocol: V1 envelope encoding. Defaults to ``"binary"``.
            allow_custom_endpoint: Allow a secure v1 endpoint outside ``rime.ai``.
            model: Model ID to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=RimeTTSService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            sample_rate: Audio sample rate in Hz.
            params: Additional configuration parameters.

                .. deprecated:: 0.0.105
                    Use ``settings=RimeTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            text_aggregation_mode: How to aggregate incoming text before synthesis.
            aggregate_sentences: Deprecated. Use text_aggregation_mode instead.

                .. deprecated:: 0.0.104
                    Use ``text_aggregation_mode`` instead.
                    Will be removed in 2.0.0.

            **kwargs: Additional arguments passed to parent class.
        """
        use_websocket_v1 = websocket_url is not None
        legacy_url_was_given = is_given(url)
        if use_websocket_v1 and legacy_url_was_given:
            raise ValueError("url and websocket_url cannot be used together")
        if not use_websocket_v1 and websocket_protocol is not None:
            raise ValueError("websocket_protocol requires websocket_url")
        if websocket_protocol not in (None, "binary", "json"):
            raise ValueError('websocket_protocol must be "binary" or "json"')
        if use_websocket_v1 and aggregate_sentences is False:
            raise ValueError("Rime WebSocket v1 requires sentence text aggregation")
        if use_websocket_v1 and text_aggregation_mode not in (
            None,
            TextAggregationMode.SENTENCE,
        ):
            raise ValueError("Rime WebSocket v1 requires sentence text aggregation")

        model_was_explicit = model is not None or (
            settings is not None and is_given(settings.model) and settings.model is not None
        )

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="coda",
            voice=None,
            language=None,
            segment=None,
            inlineSpeedAlpha=None,
            speedAlpha=None,
            repetition_penalty=None,
            temperature=None,
            top_p=None,
            # Mistv2 params
            reduceLatency=None,
            pauseBetweenBrackets=None,
            phonemizeBetweenBrackets=None,
            noTextNormalization=None,
            saveOovs=None,
            timeScaleFactor=None,
            text_lookahead_tokens=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.language = params.language
                default_settings.segment = params.segment
                default_settings.speedAlpha = params.speed_alpha
                default_settings.repetition_penalty = params.repetition_penalty
                default_settings.temperature = params.temperature
                default_settings.top_p = params.top_p
                # Mistv2 params
                default_settings.reduceLatency = params.reduce_latency
                default_settings.pauseBetweenBrackets = params.pause_between_brackets
                default_settings.phonemizeBetweenBrackets = params.phonemize_between_brackets
                default_settings.noTextNormalization = params.no_text_normalization
                default_settings.saveOovs = params.save_oovs

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        resolved_websocket_protocol: WebSocketProtocol | None = None
        resolved_v1_model: str | None = None
        if use_websocket_v1:
            assert websocket_url is not None
            resolved_websocket_protocol = cast(WebSocketProtocol, websocket_protocol or "binary")
            explicit_model = cast(str, default_settings.model) if model_was_explicit else None
            resolved_v1_model = _resolve_websocket_v1_model(
                websocket_url,
                explicit_model,
                allow_custom_endpoint=allow_custom_endpoint,
            )
            default_settings.model = resolved_v1_model
            self._validate_v1_settings(default_settings)
            text_aggregation_mode = TextAggregationMode.SENTENCE
        elif default_settings.text_lookahead_tokens is not None:
            raise ValueError("text_lookahead_tokens requires Rime WebSocket v1")

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,
            push_start_frame=use_websocket_v1,
            push_stop_frames=False,
            pause_frame_processing=True,
            append_trailing_space=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        # Init-only audio format fields (not runtime-updatable)
        self._audio_format = "pcm"
        self._sampling_rate = 0  # updated in start()

        # Always skip tags added for spelled-out text
        # Note: This is primarily to support backwards compatibility.
        #    The preferred way of taking advantage of Rime spelling is
        #    to use an LLMTextProcessor and/or a text_transformer to identify
        #    and insert these tags for the purpose of the TTS service alone.
        self._text_aggregator = SkipTagsAggregator(
            [("spell(", ")")], aggregation_type=self._text_aggregation_mode
        )

        # Store service configuration
        self._api_key = api_key
        self._url = url if isinstance(url, str) else _LEGACY_WEBSOCKET_URL
        self._websocket_url = websocket_url
        self._websocket_protocol = resolved_websocket_protocol
        self._v1_endpoint_model = resolved_v1_model
        self._use_websocket_v1 = use_websocket_v1

        # State tracking
        self._receive_task = None
        self._cumulative_time = 0  # Accumulates time across messages
        self._extra_msg_fields = {}  # Extra fields for next message
        self._audio_remainder = b""  # Held-back byte of a sample split across chunks
        self._audio_remainder_context_id = None
        self._v1_client: RimeWebSocketV1Client | None = None
        self._v1_receiving_client: RimeWebSocketV1Client | None = None
        self._v1_options_by_context: dict[str, SynthesisOptions] = {}
        self._v1_keepalive_tasks: dict[str, asyncio.Task[None]] = {}
        self._v1_start_watchdogs: dict[str, asyncio.Task[None]] = {}
        self._v1_terminal_watchdogs: dict[str, asyncio.Task[None]] = {}
        self._v1_cancel_watchdogs: dict[str, asyncio.Task[None]] = {}
        self._v1_audio_remainders: dict[str, bytes] = {}
        self._v1_contexts_with_audio: set[str] = set()
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

    @staticmethod
    def _validate_v1_settings(settings: Settings) -> None:
        """Validate one complete settings snapshot for WebSocket v1.

        Args:
            settings: Settings to validate.
        """
        unsupported = (
            "segment",
            "speedAlpha",
            "inlineSpeedAlpha",
            "reduceLatency",
            "noTextNormalization",
            "repetition_penalty",
            "temperature",
            "top_p",
        )
        invalid = [name for name in unsupported if getattr(settings, name) is not None]
        if settings.extra:
            invalid.extend(sorted(settings.extra))
        if invalid:
            names = ", ".join(invalid)
            raise ValueError(f"Rime WebSocket v1 does not support these settings: {names}")

        model = settings.model
        if not isinstance(model, str):
            raise ValueError("Rime WebSocket v1 requires a model")
        mist_fields = (
            settings.pauseBetweenBrackets,
            settings.phonemizeBetweenBrackets,
            settings.saveOovs,
        )
        if not model.startswith("mist") and any(value is not None for value in mist_fields):
            raise ValueError("Rime WebSocket v1 Mist settings require a Mist model")
        if model.startswith("mist") and settings.text_lookahead_tokens is not None:
            raise ValueError("text_lookahead_tokens requires the Coda model")
        if model == "mistv2" and settings.timeScaleFactor is not None:
            raise ValueError("timeScaleFactor is not supported by the mistv2 model")

    def _build_v1_options(self) -> SynthesisOptions:
        """Build an immutable v1 settings snapshot for a new context."""
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
            time_scale_factor=cast(float | None, self._settings.timeScaleFactor),
            text_lookahead_tokens=cast(int | None, self._settings.text_lookahead_tokens),
            pause_between_brackets=cast(bool | None, self._settings.pauseBetweenBrackets),
            phonemize_between_brackets=cast(bool | None, self._settings.phonemizeBetweenBrackets),
            save_oovs=cast(bool | None, self._settings.saveOovs),
        )

    def _build_ws_params(self) -> dict[str, Any]:
        """Build query params for the WebSocket URL from current settings.

        Returns:
            Dictionary of query parameters for the WebSocket URL.
            Only explicitly-set values are included. Boolean mistv2 params
            are serialized with ``json.dumps()`` for the wire format.
        """
        params: dict[str, Any] = {
            "speaker": self._settings.voice,
            "modelId": self._settings.model,
            "audioFormat": self._audio_format,
            "samplingRate": self._sampling_rate,
        }
        if self._settings.language is not None:
            params["lang"] = self._settings.language
        if self._settings.segment is not None:
            params["segment"] = self._settings.segment
        if self._settings.speedAlpha is not None:
            params["speedAlpha"] = self._settings.speedAlpha

        if self._settings.model == "coda":
            if self._settings.repetition_penalty is not None:
                params["repetition_penalty"] = self._settings.repetition_penalty
            if self._settings.temperature is not None:
                params["temperature"] = self._settings.temperature
            if self._settings.top_p is not None:
                params["top_p"] = self._settings.top_p
            if self._settings.timeScaleFactor is not None:
                params["timeScaleFactor"] = self._settings.timeScaleFactor
        else:  # mistv2/mist
            if self._settings.reduceLatency is not None:
                params["reduceLatency"] = self._settings.reduceLatency
            if self._settings.pauseBetweenBrackets is not None:
                params["pauseBetweenBrackets"] = json.dumps(self._settings.pauseBetweenBrackets)
            if self._settings.phonemizeBetweenBrackets is not None:
                params["phonemizeBetweenBrackets"] = json.dumps(
                    self._settings.phonemizeBetweenBrackets
                )
            if self._settings.noTextNormalization is not None:
                params["noTextNormalization"] = json.dumps(self._settings.noTextNormalization)
            if self._settings.saveOovs is not None:
                params["saveOovs"] = json.dumps(self._settings.saveOovs)

        return params

    # A set of Rime-specific helpers for text transformations
    @staticmethod
    def SPELL(text: str) -> str:
        """Wrap text in Rime spell function."""
        return f"spell({text})"

    @staticmethod
    def PAUSE_TAG(seconds: float) -> str:
        """Convenience method to create a pause tag."""
        return f"<{seconds * 1000}>"

    def PRONOUNCE(self, text: str, word: str, phoneme: str) -> str:
        """Convenience method to support Rime's custom pronunciations feature.

        https://docs.rime.ai/api-reference/custom-pronunciation
        """
        if self._use_websocket_v1:
            raise ValueError("PRONOUNCE is not supported by Rime WebSocket v1")
        self._extra_msg_fields["phonemizeBetweenBrackets"] = True
        return text.replace(word, f"{phoneme}")

    def INLINE_SPEED(self, text: str, speed: float) -> str:
        """Convenience method to support inline speeds."""
        if self._use_websocket_v1:
            raise ValueError("INLINE_SPEED is not supported by Rime WebSocket v1")
        if not self._extra_msg_fields:
            self._extra_msg_fields = {}
        speed_vals = self._extra_msg_fields.get("inlineSpeedAlpha", "").split(",")
        self._extra_msg_fields["inlineSpeedAlpha"] = ",".join(speed_vals + [str(speed)])
        return f"[{text}]"

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if necessary.

        WS3 settings require a reconnect because they are URL query parameters.
        V1 settings apply only to contexts that start after the update.
        """
        if self._use_websocket_v1:
            candidate = self._settings.copy()
            candidate_delta = self.Settings.from_mapping(delta.given_fields())
            candidate.apply_update(candidate_delta)
            if self._v1_endpoint_model and candidate.model != self._v1_endpoint_model:
                raise ValueError("settings.model does not match the v1 endpoint model")
            self._validate_v1_settings(candidate)
            return await super()._update_settings(delta)

        text_lookahead_tokens = delta.given_fields().get("text_lookahead_tokens", NOT_GIVEN)
        if is_given(text_lookahead_tokens) and text_lookahead_tokens is not None:
            raise ValueError("text_lookahead_tokens requires Rime WebSocket v1")
        changed = await super()._update_settings(delta)

        if changed and self._websocket:
            await self._disconnect()
            await self._connect()

        return changed

    def _build_msg(self, text: str = "", context_id: str = "") -> dict:
        """Build JSON message for Rime API."""
        msg = {"text": text, "contextId": context_id}
        if self._extra_msg_fields:
            msg |= self._extra_msg_fields
            self._extra_msg_fields = {}
        return msg

    def _build_clear_msg(self) -> dict:
        """Build clear operation message."""
        return {"operation": "clear"}

    def _build_eos_msg(self) -> dict:
        """Build end-of-stream operation message."""
        return {"operation": "eos"}

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._sampling_rate = self.sample_rate
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

    async def _connect_websocket(self):
        """Connect to Rime websocket API with configured settings."""
        if self._use_websocket_v1:
            await self._connect_websocket_v1()
            return

        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            ws_params = self._build_ws_params()
            params = "&".join(f"{k}={v}" for k, v in ws_params.items() if v is not None)
            url = f"{self._url}?{params}"
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._audio_remainder = b""
            self._audio_remainder_context_id = None
            self._websocket = await self._websocket_connect(url, additional_headers=headers)

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _connect_websocket_v1(self) -> None:
        """Connect to Rime v1 and wait until its engine is ready."""
        async with self._v1_connection_lock:
            await self._connect_websocket_v1_locked()

    async def _connect_websocket_v1_locked(self) -> None:
        """Connect to Rime v1 while holding the connection lifecycle lock."""
        websocket = None
        try:
            if self._websocket and self._websocket.state is State.OPEN and self._v1_client:
                return
            if self._websocket_url is None or self._websocket_protocol is None:
                raise ValueError("Rime WebSocket v1 is not configured")

            protocol = cast(WebSocketProtocol, self._websocket_protocol)
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

    async def _disconnect_websocket(self):
        """Close websocket connection and reset state."""
        if self._use_websocket_v1:
            await self._disconnect_websocket_v1()
            return

        try:
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.send(json.dumps(self._build_eos_msg()))
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _disconnect_websocket_v1(self) -> None:
        """Close Rime v1 and clear all connection-scoped state."""
        async with self._v1_connection_lock:
            await self._disconnect_websocket_v1_locked()

    async def _disconnect_websocket_v1_locked(self) -> None:
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
                    await self._discard_v1_context_text(context_id)
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                    await self.remove_audio_context(context_id)
            self._v1_options_by_context.clear()
            self._v1_audio_remainders.clear()
            self._v1_contexts_with_audio.clear()
            self._v1_closed_contexts.clear()
            await self._call_event_handler("on_disconnected")

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        """Reconnect the v1 socket without replacing a newer ready connection."""
        if not self._use_websocket_v1:
            return await super()._reconnect_websocket(attempt_number)

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

            await self._disconnect_websocket_v1_locked()
            await self._connect_websocket_v1_locked()
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
        discard_text: bool = False,
    ) -> None:
        """Finish protocol and Pipecat state for one context once."""
        first_close = context_id not in self._v1_closed_contexts
        self._v1_closed_contexts.add(context_id)
        await self.stop_ttfb_metrics()
        if error:
            await self.stop_all_metrics()
        await self._cancel_v1_context_tasks(context_id)
        self._v1_options_by_context.pop(context_id, None)
        self._v1_audio_remainders.pop(context_id, None)
        self._v1_contexts_with_audio.discard(context_id)

        if error or not emit_stop or discard_text:
            await self._discard_v1_context_text(context_id)

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

    async def _discard_v1_context_text(self, context_id: str) -> None:
        """Remove unspoken text while preserving skipped-frame order."""
        frames = self._aggregated_frame_sequencer.discard_context(
            context_id, last_word_pts=self._word_last_pts
        )
        for frame in frames:
            if self.audio_context_available(context_id):
                await self.append_to_audio_context(context_id, frame)
            else:
                await self.push_frame(frame)

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

    def _get_websocket(self):
        """Get active websocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _close_context(self, context_id: str):
        """Clear the Rime speech queue and stop metrics."""
        if self._use_websocket_v1:
            await self.stop_all_metrics()
            return
        await self.stop_all_metrics()
        if context_id:
            await self._get_websocket().send(json.dumps(self._build_clear_msg()))

    async def on_audio_context_interrupted(self, context_id: str):
        """Clear the Rime speech queue and stop metrics when the bot is interrupted."""
        if self._use_websocket_v1:
            await self.stop_all_metrics()
            self._v1_failed_contexts.discard(context_id)
            await self._cancel_v1_context_tasks(
                context_id, include_terminal=True, include_cancel=False
            )
            client = self._v1_client
            if client and client.has_context(context_id):
                self._v1_closed_contexts.add(context_id)
                try:
                    await client.cancel(context_id)
                    self._start_v1_cancel_watchdog(context_id)
                except Exception:
                    await self._invalidate_v1_connection(
                        "Failed to cancel a Rime v1 context", ErrorCategory.CONNECTIVITY, client
                    )
            else:
                self._v1_options_by_context.pop(context_id, None)
                self._v1_audio_remainders.pop(context_id, None)
                self._v1_contexts_with_audio.discard(context_id)
                self._v1_closed_contexts.discard(context_id)
                if client:
                    client.discard_context(context_id)
            await super().on_audio_context_interrupted(context_id)
            return
        await self._close_context(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str):
        """Clear legacy server state after the Rime context finishes playing."""
        if not self._use_websocket_v1:
            await self._close_context(context_id)
        await super().on_audio_context_completed(context_id)

    async def on_turn_context_completed(self) -> None:
        """Close a failed v1 audio context after its Pipecat turn ends."""
        context_id = self._turn_context_id
        await super().on_turn_context_completed()
        if (
            self._use_websocket_v1
            and context_id is not None
            and context_id in self._v1_failed_contexts
        ):
            await self._discard_v1_context_text(context_id)
            if self.audio_context_available(context_id):
                await self.append_to_audio_context(
                    context_id, TTSStoppedFrame(context_id=context_id)
                )
                await self.remove_audio_context(context_id)
            self._v1_failed_contexts.discard(context_id)

    def _calculate_word_times(self, words: list, starts: list, ends: list) -> list:
        """Calculate word timing pairs with proper spacing and punctuation.

        Args:
            words: List of words from Rime.
            starts: List of start times for each word.
            ends: List of end times for each word.

        Returns:
            List of (word, timestamp) pairs with proper timing.
        """
        word_pairs = []
        for i, (word, start_time, _) in enumerate(zip(words, starts, ends)):
            if not word.strip():
                continue

            # Adjust timing by adding cumulative time
            adjusted_start = start_time + self._cumulative_time

            # Handle punctuation by appending to previous word
            is_punctuation = bool(word.strip(",.!?") == "")
            if is_punctuation and word_pairs:
                prev_word, prev_time = word_pairs[-1]
                word_pairs[-1] = (prev_word + word, prev_time)
            else:
                word_pairs.append((word, adjusted_start))

        return word_pairs

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio synthesis."""
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket:
            return

        if self._use_websocket_v1:
            client = self._v1_client
            if not client or not client.has_context(flush_id):
                return
            try:
                await client.end(flush_id)
                self._start_v1_terminal_watchdog(flush_id)
            except Exception:
                await self._invalidate_v1_connection(
                    "Failed to end a Rime v1 context", ErrorCategory.CONNECTIVITY, client
                )
            return

        logger.trace(f"{self}: flushing audio")
        await self._get_websocket().send(json.dumps({"operation": "flush"}))

    def _sample_aligned_audio(self, context_id: str, audio: bytes) -> bytes:
        """Return whole 16-bit samples, holding back any dangling byte.

        Rime chops its PCM stream at arbitrary byte boundaries, so a chunk may
        end mid-sample. The dangling byte is held back and prepended to the
        context's next chunk so emitted frames always contain whole samples.
        """
        if self._audio_remainder_context_id != context_id:
            self._audio_remainder = b""
            self._audio_remainder_context_id = context_id
        audio = self._audio_remainder + audio
        aligned = len(audio) - (len(audio) % 2)
        self._audio_remainder = audio[aligned:]
        return audio[:aligned]

    def _sample_aligned_v1_audio(self, context_id: str, audio: bytes) -> bytes:
        """Return whole PCM16 samples without mixing context remainders."""
        combined = self._v1_audio_remainders.pop(context_id, b"") + audio
        aligned = len(combined) - (len(combined) % 2)
        if aligned != len(combined):
            self._v1_audio_remainders[context_id] = combined[-1:]
        return combined[:aligned]

    async def _receive_messages(self):
        """Process incoming websocket messages."""
        if self._use_websocket_v1:
            await self._receive_v1_messages()
            return

        async for message in self._get_websocket():
            msg = json.loads(message)

            if not msg or not self.audio_context_available(msg.get("contextId")):
                continue

            context_id = msg["contextId"]
            if msg["type"] == "chunk":
                # Process audio chunk
                audio = self._sample_aligned_audio(context_id, base64.b64decode(msg["data"]))
                if not audio:
                    continue
                frame = TTSAudioRawFrame(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
                await self.append_to_audio_context(context_id, frame)

            elif msg["type"] == "timestamps":
                # Process word timing information
                timestamps = msg.get("word_timestamps", {})
                words = timestamps.get("words", [])
                starts = timestamps.get("start", [])
                ends = timestamps.get("end", [])

                if words and starts:
                    # Calculate word timing pairs
                    word_pairs = self._calculate_word_times(words, starts, ends)
                    if word_pairs:
                        await self.add_word_timestamps(word_pairs, context_id=context_id)
                        self._cumulative_time = ends[-1] + self._cumulative_time
                        logger.debug(f"Updated cumulative time to: {self._cumulative_time}")

            elif msg["type"] == "done":
                await self.stop_ttfb_metrics()
                await self.append_to_audio_context(
                    context_id, TTSStoppedFrame(context_id=context_id)
                )
                await self.remove_audio_context(context_id)

            elif msg["type"] == "error":
                await self.push_frame(TTSStoppedFrame())
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Error: {msg['message']}")
                self.reset_active_audio_context()

    async def _receive_v1_messages(self) -> None:
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
                    audio = self._sample_aligned_v1_audio(event.context_id, event.audio)
                    if audio and self.audio_context_available(event.context_id):
                        self._v1_contexts_with_audio.add(event.context_id)
                        await self.append_to_audio_context(
                            event.context_id,
                            TTSAudioRawFrame(
                                audio=audio,
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
                    if self._v1_audio_remainders.get(event.context_id):
                        await self._finish_v1_context(
                            event.context_id,
                            error=ErrorFrame(
                                error="Rime v1 ended with an incomplete PCM16 sample",
                                category=ErrorCategory.SERVER,
                            ),
                        )
                    else:
                        await self._finish_v1_context(
                            event.context_id,
                            discard_text=event.context_id not in self._v1_contexts_with_audio,
                        )
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
        """Generate speech from text using Rime's streaming API.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        if self._use_websocket_v1:
            async for frame in self._run_tts_v1(text, context_id):
                yield frame
            return

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self.audio_context_available(context_id):
                    await self.create_audio_context(context_id)
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame(context_id=context_id)
                    self._cumulative_time = 0

                msg = self._build_msg(text=text, context_id=context_id)
                await self._get_websocket().send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")

    async def _run_tts_v1(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send sentence units to a Rime v1 context."""
        client = None
        try:
            if context_id in self._v1_failed_contexts:
                await self._discard_v1_context_text(context_id)
                return
            if not self._websocket or self._websocket.state is State.CLOSED or not self._v1_client:
                await self._connect()
            client = self._get_v1_client()
            first_text = context_id not in self._v1_options_by_context
            if first_text:
                self._v1_closed_contexts.discard(context_id)
                self._v1_contexts_with_audio.discard(context_id)
                self._v1_options_by_context[context_id] = self._build_v1_options()
            options = self._v1_options_by_context[context_id]
            for sentence in await self._split_v1_sentences(text):
                await client.send_text(context_id, options, sentence)
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
            self._v1_audio_remainders.pop(context_id, None)
            self._v1_contexts_with_audio.discard(context_id)
            await self._discard_v1_context_text(context_id)
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


class RimeHttpTTSService(TTSService):
    """Rime HTTP-based text-to-speech service.

    Provides text-to-speech synthesis using Rime's HTTP API for batch processing.
    Suitable for use cases where streaming is not required.
    """

    Settings = RimeTTSSettings
    _settings: Settings

    @deprecated(
        "`RimeHttpTTSService.InputParams` is deprecated since 0.0.105 and will be removed in "
        "2.0.0. Use `RimeHttpTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Configuration parameters for Rime HTTP TTS service.

        .. deprecated:: 0.0.105
            Use ``settings=RimeHttpTTSService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Language for synthesis. Defaults to English.
            pause_between_brackets: Whether to add pauses between bracketed content.
            phonemize_between_brackets: Whether to phonemize bracketed content.
            inline_speed_alpha: Inline speed control markup.
            speed_alpha: Speech speed multiplier. Defaults to 1.0.
            reduce_latency: Whether to reduce latency at potential quality cost.
        """

        language: Language | None = Language.EN
        pause_between_brackets: bool | None = False
        phonemize_between_brackets: bool | None = False
        inline_speed_alpha: str | None = None
        speed_alpha: float | None = 1.0
        reduce_latency: bool | None = False

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str | None = None,
        aiohttp_session: aiohttp.ClientSession,
        model: str | None = None,
        sample_rate: int | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize Rime HTTP TTS service.

        Args:
            api_key: Rime API key for authentication.
            voice_id: ID of the voice to use.

                .. deprecated:: 0.0.105
                    Use ``settings=RimeHttpTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            aiohttp_session: Shared aiohttp session for HTTP requests.
            model: Model ID to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=RimeHttpTTSService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            sample_rate: Audio sample rate in Hz.
            params: Additional configuration parameters.

                .. deprecated:: 0.0.105
                    Use ``settings=RimeHttpTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="coda",
            voice=None,
            language="eng",
            segment=None,
            speedAlpha=None,
            reduceLatency=None,
            pauseBetweenBrackets=None,
            phonemizeBetweenBrackets=None,
            noTextNormalization=None,
            saveOovs=None,
            inlineSpeedAlpha=None,
            repetition_penalty=None,
            temperature=None,
            top_p=None,
            timeScaleFactor=None,
            text_lookahead_tokens=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.language = params.language
                default_settings.speedAlpha = params.speed_alpha
                default_settings.reduceLatency = params.reduce_latency
                default_settings.pauseBetweenBrackets = params.pause_between_brackets
                default_settings.phonemizeBetweenBrackets = params.phonemize_between_brackets
                default_settings.inlineSpeedAlpha = (
                    params.inline_speed_alpha if params.inline_speed_alpha else None
                )

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)
        if default_settings.text_lookahead_tokens is not None:
            raise ValueError("text_lookahead_tokens requires Rime WebSocket v1")

        super().__init__(
            sample_rate=sample_rate,
            push_stop_frames=True,
            push_start_frame=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._session = aiohttp_session
        self._base_url = "https://users.rime.ai/v1/rime-tts"

        # Init-only audio format fields (not runtime-updatable)
        self._audio_format = "pcm"

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Rime HTTP service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply settings supported by the HTTP interface."""
        text_lookahead_tokens = delta.given_fields().get("text_lookahead_tokens", NOT_GIVEN)
        if is_given(text_lookahead_tokens) and text_lookahead_tokens is not None:
            raise ValueError("text_lookahead_tokens requires Rime WebSocket v1")
        return await super()._update_settings(delta)

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert pipecat language to Rime language code.

        Args:
            language: The language to convert.

        Returns:
            The Rime-specific language code, or None if not supported.
        """
        return language_to_rime_language(language)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Rime's HTTP API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        headers = {
            "Accept": "audio/pcm",
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "text": text,
            "speaker": self._settings.voice,
            "modelId": self._settings.model,
            "samplingRate": self.sample_rate,
        }
        if self._settings.language is not None:
            payload["lang"] = self._settings.language
        if self._settings.speedAlpha is not None:
            payload["speedAlpha"] = self._settings.speedAlpha
        if self._settings.inlineSpeedAlpha is not None:
            payload["inlineSpeedAlpha"] = self._settings.inlineSpeedAlpha

        if self._settings.model == "coda":
            if self._settings.repetition_penalty is not None:
                payload["repetition_penalty"] = self._settings.repetition_penalty
            if self._settings.temperature is not None:
                payload["temperature"] = self._settings.temperature
            if self._settings.top_p is not None:
                payload["top_p"] = self._settings.top_p
            if self._settings.timeScaleFactor is not None:
                payload["timeScaleFactor"] = self._settings.timeScaleFactor
        else:  # mistv2/mist
            if self._settings.reduceLatency is not None:
                payload["reduceLatency"] = self._settings.reduceLatency
            if self._settings.pauseBetweenBrackets is not None:
                payload["pauseBetweenBrackets"] = self._settings.pauseBetweenBrackets
            if self._settings.phonemizeBetweenBrackets is not None:
                payload["phonemizeBetweenBrackets"] = self._settings.phonemizeBetweenBrackets

        try:
            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_message = f"Rime TTS error: HTTP {response.status}"
                    yield ErrorFrame(error=error_message)
                    return

                await self.start_tts_usage_metrics(text)

                CHUNK_SIZE = self.chunk_size

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(CHUNK_SIZE),
                    strip_wav_header=False,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()


@deprecated(
    "`RimeNonJsonTTSService` is deprecated since 0.0.102 and will be removed in 2.0.0. "
    "Use `RimeTTSService` instead."
)
class RimeNonJsonTTSService(InterruptibleTTSService):
    """Pipecat TTS service for Rime's non-JSON WebSocket API.

    .. deprecated:: 0.0.102
        Use :class:`RimeTTSService` instead. Will be removed in 2.0.0.

    This service enables Text-to-Speech synthesis over WebSocket endpoints
    that require plain text (not JSON) messages and return raw audio bytes.

    Limitations:
        - Does not support word-level timestamps or context IDs.
        - Intended specifically for integrations where the TTS provider only
          accepts and returns non-JSON messages.
    """

    Settings = RimeNonJsonTTSSettings
    _settings: Settings

    @deprecated(
        "`RimeNonJsonTTSService.InputParams` is deprecated since 0.0.105 and will be removed in "
        "2.0.0. Use `RimeNonJsonTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Configuration parameters for Rime Non-JSON WebSocket TTS service.

        .. deprecated:: 0.0.105
            Use ``settings=RimeNonJsonTTSService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Args:
            language: Language for synthesis. Defaults to English.
            segment: Text segmentation mode ("immediate", "bySentence", "never").
            repetition_penalty: Token repetition penalty (1.0-2.0).
            temperature: Sampling temperature (0.0-1.0).
            top_p: Cumulative probability threshold (0.0-1.0).
            extra: Additional parameters to pass to the API (for future compatibility).
        """

        language: Language | None = None
        segment: str | None = None
        repetition_penalty: float | None = None
        temperature: float | None = None
        top_p: float | None = None
        extra: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str | None = None,
        url: str = "wss://users.rime.ai/ws",
        model: str | None = None,
        audio_format: str = "pcm",
        sample_rate: int | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        aggregate_sentences: bool | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        **kwargs,
    ):
        """Initialize Rime Non-JSON WebSocket TTS service.

        Args:
            api_key: Rime API key for authentication.
            voice_id: ID of the voice to use.

                .. deprecated:: 0.0.105
                    Use ``settings=RimeNonJsonTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            url: Rime websocket API endpoint.
            model: Model ID to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=RimeNonJsonTTSService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            audio_format: Audio format to use.
            sample_rate: Audio sample rate in Hz.
            params: Additional configuration parameters.

                .. deprecated:: 0.0.105
                    Use ``settings=RimeNonJsonTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            aggregate_sentences: Deprecated. Use text_aggregation_mode instead.

                .. deprecated:: 0.0.104
                    Use ``text_aggregation_mode`` instead. Set to ``TextAggregationMode.SENTENCE``
                    to aggregate text into sentences before synthesis, or
                    ``TextAggregationMode.TOKEN`` to stream tokens directly for lower latency.
                    Will be removed in 2.0.0.

            text_aggregation_mode: How to aggregate text before synthesis.
            **kwargs: Additional arguments passed to parent class.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            voice=None,
            model="coda",
            language=None,
            segment=None,
            repetition_penalty=None,
            temperature=None,
            top_p=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.language = params.language
                default_settings.segment = params.segment
                default_settings.repetition_penalty = params.repetition_penalty
                default_settings.temperature = params.temperature
                default_settings.top_p = params.top_p

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            aggregate_sentences=aggregate_sentences,
            text_aggregation_mode=text_aggregation_mode,
            push_stop_frames=True,
            push_start_frame=True,
            pause_frame_processing=True,
            append_trailing_space=True,
            settings=default_settings,
            **kwargs,
        )

        # Init-only audio format fields (not runtime-updatable)
        self._audio_format = audio_format
        self._sampling_rate = sample_rate

        self._api_key = api_key
        self._url = url
        # Add any extra parameters for future compatibility
        if params and params.extra:
            self._settings.extra.update(params.extra)

        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Rime Non-JSON WebSocket service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert pipecat Language enum to Rime language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            Three-letter Rime language code (e.g., 'eng' for English).
            Falls back to the language's base code with a warning if not in the verified list.
        """
        return language_to_rime_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._sampling_rate = self.sample_rate
        await self._connect()

    async def _connect(self):
        """Establish WebSocket connection and start receive task."""
        await super()._connect()

        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close WebSocket connection and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to Rime non-JSON websocket."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            # Build URL with query parameters (only given, non-None values)
            settings_dict = {
                "speaker": self._settings.voice,
                "modelId": self._settings.model,
                "audioFormat": self._audio_format,
                "samplingRate": self._sampling_rate,
            }
            if self._settings.language is not None:
                settings_dict["lang"] = self._settings.language
            if self._settings.segment is not None:
                settings_dict["segment"] = self._settings.segment
            if self._settings.repetition_penalty is not None:
                settings_dict["repetition_penalty"] = self._settings.repetition_penalty
            if self._settings.temperature is not None:
                settings_dict["temperature"] = self._settings.temperature
            if self._settings.top_p is not None:
                settings_dict["top_p"] = self._settings.top_p
            # Include extras
            settings_dict.update(self._settings.extra)
            params = "&".join(f"{k}={v}" for k, v in settings_dict.items() if v is not None)
            url = f"{self._url}?{params}"
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await self._websocket_connect(
                url, additional_headers=headers, max_size=1024 * 1024 * 16
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()
            if self._websocket:
                # Send EOS command to gracefully close
                await self._websocket.send("<EOS>")
                await self._websocket.close()
                logger.debug("Disconnected from Rime non-JSON websocket")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get active WebSocket connection or raise exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio synthesis."""
        if not self._websocket:
            return

        logger.trace(f"{self}: flushing audio")
        await self._websocket.send("<FLUSH>")

    async def _receive_messages(self):
        """Process incoming WebSocket messages (raw audio bytes)."""
        async for message in self._get_websocket():
            try:
                # Rime sends raw audio bytes directly.
                if isinstance(message, bytes):
                    await self.stop_ttfb_metrics()

                    context_id = self.get_active_audio_context_id()
                    frame = TTSAudioRawFrame(
                        audio=message,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
                    await self.append_to_audio_context(context_id, frame)
            except Exception as e:
                await self.push_error(error_msg=f"Error: {e}", exception=e)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Rime's streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()
            try:
                # Send bare text (not JSON)
                await self._get_websocket().send(text)
                await self.start_tts_usage_metrics(text)

            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if necessary.

        Since all settings are WebSocket URL query parameters,
        any setting change requires reconnecting to apply the new values.
        """
        changed = await super()._update_settings(delta)

        if changed:
            logger.debug("Settings changed, reconnecting WebSocket with new parameters")
            await self._disconnect()
            await self._connect()

        return changed
