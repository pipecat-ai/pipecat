#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ElevenLabs text-to-speech service implementations.

This module provides WebSocket and HTTP-based TTS services using ElevenLabs API
with support for streaming audio, word timestamps, and voice customization.
"""

import base64
import json
import warnings
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Literal,
    Union,
)

import aiohttp
from loguru import logger
from pydantic import BaseModel
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.elevenlabs.tts_base import (
    ELEVENLABS_MODEL_LANGUAGES,
    ElevenLabsTTSBase,
    ElevenLabsTTSSettingsBase,
    _select_alignment,
    _strip_utterance_leading_spaces,
    _word_timestamps_include_inter_frame_spaces,
    calculate_word_times,
    elevenlabs_language_code,
    language_to_elevenlabs_language,
    output_format_from_sample_rate,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import (
    TextAggregationMode,
    TTSService,
)
from pipecat.transcriptions.language import Language
from pipecat.utils.deprecation import deprecated
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

# Re-exported so the documented import path for these helpers keeps working.
__all__ = [
    "ELEVENLABS_CONTEXT_UNSUPPORTED_MODELS",
    "ELEVENLABS_MODEL_LANGUAGES",
    "ELEVENLABS_MULTILINGUAL_MODELS",
    "ElevenLabsHttpTTSService",
    "ElevenLabsHttpTTSSettings",
    "ElevenLabsTTSService",
    "ElevenLabsTTSSettings",
    "PronunciationDictionaryLocator",
    "build_elevenlabs_voice_settings",
    "calculate_word_times",
    "language_to_elevenlabs_language",
    "output_format_from_sample_rate",
]

# Models that support language codes. Which languages each one accepts differs,
# so the language itself is validated by `elevenlabs_language_code`.
ELEVENLABS_MULTILINGUAL_MODELS = set(ELEVENLABS_MODEL_LANGUAGES)

# Models that reject the previous_text/next_text context parameters
ELEVENLABS_CONTEXT_UNSUPPORTED_MODELS = {
    "eleven_v3",
}


def build_elevenlabs_voice_settings(
    settings: Union[dict[str, Any], "TTSSettings"],
) -> dict[str, float | bool] | None:
    """Build voice settings dictionary for ElevenLabs based on provided settings.

    Args:
        settings: Dictionary or settings containing voice settings parameters.

    Returns:
        Dictionary of voice settings or None if no valid settings are provided.
    """
    voice_setting_keys = ["stability", "similarity_boost", "style", "use_speaker_boost", "speed"]

    voice_settings = {}
    for key in voice_setting_keys:
        val = (
            getattr(settings, key, None) if isinstance(settings, TTSSettings) else settings.get(key)
        )
        if val is not None:
            voice_settings[key] = val

    return voice_settings or None


@deprecated(
    "`PronunciationDictionaryLocator` is deprecated since 1.6.0 and will be removed in 2.0.0. "
    "Use `text_transforms` -> `replace_text` instead."
)
class PronunciationDictionaryLocator(BaseModel):
    """Locator for a pronunciation dictionary.

    .. deprecated:: 1.6.0
        Use the ``text_transforms`` parameter with
        :func:`pipecat.utils.text.transforms.replace_text` instead. Pronunciation
        dictionary substitutions can rewrite the spoken words in ways that no
        longer match the text sent to synthesis, which breaks the
        alignment-based word-completion tracking used to attribute spoken text
        back to the conversation context. ``replace_text`` transforms happen
        client-side, so they're tracked correctly. Will be removed in 2.0.0.

    Parameters:
        pronunciation_dictionary_id: The ID of the pronunciation dictionary.
        version_id: The version ID of the pronunciation dictionary.
    """

    pronunciation_dictionary_id: str
    version_id: str


@dataclass
class ElevenLabsTTSSettings(ElevenLabsTTSSettingsBase):
    """Settings for ElevenLabsTTSService.

    Fields that appear in the WebSocket URL (``voice``, ``model``,
    ``language``) require a full reconnect when changed.  Fields that
    affect the voice character (``stability``, ``similarity_boost``,
    ``style``, ``use_speaker_boost``, ``speed``) can be applied by closing
    the current audio context so a new one is opened with updated settings.

    Parameters:
        stability: Voice stability control (0.0 to 1.0).
        similarity_boost: Similarity boost control (0.0 to 1.0).
        style: Style control for voice expression (0.0 to 1.0).
        use_speaker_boost: Whether to use speaker boost enhancement.
        speed: Voice speed control (0.7 to 1.2).
        apply_text_normalization: Text normalization mode ("auto", "on", "off").
    """

    stability: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    similarity_boost: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    style: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    use_speaker_boost: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    apply_text_normalization: Literal["auto", "on", "off"] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )

    #: Fields in the WS URL — changing any of these requires a reconnect.
    URL_FIELDS: ClassVar[frozenset[str]] = frozenset({"voice", "model", "language"})

    #: Fields affecting voice character — changing these requires closing the
    #: current audio context so the next one picks up new settings.
    VOICE_SETTINGS_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"stability", "similarity_boost", "style", "use_speaker_boost", "speed"}
    )


@dataclass
class ElevenLabsHttpTTSSettings(TTSSettings):
    """Settings for ElevenLabsHttpTTSService.

    Parameters:
        optimize_streaming_latency: Latency optimization level (0-4).
        stability: Voice stability control (0.0 to 1.0).
        similarity_boost: Similarity boost control (0.0 to 1.0).
        style: Style control for voice expression (0.0 to 1.0).
        use_speaker_boost: Whether to use speaker boost enhancement.
        speed: Voice speed control (0.25 to 4.0).
        apply_text_normalization: Text normalization mode ("auto", "on", "off").
    """

    optimize_streaming_latency: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    stability: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    similarity_boost: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    style: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    use_speaker_boost: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    apply_text_normalization: Literal["auto", "on", "off"] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


class ElevenLabsTTSService(ElevenLabsTTSBase):
    """ElevenLabs WebSocket-based TTS service with word timestamps.

    Provides real-time text-to-speech using ElevenLabs' WebSocket streaming API.
    Supports word-level timestamps, audio context management, and various voice
    customization options including stability, similarity boost, and speed controls.
    """

    Settings = ElevenLabsTTSSettings
    _settings: Settings

    @deprecated(
        "`ElevenLabsTTSService.InputParams` is deprecated since 0.0.105 and will be removed in "
        "2.0.0. Use `ElevenLabsTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Input parameters for ElevenLabs TTS configuration.

        .. deprecated:: 0.0.105
            Use ``settings=ElevenLabsTTSService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Language to use for synthesis.
            stability: Voice stability control (0.0 to 1.0).
            similarity_boost: Similarity boost control (0.0 to 1.0).
            style: Style control for voice expression (0.0 to 1.0).
            use_speaker_boost: Whether to use speaker boost enhancement.
            speed: Voice speed control (0.7 to 1.2).
            auto_mode: Whether to enable automatic mode optimization.
            enable_ssml_parsing: Whether to parse SSML tags in text.
            enable_logging: Whether to enable ElevenLabs logging.
            apply_text_normalization: Text normalization mode ("auto", "on", "off").
            pronunciation_dictionary_locators: List of pronunciation dictionary locators to use.
        """

        language: Language | None = None
        stability: float | None = None
        similarity_boost: float | None = None
        style: float | None = None
        use_speaker_boost: bool | None = None
        speed: float | None = None
        auto_mode: bool | None = True
        enable_ssml_parsing: bool | None = None
        enable_logging: bool | None = None
        apply_text_normalization: Literal["auto", "on", "off"] | None = None
        pronunciation_dictionary_locators: list[PronunciationDictionaryLocator] | None = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str | None = None,
        model: str | None = None,
        url: str = "wss://api.elevenlabs.io",
        sample_rate: int | None = None,
        auto_mode: bool | None = None,
        enable_ssml_parsing: bool | None = None,
        enable_logging: bool | None = None,
        pronunciation_dictionary_locators: list[PronunciationDictionaryLocator] | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        aggregate_sentences: bool | None = None,
        **kwargs,
    ):
        """Initialize the ElevenLabs TTS service.

        Args:
            api_key: ElevenLabs API key for authentication.
            voice_id: ID of the voice to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=ElevenLabsTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            model: TTS model to use (e.g., "eleven_flash_v2_5").

                .. deprecated:: 0.0.105
                    Use ``settings=ElevenLabsTTSService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            url: WebSocket URL for ElevenLabs TTS API.
            sample_rate: Audio sample rate. If None, uses default.
            auto_mode: Whether to enable ElevenLabs' auto mode, which reduces
                latency by disabling server-side chunk scheduling and buffering.
                Recommended when sending complete sentences or phrases. When
                None (default), auto mode is enabled for ``SENTENCE``
                aggregation and disabled for ``TOKEN`` aggregation — because
                token streaming relies on the server-side chunk scheduler to
                accumulate enough text for natural-sounding synthesis.
            enable_ssml_parsing: Whether to parse SSML tags in text.
            enable_logging: Whether to enable ElevenLabs server-side logging.
            pronunciation_dictionary_locators: List of pronunciation dictionary
                locators to use.

                .. deprecated:: 1.6.0
                    Use the ``text_transforms`` parameter with
                    :func:`pipecat.utils.text.transforms.replace_text`
                    instead. Pronunciation dictionary substitutions can
                    rewrite the spoken words in ways that no longer match
                    the text sent to synthesis, which breaks the
                    alignment-based word-completion tracking used to
                    attribute spoken text back to the conversation context.
                    ``replace_text`` transforms happen client-side, so
                    they're tracked correctly. Will be removed in 2.0.0.

            params: Additional input parameters for voice customization.

                .. deprecated:: 0.0.105
                    Use ``settings=ElevenLabsTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            text_aggregation_mode: How to aggregate incoming text before synthesis.
            aggregate_sentences: Whether to aggregate sentences within the TTSService.

                .. deprecated:: 0.0.104
                    Use ``text_aggregation_mode`` instead.
                    Will be removed in 2.0.0.

            **kwargs: Additional arguments passed to the parent service.
        """
        # By default, we aggregate sentences before sending to TTS. This adds
        # ~200-300ms of latency per sentence (waiting for the sentence-ending
        # punctuation token from the LLM). Setting
        # text_aggregation_mode=TextAggregationMode.TOKEN streams tokens
        # directly. To use this mode, you must set auto_mode=False. This
        # eliminates aggregation time, but slows down ElevenLabs.
        #
        # We also don't want to automatically push LLM response text frames,
        # because the context aggregators will add them to the LLM context even
        # if we're interrupted. ElevenLabs gives us word-by-word timestamps. We
        # can use those to generate text frames ourselves aligned with the
        # playout timing of the audio!
        #
        # Finally, ElevenLabs doesn't provide information on when the bot stops
        # speaking for a while, so we want the parent class to send TTSStopFrame
        # after a short period not receiving any audio.

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="eleven_flash_v2_5",
            voice=None,
            language=None,
            stability=None,
            similarity_boost=None,
            style=None,
            use_speaker_boost=None,
            speed=None,
            apply_text_normalization=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. Apply params overrides — only if settings not provided
        _pronunciation_dictionary_locators = pronunciation_dictionary_locators
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.language is not None:
                    default_settings.language = params.language
                if params.stability is not None:
                    default_settings.stability = params.stability
                if params.similarity_boost is not None:
                    default_settings.similarity_boost = params.similarity_boost
                if params.style is not None:
                    default_settings.style = params.style
                if params.use_speaker_boost is not None:
                    default_settings.use_speaker_boost = params.use_speaker_boost
                if params.speed is not None:
                    default_settings.speed = params.speed
                if params.auto_mode is not None:
                    auto_mode = params.auto_mode
                if params.enable_ssml_parsing is not None:
                    enable_ssml_parsing = params.enable_ssml_parsing
                if params.enable_logging is not None:
                    enable_logging = params.enable_logging
                if params.apply_text_normalization is not None:
                    default_settings.apply_text_normalization = params.apply_text_normalization
                if _pronunciation_dictionary_locators is None:
                    _pronunciation_dictionary_locators = params.pronunciation_dictionary_locators

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            url=url,
            enable_logging=enable_logging,
            text_aggregation_mode=text_aggregation_mode,
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,
            push_stop_frames=False,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        # Init-only WebSocket URL params (not runtime-updatable).
        #
        # ElevenLabs' auto mode reduces latency by disabling server-side chunk
        # scheduling and buffering — it's designed for inputs that are already
        # complete sentences or phrases. In TOKEN mode we stream individual LLM
        # tokens, so we need the server-side scheduler to accumulate enough
        # text for natural-sounding synthesis; enabling auto mode there would
        # hurt quality. When the caller hasn't set auto_mode explicitly, we
        # derive the right default from the text aggregation strategy.
        if auto_mode is None:
            auto_mode = self._text_aggregation_mode != TextAggregationMode.TOKEN

        self._auto_mode = auto_mode
        self._enable_ssml_parsing = enable_ssml_parsing

        if _pronunciation_dictionary_locators is not None:
            warnings.warn(
                "`pronunciation_dictionary_locators` is deprecated since 1.6.0 and will be "
                "removed in 2.0.0. Use `text_transforms` -> `replace_text` instead. "
                "Pronunciation dictionary substitutions can rewrite the spoken words in "
                "ways that no longer match the text sent to synthesis, which breaks the "
                "alignment-based word-completion tracking used to attribute spoken text "
                "back to the conversation context.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._pronunciation_dictionary_locators = _pronunciation_dictionary_locators

        self._alignment_started_context_ids: set[str | None] = set()

        # Context IDs whose context-init has been sent, so the keepalive knows
        # which contexts are safe to target.
        self._context_init_sent: set[str] = set()

    def _set_voice_settings(self):
        return build_elevenlabs_voice_settings(self._settings)

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio and finalize the current context.

        Args:
            context_id: The specific context to flush. If None, falls back to the
                currently active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket:
            return
        logger.trace(f"{self}: flushing audio")
        msg = {"context_id": flush_id, "flush": True}
        await self._websocket.send(json.dumps(msg))

    def _build_websocket_url(self) -> str:
        voice_id = self._settings.voice
        model = self._settings.model
        output_format = self._output_format
        url = f"{self._url}/v1/text-to-speech/{voice_id}/multi-stream-input?model_id={model}&output_format={output_format}&auto_mode={str(self._auto_mode).lower()}"

        if self._enable_ssml_parsing is not None:
            url += f"&enable_ssml_parsing={str(self._enable_ssml_parsing).lower()}"

        if self._enable_logging is not None:
            url += f"&enable_logging={str(self._enable_logging).lower()}"

        if self._settings.apply_text_normalization is not None:
            url += f"&apply_text_normalization={self._settings.apply_text_normalization}"

        language_code = elevenlabs_language_code(
            assert_given(model), assert_given(self._settings.language)
        )
        if language_code:
            url += f"&language_code={language_code}"

        return url

    def _clear_connection_state(self):
        self._context_init_sent.clear()

    async def _close_context(self, context_id: str):
        # ElevenLabs requires that Pipecat explicitly closes contexts to free
        # server-side resources, both on interruption and on normal completion.
        if context_id and self._websocket:
            logger.trace(f"{self}: Closing context {context_id}")
            try:
                # ElevenLabs requires that Pipecat manages the contexts and closes them
                # when they're not longer in use. Since an InterruptionFrame is pushed
                # every time the user speaks, we'll use this as a trigger to close the context
                # and reset the state.
                # Note: We do not need to call remove_audio_context here, as the context is
                # automatically reset when super ()._handle_interruption is called.
                await self._websocket.send(
                    json.dumps({"context_id": context_id, "close_context": True})
                )
            except Exception as e:
                await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

    def _reset_alignment_state(self, context_id: str):
        super()._reset_alignment_state(context_id)
        self._alignment_started_context_ids.discard(context_id)
        self._context_init_sent.discard(context_id)

    async def on_turn_context_completed(self):
        """Close the server-side context at end of turn.

        Sends close_context so isFinal arrives immediately after the last audio byte.
        """
        context_id = self._turn_context_id
        await super().on_turn_context_completed()
        if context_id:
            await self._close_context(context_id)

    async def _receive_messages(self):
        """Handle incoming WebSocket messages from ElevenLabs."""
        async for message in self._get_websocket():
            msg = json.loads(message)

            received_ctx_id = msg.get("contextId")

            # Handle final messages first, regardless of context availability
            if msg.get("isFinal") is True:
                logger.debug(f"Received final message for context {received_ctx_id}")
                # In case of interruption, there is no audio context available, so we don’t need to do anything.
                if self.audio_context_available(received_ctx_id):
                    await self.append_to_audio_context(
                        received_ctx_id, TTSStoppedFrame(context_id=received_ctx_id)
                    )
                    await self.remove_audio_context(received_ctx_id)
                continue

            if msg.get("audio"):
                audio = base64.b64decode(msg["audio"])
                frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=received_ctx_id)
                await self.append_to_audio_context(received_ctx_id, frame)

            raw_alignment = _select_alignment(
                msg,
                normalized_key="normalizedAlignment",
                alignment_key="alignment",
                prefer_normalized=bool(self._pronunciation_dictionary_locators),
            )
            if raw_alignment:
                alignment = _strip_utterance_leading_spaces(
                    raw_alignment,
                    ("chars", "charStartTimesMs", "charDurationsMs"),
                    received_ctx_id not in self._alignment_started_context_ids,
                )
                self._alignment_started_context_ids.add(received_ctx_id)
                word_times, self._partial_word, self._partial_word_start_time = (
                    calculate_word_times(
                        alignment,
                        self._cumulative_time,
                        self._partial_word,
                        self._partial_word_start_time,
                    )
                )

                if word_times:
                    await self.add_word_timestamps(
                        word_times,
                        received_ctx_id,
                        includes_inter_frame_spaces=(
                            True
                            if _word_timestamps_include_inter_frame_spaces(
                                assert_given(self._settings.language)
                            )
                            else None
                        ),
                    )

                    # Calculate the actual end time of this audio chunk
                    char_start_times_ms = alignment.get("charStartTimesMs", [])
                    char_durations_ms = alignment.get("charDurationsMs", [])

                    if char_start_times_ms and char_durations_ms:
                        # End time = start time of last character + duration of last character
                        chunk_end_time_ms = char_start_times_ms[-1] + char_durations_ms[-1]
                        chunk_end_time_seconds = chunk_end_time_ms / 1000.0
                        self._cumulative_time += chunk_end_time_seconds
                    else:
                        # Fallback: use the last word's start time (current behavior)
                        self._cumulative_time = word_times[-1][1]
                        logger.warning(
                            "_receive_messages: using fallback timing method - consider investigating alignment data structure"
                        )

    async def _send_keepalive(self):
        """Send a single keepalive message to keep the WebSocket connection alive.

        Only stamps a ``context_id`` once its context-init (carrying
        ``voice_settings``) has been sent. Otherwise the keepalive would be the
        context's first message, with no ``voice_settings``, and ElevenLabs would
        reject the later context-init with a 1008 policy violation. A context-less
        keepalive is sufficient until the context-init is sent.
        """
        if not self._websocket or self._websocket.state is not State.OPEN:
            return

        context_id = self.get_active_audio_context_id()
        if context_id and context_id in self._context_init_sent:
            # The context's voice_settings context-init has been sent, so it's
            # safe to keep that context alive.
            keepalive_message = {"text": "", "context_id": context_id}
        else:
            # No active context, or the active context's context-init hasn't been
            # sent yet. A context-less keepalive keeps the connection alive without
            # opening the context prematurely.
            keepalive_message = {"text": ""}
        await self._websocket.send(json.dumps(keepalive_message))

    async def _send_context_init(self, context_id: str):
        """Open a context, carrying voice settings and pronunciation dictionaries."""
        # Mark the context-init as sent so the keepalive may now target this
        # context_id.
        self._context_init_sent.add(context_id)
        msg: dict[str, Any] = {"text": " ", "context_id": context_id}
        if self._voice_settings:
            msg["voice_settings"] = self._voice_settings
        if self._pronunciation_dictionary_locators:
            msg["pronunciation_dictionary_locators"] = [
                locator.model_dump() for locator in self._pronunciation_dictionary_locators
            ]
        await self._get_websocket().send(json.dumps(msg))

    async def _send_text(self, text: str, context_id: str):
        """Send text to the WebSocket for synthesis."""
        if self._websocket and context_id:
            msg = {"text": text, "context_id": context_id}
            await self._websocket.send(json.dumps(msg))


class ElevenLabsHttpTTSService(TTSService):
    """ElevenLabs HTTP-based TTS service with word timestamps.

    Provides text-to-speech using ElevenLabs' HTTP streaming API for simpler,
    non-WebSocket integration. Suitable for use cases where streaming WebSocket
    connection is not required or desired.
    """

    Settings = ElevenLabsHttpTTSSettings
    _settings: Settings

    @deprecated(
        "`ElevenLabsHttpTTSService.InputParams` is deprecated since 0.0.105 and will be removed "
        "in 2.0.0. Use `ElevenLabsHttpTTSService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Input parameters for ElevenLabs HTTP TTS configuration.

        .. deprecated:: 0.0.105
            Use ``settings=ElevenLabsHttpTTSService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Language to use for synthesis.
            optimize_streaming_latency: Latency optimization level (0-4).
            stability: Voice stability control (0.0 to 1.0).
            similarity_boost: Similarity boost control (0.0 to 1.0).
            style: Style control for voice expression (0.0 to 1.0).
            use_speaker_boost: Whether to use speaker boost enhancement.
            speed: Voice speed control (0.25 to 4.0).
            apply_text_normalization: Text normalization mode ("auto", "on", "off").
            pronunciation_dictionary_locators: List of pronunciation dictionary locators to use.
        """

        language: Language | None = None
        optimize_streaming_latency: int | None = None
        stability: float | None = None
        similarity_boost: float | None = None
        style: float | None = None
        use_speaker_boost: bool | None = None
        speed: float | None = None
        apply_text_normalization: Literal["auto", "on", "off"] | None = None
        pronunciation_dictionary_locators: list[PronunciationDictionaryLocator] | None = None

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str | None = None,
        aiohttp_session: aiohttp.ClientSession,
        model: str | None = None,
        base_url: str = "https://api.elevenlabs.io",
        sample_rate: int | None = None,
        enable_logging: bool | None = None,
        pronunciation_dictionary_locators: list[PronunciationDictionaryLocator] | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        aggregate_sentences: bool | None = None,
        **kwargs,
    ):
        """Initialize the ElevenLabs HTTP TTS service.

        Args:
            api_key: ElevenLabs API key for authentication.
            voice_id: ID of the voice to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=ElevenLabsHttpTTSService.Settings(voice=...)`` instead.
                    Will be removed in 2.0.0.

            aiohttp_session: aiohttp ClientSession for HTTP requests.
            model: TTS model to use (e.g., "eleven_flash_v2_5").

                .. deprecated:: 0.0.105
                    Use ``settings=ElevenLabsHttpTTSService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            base_url: Base URL for ElevenLabs HTTP API.
            sample_rate: Audio sample rate. If None, uses default.
            enable_logging: Whether to enable ElevenLabs server-side logging.
                Set to False for zero retention mode (enterprise only).
            pronunciation_dictionary_locators: List of pronunciation dictionary
                locators to use.

                .. deprecated:: 1.6.0
                    Use the ``text_transforms`` parameter with
                    :func:`pipecat.utils.text.transforms.replace_text`
                    instead. Pronunciation dictionary substitutions can
                    rewrite the spoken words in ways that no longer match
                    the text sent to synthesis, which breaks the
                    alignment-based word-completion tracking used to
                    attribute spoken text back to the conversation context.
                    ``replace_text`` transforms happen client-side, so
                    they're tracked correctly. Will be removed in 2.0.0.

            params: Additional input parameters for voice customization.

                .. deprecated:: 0.0.105
                    Use ``settings=ElevenLabsHttpTTSService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            text_aggregation_mode: How to aggregate incoming text before synthesis.
            aggregate_sentences: Whether to aggregate sentences within the TTSService.

                .. deprecated:: 0.0.104
                    Use ``text_aggregation_mode`` instead.
                    Will be removed in 2.0.0.

            **kwargs: Additional arguments passed to the parent service.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="eleven_flash_v2_5",
            voice=None,
            language=None,
            optimize_streaming_latency=None,
            stability=None,
            similarity_boost=None,
            style=None,
            use_speaker_boost=None,
            speed=None,
            apply_text_normalization=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        # 3. Apply params overrides — only if settings not provided
        _pronunciation_dictionary_locators = pronunciation_dictionary_locators
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.language is not None:
                    default_settings.language = params.language
                if params.optimize_streaming_latency is not None:
                    default_settings.optimize_streaming_latency = params.optimize_streaming_latency
                if params.stability is not None:
                    default_settings.stability = params.stability
                if params.similarity_boost is not None:
                    default_settings.similarity_boost = params.similarity_boost
                if params.style is not None:
                    default_settings.style = params.style
                if params.use_speaker_boost is not None:
                    default_settings.use_speaker_boost = params.use_speaker_boost
                if params.speed is not None:
                    default_settings.speed = params.speed
                if params.apply_text_normalization is not None:
                    default_settings.apply_text_normalization = params.apply_text_normalization
                if _pronunciation_dictionary_locators is None:
                    _pronunciation_dictionary_locators = params.pronunciation_dictionary_locators

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            aggregate_sentences=aggregate_sentences,
            push_text_frames=False,
            push_stop_frames=True,
            push_start_frame=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session
        self._enable_logging = enable_logging

        self._output_format = ""  # initialized in start()
        self._voice_settings = self._set_voice_settings()
        if _pronunciation_dictionary_locators is not None:
            warnings.warn(
                "`pronunciation_dictionary_locators` is deprecated since 1.6.0 and will be "
                "removed in 2.0.0. Use `text_transforms` -> `replace_text` instead. "
                "Pronunciation dictionary substitutions can rewrite the spoken words in "
                "ways that no longer match the text sent to synthesis, which breaks the "
                "alignment-based word-completion tracking used to attribute spoken text "
                "back to the conversation context.",
                DeprecationWarning,
                stacklevel=2,
            )
        self._pronunciation_dictionary_locators = _pronunciation_dictionary_locators

        # Track cumulative time to properly sequence word timestamps across utterances
        self._cumulative_time = 0

        # Store previous text for context within a turn
        self._previous_text = ""

        # Track partial words that span across alignment chunks
        self._partial_word = ""
        self._partial_word_start_time = 0.0

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert pipecat Language to ElevenLabs language code.

        Args:
            language: The language to convert.

        Returns:
            The ElevenLabs-specific language code, or None if not supported.
        """
        return language_to_elevenlabs_language(language)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as ElevenLabs HTTP service supports metrics generation.
        """
        return True

    def _set_voice_settings(self):
        return build_elevenlabs_voice_settings(self._settings)

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta and rebuild voice settings.

        Args:
            delta: A :class:`TTSSettings` (or ``ElevenLabsHttpTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed:
            self._voice_settings = self._set_voice_settings()
        return changed

    def _reset_state(self):
        """Reset internal state variables."""
        self._cumulative_time = 0
        self._previous_text = ""
        self._partial_word = ""
        self._partial_word_start_time = 0.0
        logger.debug(f"{self}: Reset internal state")

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._output_format = output_format_from_sample_rate(self.sample_rate)
        self._reset_state()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, TTSStoppedFrame)):
            # Reset timing on interruption or stop
            self._reset_state()
        elif isinstance(frame, LLMFullResponseEndFrame):
            # End of turn - reset previous text
            self._previous_text = ""

    def calculate_word_times(self, alignment_info: Mapping[str, Any]) -> list[tuple[str, float]]:
        """Calculate word timing from character alignment data.

        This method handles partial words that may span across multiple alignment chunks.

        Args:
            alignment_info: Character timing data from ElevenLabs.

        Returns:
            List of (word, timestamp) pairs for complete words in this chunk.

        Example input data::

            {
                "characters": [" ", "H", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"],
                "character_start_times_seconds": [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "character_end_times_seconds": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }

        Would produce word times (with cumulative_time=0)::

            [("Hello", 0.1), ("world", 0.5)]
        """
        chars = alignment_info.get("characters", [])
        char_start_times = alignment_info.get("character_start_times_seconds", [])

        if not chars or not char_start_times or len(chars) != len(char_start_times):
            logger.warning(
                f"Invalid alignment data: chars={len(chars)}, times={len(char_start_times)}"
            )
            return []

        # Build the words and find their start times
        words = []
        word_start_times = []
        # Start with any partial word from previous chunk
        current_word = self._partial_word
        word_start_time = self._partial_word_start_time if self._partial_word else None

        for i, char in enumerate(chars):
            if char == " ":
                if current_word:  # Only add non-empty words
                    words.append(current_word)
                    word_start_times.append(word_start_time)
                    current_word = ""
                    word_start_time = None
            else:
                if word_start_time is None:  # First character of a new word
                    # Use time of the first character of the word, offset by cumulative time
                    word_start_time = self._cumulative_time + char_start_times[i]
                current_word += char

        # Store any incomplete word at the end of this chunk
        self._partial_word = current_word if current_word else ""
        self._partial_word_start_time = word_start_time if word_start_time is not None else 0.0

        # Create word-time pairs for complete words only
        word_times = list(zip(words, word_start_times))

        return word_times

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using ElevenLabs streaming API with timestamps.

        Makes a request to the ElevenLabs API to generate audio and timing data.
        Tracks the duration of each utterance to ensure correct sequencing.
        Includes previous text as context for better prosody continuity.

        Args:
            text: Text to convert to speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio and control frames containing the synthesized speech.
        """
        # Use the with-timestamps endpoint
        url = f"{self._base_url}/v1/text-to-speech/{self._settings.voice}/stream/with-timestamps"

        model_id = assert_given(self._settings.model)
        payload: dict[str, Any] = {
            "text": text,
            "model_id": model_id,
        }

        # Include previous text as context when the model supports it
        if self._previous_text and model_id not in ELEVENLABS_CONTEXT_UNSUPPORTED_MODELS:
            payload["previous_text"] = self._previous_text

        if self._voice_settings:
            payload["voice_settings"] = self._voice_settings

        if self._pronunciation_dictionary_locators:
            payload["pronunciation_dictionary_locators"] = [
                locator.model_dump() for locator in self._pronunciation_dictionary_locators
            ]

        apply_text_normalization = assert_given(self._settings.apply_text_normalization)
        if apply_text_normalization is not None:
            payload["apply_text_normalization"] = apply_text_normalization

        language_code = elevenlabs_language_code(model_id, assert_given(self._settings.language))
        if language_code:
            payload["language_code"] = language_code

        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }

        # Build query parameters
        params = {
            "output_format": self._output_format,
        }
        optimize_streaming_latency = assert_given(self._settings.optimize_streaming_latency)
        if optimize_streaming_latency is not None:
            params["optimize_streaming_latency"] = str(optimize_streaming_latency)
        if self._enable_logging is not None:
            params["enable_logging"] = str(self._enable_logging).lower()

        try:
            async with self._session.post(
                url, json=payload, headers=headers, params=params
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=f"ElevenLabs API error: {error_text}")
                    return

                await self.start_tts_usage_metrics(text)

                # Track the duration of this utterance based on the last character's end time
                utterance_duration = 0
                alignment_started = False
                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue

                    try:
                        # Parse the JSON object
                        data = json.loads(line_str)

                        # Process audio if present
                        if data and "audio_base64" in data:
                            await self.stop_ttfb_metrics()
                            audio = base64.b64decode(data["audio_base64"])
                            yield TTSAudioRawFrame(
                                audio, self.sample_rate, 1, context_id=context_id
                            )

                        raw_alignment = data and _select_alignment(
                            data,
                            normalized_key="normalized_alignment",
                            alignment_key="alignment",
                            prefer_normalized=bool(self._pronunciation_dictionary_locators),
                        )
                        if raw_alignment:
                            alignment = _strip_utterance_leading_spaces(
                                raw_alignment,
                                (
                                    "characters",
                                    "character_start_times_seconds",
                                    "character_end_times_seconds",
                                ),
                                not alignment_started,
                            )
                            alignment_started = True
                            # Get end time of the last character in this chunk
                            char_end_times = alignment.get("character_end_times_seconds", [])
                            if char_end_times:
                                chunk_end_time = char_end_times[-1]
                                # Update to the longest end time seen so far
                                utterance_duration = max(utterance_duration, chunk_end_time)

                            # Calculate word timestamps
                            word_times = self.calculate_word_times(alignment)
                            if word_times:
                                await self.add_word_timestamps(
                                    word_times,
                                    context_id,
                                    includes_inter_frame_spaces=(
                                        True
                                        if _word_timestamps_include_inter_frame_spaces(
                                            assert_given(self._settings.language)
                                        )
                                        else None
                                    ),
                                )
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON from stream: {e}")
                        continue
                    except Exception as e:
                        yield ErrorFrame(error=f"Unknown error occurred: {e}")
                        continue

                # After processing all chunks, emit any remaining partial word
                # since this is the end of the utterance
                if self._partial_word:
                    final_word_time = [(self._partial_word, self._partial_word_start_time)]
                    await self.add_word_timestamps(
                        final_word_time,
                        context_id,
                        includes_inter_frame_spaces=(
                            True
                            if _word_timestamps_include_inter_frame_spaces(
                                assert_given(self._settings.language)
                            )
                            else None
                        ),
                    )
                    self._partial_word = ""
                    self._partial_word_start_time = 0.0

                # After processing all chunks, add the total utterance duration
                # to the cumulative time to ensure next utterance starts after this one
                if utterance_duration > 0:
                    self._cumulative_time += utterance_duration

                # Append the current text to previous_text for context continuity
                # Only add a space if there's already text
                if self._previous_text:
                    self._previous_text += " " + text
                else:
                    self._previous_text = text

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()
