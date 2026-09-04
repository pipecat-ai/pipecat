#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ElevenLabs Text-to-Dialogue text-to-speech service implementation.

This module provides a WebSocket TTS service for ElevenLabs' multi-context
Text-to-Dialogue endpoint, the only way to reach Eleven v3 models.
"""

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Union

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.elevenlabs.tts_base import (
    ElevenLabsTTSBase,
    ElevenLabsTTSSettingsBase,
    _select_alignment,
    _strip_utterance_leading_spaces,
    _word_timestamps_include_inter_frame_spaces,
    calculate_word_times,
    elevenlabs_language_code,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

# Text-to-Dialogue rejects a keepalive that doesn't name a registered context,
# and between bot turns there isn't one. This context is registered at connect
# and never receives text.
_KEEPALIVE_CONTEXT_ID = "pipecat-keepalive"


@dataclass
class ElevenLabsDialogueTTSSettings(ElevenLabsTTSSettingsBase):
    """Settings for ElevenLabsDialogueTTSService.

    Parameters:
        stability: Voice stability control (0.0 to 1.0), the only voice setting
            Text-to-Dialogue reads. ElevenLabs documents three reference points:
            0.0 is the most expressive but can hallucinate, 0.5 stays closest to
            the reference recording, and 1.0 is the most consistent but least
            responsive to audio tags. Values in between are accepted.
    """

    stability: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    #: Fields in the WS URL — changing any of these requires a reconnect. Voice
    #: is absent because Text-to-Dialogue registers voices per context rather
    #: than in the URL, so a voice change only needs a new context.
    URL_FIELDS: ClassVar[frozenset[str]] = frozenset({"model", "language"})

    #: Fields carried in the per-context registration message — changing these
    #: requires closing the current audio context so the next one picks them up.
    VOICE_SETTINGS_FIELDS: ClassVar[frozenset[str]] = frozenset({"voice", "stability"})


def build_elevenlabs_ttd_voice_settings(
    settings: Union[dict[str, Any], "TTSSettings"],
) -> dict[str, float] | None:
    """Build a Text-to-Dialogue voice settings dict.

    ``stability`` is the only setting Text-to-Dialogue reads. The endpoint
    silently ignores the other voice settings the text-to-speech endpoint
    accepts, so sending them would suggest an effect they don't have.

    Args:
        settings: A settings object or dict to read ``stability`` from.

    Returns:
        The voice settings dict, or None if stability is unset.
    """
    stability = (
        getattr(settings, "stability", None)
        if isinstance(settings, TTSSettings)
        else settings.get("stability")
    )
    if stability is None or stability is NOT_GIVEN:
        return None
    return {"stability": stability}


def _normalize_ttd_alignment(alignment: Mapping[str, Any]) -> dict[str, Any]:
    """Convert Text-to-Dialogue alignment keys to the casing shared helpers expect.

    Text-to-Dialogue emits snake_case alignment fields where the text-to-speech
    endpoint emits camelCase, and
    :func:`~pipecat.services.elevenlabs.tts_base.calculate_word_times` expects
    the
    latter.
    """
    return {
        "chars": alignment.get("chars") or [],
        "charStartTimesMs": alignment.get("char_start_times_ms") or [],
        "charDurationsMs": alignment.get("char_durations_ms") or [],
    }


@dataclass
class _DialogueContext:
    """Server-side state for one Text-to-Dialogue context.

    Tracked separately from the audio context the base class owns, because the
    two have different lifetimes: a Text-to-Dialogue context outlives its audio
    context while draining after an interruption, and the keepalive context
    never has one at all.

    Parameters:
        registered: Whether the server will accept messages naming this context.
            It rejects any message for a context that has no ``voices``
            registration, and a closed context counts as unregistered.
        new_turn_pending: Whether the next input should reset prosody.
        alignment_started: Whether an alignment message has been seen, so
            leading spaces are only stripped from the first one.
    """

    registered: bool = True
    new_turn_pending: bool = True
    alignment_started: bool = False


class ElevenLabsDialogueTTSService(ElevenLabsTTSBase):
    """ElevenLabs Text-to-Dialogue WebSocket TTS service for Eleven v3 models.

    Uses the multi-context Text-to-Dialogue endpoint, which is the only way to
    reach ``eleven_v3`` models. Use
    :class:`~pipecat.services.elevenlabs.tts.ElevenLabsTTSService` for Flash,
    Turbo, and Multilingual models — it has lower latency and a fuller set of
    voice controls.

    Eleven v3 performs inline audio tags such as ``[laughs]`` and ``[excited]``.
    They come back as spoken characters in the alignment, so they reach the LLM
    context as text unless a text filter removes them.
    """

    Settings = ElevenLabsDialogueTTSSettings
    _settings: Settings

    CONNECTION_NAME = "ElevenLabs Text-to-Dialogue"

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.elevenlabs.io",
        sample_rate: int | None = None,
        enable_logging: bool | None = None,
        seed: int | None = None,
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        **kwargs,
    ):
        """Initialize the ElevenLabs Text-to-Dialogue TTS service.

        Args:
            api_key: ElevenLabs API key for authentication.
            url: WebSocket URL for the ElevenLabs API.
            sample_rate: Audio sample rate. If None, uses default.
            enable_logging: Whether to enable ElevenLabs server-side logging.
            seed: Seed for reproducible generation.
            settings: Runtime-updatable settings.
            text_aggregation_mode: How to aggregate incoming text before
                synthesis. Only :attr:`TextAggregationMode.SENTENCE` is
                supported; any other value is ignored with a warning.
            **kwargs: Additional arguments passed to the parent service.
        """
        default_settings = self.Settings(
            model="eleven_v3_conversational",
            voice=None,
            language=None,
            stability=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            url=url,
            enable_logging=enable_logging,
            push_text_frames=False,
            push_stop_frames=False,
            pause_frame_processing=True,
            # Consecutive inputs are concatenated verbatim, so without a
            # trailing space one input's last word merges into the next's first.
            append_trailing_space=True,
            # Verbatim concatenation plus the trailing space above means a
            # token-sized input would insert a space mid-word.
            text_aggregation_mode=TextAggregationMode.SENTENCE,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        model = default_settings.model
        if isinstance(model, str) and not model.startswith("eleven_v3"):
            logger.warning(
                f"{self}: Text-to-Dialogue requires an eleven_v3 model, got {model!r}. "
                "Use ElevenLabsTTSService for Flash, Turbo, and Multilingual models."
            )

        if (
            text_aggregation_mode is not None
            and text_aggregation_mode is not TextAggregationMode.SENTENCE
        ):
            logger.warning(
                f"{self}: Text-to-Dialogue aggregates by sentence; ignoring "
                f"text_aggregation_mode={text_aggregation_mode}."
            )

        self._seed = seed

        self._contexts: dict[str, _DialogueContext] = {}

    def _set_voice_settings(self):
        return build_elevenlabs_ttd_voice_settings(self._settings)

    async def flush_audio(self, context_id: str | None = None):
        """Force generation of any text still held in the server-side buffer.

        Text accumulates until roughly 40 characters and 8 words, at which
        point the server starts generating on its own, so a turn is usually
        under way before the end-of-turn flush covers what is left.

        Args:
            context_id: The specific context to flush. If None, falls back to the
                currently active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket:
            return
        context = self._contexts.get(flush_id)
        if not context or not context.registered:
            return

        logger.trace(f"{self}: flushing context {flush_id}")
        await self._websocket.send(json.dumps({"context_id": flush_id, "flush": True}))

    def _build_websocket_url(self) -> str:
        model = self._settings.model
        url = (
            f"{self._url}/v1/text-to-dialogue/multi-stream-input"
            f"?model_id={model}&output_format={self._output_format}&sync_alignment=true"
        )

        if self._enable_logging is not None:
            url += f"&enable_logging={str(self._enable_logging).lower()}"

        if self._seed is not None:
            url += f"&seed={self._seed}"

        language_code = elevenlabs_language_code(
            assert_given(model), assert_given(self._settings.language)
        )
        if language_code:
            url += f"&language_code={language_code}"

        return url

    async def _on_websocket_connected(self):
        await self._register_keepalive_context()

    def _clear_connection_state(self):
        self._contexts.clear()

    async def _close_context(self, context_id: str):
        """Close a server-side context to free its slot.

        A close generates whatever text the context still holds, streams that
        audio, and ends with an ``is_final``. A context closed on an
        interruption therefore keeps producing audio for a while; removing the
        audio context discards it.
        """
        context = self._contexts.get(context_id)
        if not context or not context.registered:
            return
        # The entry outlives the close so that alignment still arriving for this
        # context is handled with the right state; `is_final` removes it.
        context.registered = False
        if not self._websocket:
            return
        logger.trace(f"{self}: Closing context {context_id}")
        try:
            await self._websocket.send(
                json.dumps({"context_id": context_id, "close_context": True})
            )
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

    def _reset_alignment_state(self, context_id: str):
        super()._reset_alignment_state(context_id)
        self._contexts.pop(context_id, None)

    async def on_turn_context_completed(self):
        """Close the turn's context, which generates any text still buffered in it."""
        context_id = self._turn_context_id
        await super().on_turn_context_completed()
        if context_id:
            await self._close_context(context_id)

    async def _receive_messages(self):
        """Handle incoming WebSocket messages from ElevenLabs Text-to-Dialogue."""
        async for message in self._get_websocket():
            await self._handle_message(json.loads(message))

    async def _handle_message(self, msg: Mapping[str, Any]):
        """Dispatch a single Text-to-Dialogue server message."""
        if msg.get("error"):
            await self.push_error(
                error_msg=f"ElevenLabs Text-to-Dialogue error: {msg.get('message', msg)}"
            )
            return

        received_ctx_id: str | None = msg.get("context_id")
        if not received_ctx_id:
            logger.debug(f"{self}: ignoring message with no context: {msg}")
            return

        if received_ctx_id == _KEEPALIVE_CONTEXT_ID:
            return

        # `is_final` marks the end of a context. `is_final_audio_for_turn`
        # arrives each time the server finishes a batch of generation, several
        # times within a turn, so it can't be used to end one.
        if msg.get("is_final") is True:
            logger.debug(f"Received final message for context {received_ctx_id}")
            self._contexts.pop(received_ctx_id, None)
            if self.audio_context_available(received_ctx_id):
                await self.append_to_audio_context(
                    received_ctx_id, TTSStoppedFrame(context_id=received_ctx_id)
                )
                await self.remove_audio_context(received_ctx_id)
            return

        # An interrupted context keeps producing audio until the server finishes
        # the batch it had started. That audio is discarded, and its alignment
        # must not advance the word clock, which by now belongs to the next turn.
        context = self._contexts.get(received_ctx_id)
        if not context or not self.audio_context_available(received_ctx_id):
            return

        if msg.get("audio"):
            audio = base64.b64decode(msg["audio"])
            frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=received_ctx_id)
            await self.append_to_audio_context(received_ctx_id, frame)

        raw_alignment = _select_alignment(
            msg,
            normalized_key="normalized_alignment",
            alignment_key="alignment",
            prefer_normalized=False,
        )
        if raw_alignment:
            alignment = _strip_utterance_leading_spaces(
                _normalize_ttd_alignment(raw_alignment),
                ("chars", "charStartTimesMs", "charDurationsMs"),
                not context.alignment_started,
            )
            context.alignment_started = True
            word_times, self._partial_word, self._partial_word_start_time = calculate_word_times(
                alignment,
                self._cumulative_time,
                self._partial_word,
                self._partial_word_start_time,
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

            # Alignment times restart at zero in every message, so the clock
            # advances by each chunk's span. It has to advance for chunks that
            # complete no word too — trailing punctuation arrives on its own —
            # or every later word is timestamped early by the shortfall.
            char_start_times_ms = alignment.get("charStartTimesMs", [])
            char_durations_ms = alignment.get("charDurationsMs", [])

            if char_start_times_ms and char_durations_ms:
                chunk_end_time_ms = char_start_times_ms[-1] + char_durations_ms[-1]
                self._cumulative_time += chunk_end_time_ms / 1000.0
            elif word_times:
                self._cumulative_time = word_times[-1][1]

    async def _register_keepalive_context(self):
        """Open a context that exists only to keep the connection alive.

        ElevenLabs closes the connection after 20 seconds without a message, and
        rejects a keepalive that doesn't name a registered context. Between bot
        turns there is no such context, so this one is registered up front and
        never receives text.
        """
        if not self._websocket:
            return

        voice = self._settings.voice
        if voice is None or voice is NOT_GIVEN:
            logger.debug(f"{self}: no voice set, connection will idle out after 20s")
            return

        await self._websocket.send(
            json.dumps({"context_id": _KEEPALIVE_CONTEXT_ID, "voices": [voice]})
        )
        self._contexts[_KEEPALIVE_CONTEXT_ID] = _DialogueContext()

    async def _send_keepalive(self):
        """Reset the server's inactivity timer on the connection."""
        if not self._websocket or self._websocket.state is not State.OPEN:
            return

        context = self._contexts.get(_KEEPALIVE_CONTEXT_ID)
        if not context or not context.registered:
            return

        await self._websocket.send(
            json.dumps({"context_id": _KEEPALIVE_CONTEXT_ID, "keep_alive": True})
        )

    async def _send_context_init(self, context_id: str):
        """Open a context, registering the voice it speaks with.

        Every context must open with a ``voices`` registration; ElevenLabs
        closes the socket otherwise.
        """
        msg: dict[str, Any] = {
            "context_id": context_id,
            "voices": [assert_given(self._settings.voice)],
        }
        if self._voice_settings:
            msg["voice_settings"] = self._voice_settings
        await self._get_websocket().send(json.dumps(msg))
        self._contexts[context_id] = _DialogueContext()

    async def _send_text(self, text: str, context_id: str):
        """Send text to the WebSocket for synthesis."""
        context = self._contexts.get(context_id)
        if not self._websocket or not context or not context.registered:
            return

        new_turn = context.new_turn_pending
        context.new_turn_pending = False
        logger.trace(f"{self}: input for context {context_id} (new_turn={new_turn}): {text!r}")
        msg = {
            "context_id": context_id,
            "inputs": [
                {
                    "text": text,
                    "voice_id": assert_given(self._settings.voice),
                    "new_turn": new_turn,
                }
            ],
        }
        await self._websocket.send(json.dumps(msg))
